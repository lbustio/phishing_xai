from __future__ import annotations

import importlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.experiment import CV_SETTINGS, METRICS, PRIMARY_METRIC
from src.utils.checkpoint import ExperimentCheckpoint
from src.utils.io_utils import atomic_write_csv, atomic_write_json
from src.utils.logging_setup import log_subsection

import warnings
from sklearn.exceptions import ConvergenceWarning

logger = logging.getLogger("phishing_xai.trainer")


@dataclass
class EvaluationArtifacts:
    model_path: Path
    metadata_path: Path
    nested_details_path: Path


def _build_estimator(classifier_config: dict):
    module = importlib.import_module(classifier_config["module"])
    estimator_cls = getattr(module, classifier_config["class_name"])
    estimator = estimator_cls(**classifier_config.get("init_params", {}))

    steps: list[tuple[str, Any]] = []
    if classifier_config.get("use_scaler", False):
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict[str, float]:
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba[:, 1])) if y_proba is not None else float("nan"),
    }


def _aggregate_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    for metric_name in METRICS:
        values = [fold[metric_name] for fold in fold_metrics if not np.isnan(fold[metric_name])]
        if values:
            aggregated[metric_name] = float(np.mean(values))
            aggregated[f"{metric_name}_std"] = float(np.std(values))
        else:
            aggregated[metric_name] = float("nan")
            aggregated[f"{metric_name}_std"] = float("nan")
    return aggregated


def _prettify_params(params: dict[str, Any]) -> dict[str, Any]:
    pretty: dict[str, Any] = {}
    for key, value in params.items():
        pretty[key.replace("classifier__", "")] = value
    return pretty


def _summarize_grid(grid_params: dict[str, list[Any]]) -> str:
    parts = []
    for key, values in grid_params.items():
        clean_key = key.replace("classifier__", "")
        parts.append(f"{clean_key}: {len(values)} options")
    return ", ".join(parts)


def _fmt_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def evaluate_classifier_with_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    embedding_id: str,
    classifier_name: str,
    classifier_config: dict,
    artifacts: EvaluationArtifacts,
) -> dict[str, Any]:
    log_subsection(logger, f"CLASSIFIER EVALUATION: {classifier_name}")
    logger.info("[EVAL] Embedding under analysis: %s", embedding_id)
    logger.info("[EVAL] Objective: internal hyperparameter tuning plus external robust validation.")
    logger.info("[EVAL] Classifier rationale: %s", classifier_config.get("note", "No description available."))
    logger.info("[EVAL] Search space summary: %s", _summarize_grid(classifier_config["grid_params"]))
    logger.debug("[EVAL] Full hyperparameter grid: %s", classifier_config["grid_params"])

    outer_cv = StratifiedKFold(
        n_splits=CV_SETTINGS["outer_folds"],
        shuffle=CV_SETTINGS["shuffle"],
        random_state=CV_SETTINGS["random_state"],
    )
    inner_cv = StratifiedKFold(
        n_splits=CV_SETTINGS["inner_folds"],
        shuffle=CV_SETTINGS["shuffle"],
        random_state=CV_SETTINGS["random_state"],
    )

    fold_details: list[dict[str, Any]] = []
    best_params_counter: Counter[str] = Counter()

    for fold_index, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        logger.info(
            "[CV] Fold %s/%s. Inner tuning samples: %s | External validation samples: %s.",
            fold_index,
            CV_SETTINGS["outer_folds"],
            len(train_idx),
            len(test_idx),
        )

        estimator = _build_estimator(classifier_config)
        search = GridSearchCV(
            estimator=estimator,
            param_grid=classifier_config["grid_params"],
            scoring=PRIMARY_METRIC,
            cv=inner_cv,
            n_jobs=CV_SETTINGS["n_jobs"],
            refit=True,
            verbose=0,
            error_score="raise",
        )
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            search.fit(X[train_idx], y[train_idx])

        convergence_warnings = [
            warning for warning in caught_warnings
            if issubclass(warning.category, ConvergenceWarning)
        ]
        
        if convergence_warnings:
            logger.warning(
                "[WARN] Fold %s produced %s convergence warning(s). "
                "This usually means the optimizer needed more iterations.",
                fold_index,
                len(convergence_warnings),
            )

        best_model = search.best_estimator_
        y_pred = best_model.predict(X[test_idx])
        y_proba = best_model.predict_proba(X[test_idx]) if hasattr(best_model, "predict_proba") else None
        fold_metrics = _compute_metrics(y[test_idx], y_pred, y_proba)

        pretty_best_params = _prettify_params(search.best_params_)
        best_params_serialized = json.dumps(pretty_best_params, sort_keys=True)
        best_params_counter[best_params_serialized] += 1

        logger.info(
            "[CV] Best configuration selected inside fold %s: %s.",
            fold_index,
            pretty_best_params,
        )
        logger.info(
            "[CV] External validation for fold %s -> F1-macro: %.4f | Accuracy: %.4f | ROC-AUC: %.4f.",
            fold_index,
            fold_metrics["f1_macro"],
            fold_metrics["accuracy"],
            fold_metrics["roc_auc"],
        )

        fold_details.append(
            {
                "fold_index": fold_index,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "best_params": pretty_best_params,
                "best_inner_score": float(search.best_score_),
                "metrics": fold_metrics,
            }
        )

    aggregated_metrics = _aggregate_fold_metrics([fold["metrics"] for fold in fold_details])
    logger.info(
        "[EVAL] Aggregated nested-CV result for '%s' + '%s': F1-macro %.4f (+/- %.4f) | Accuracy %.4f | ROC-AUC %.4f.",
        embedding_id,
        classifier_name,
        aggregated_metrics["f1_macro"],
        aggregated_metrics["f1_macro_std"],
        aggregated_metrics["accuracy"],
        aggregated_metrics["roc_auc"],
    )

    logger.info("[MODEL] Refitting the hyperparameter search on the full dataset.")
    final_estimator = _build_estimator(classifier_config)
    final_search = GridSearchCV(
        estimator=final_estimator,
        param_grid=classifier_config["grid_params"],
        scoring=PRIMARY_METRIC,
        cv=inner_cv,
        n_jobs=CV_SETTINGS["n_jobs"],
        refit=True,
        verbose=0,
        error_score="raise",
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        final_search.fit(X, y)
    
    final_convergence_warnings = [
        warning for warning in caught_warnings
        if issubclass(warning.category, ConvergenceWarning)
    ]
    
    if final_convergence_warnings:
        logger.warning(
            "[WARN] Full-data refit produced %s convergence warning(s).",
            len(final_convergence_warnings),
        )


    full_refit_params = _prettify_params(final_search.best_params_)
    logger.info("[MODEL] Best configuration after full-data refit: %s.", full_refit_params)

    final_metadata = {
        "embedding": embedding_id,
        "classifier": classifier_name,
        "metrics": aggregated_metrics,
        "selected_params_full_refit": full_refit_params,
        "selected_params_frequency_in_outer_folds": {
            key: value for key, value in best_params_counter.most_common()
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    joblib.dump(final_search.best_estimator_, artifacts.model_path)
    atomic_write_json(artifacts.metadata_path, final_metadata)
    atomic_write_json(
        artifacts.nested_details_path,
        {
            "embedding": embedding_id,
            "classifier": classifier_name,
            "fold_details": fold_details,
            "aggregated_metrics": aggregated_metrics,
            "full_refit_best_params": full_refit_params,
        },
    )

    logger.info("[MODEL] Artifacts saved for '%s' + '%s'.", embedding_id, classifier_name)

    return {
        "metrics": aggregated_metrics,
        "best_params": full_refit_params,
        "model_path": str(artifacts.model_path),
        "metadata_path": str(artifacts.metadata_path),
        "nested_details_path": str(artifacts.nested_details_path),
    }


def append_result_row(
    *,
    csv_path: Path,
    run_id: str,
    dataset_fingerprint: str,
    embedding_id: str,
    embedding_paradigm: str,
    embedding_rationale: str,
    classifier_name: str,
    metrics: dict[str, float],
    best_params: dict[str, Any],
) -> None:
    row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_fingerprint": dataset_fingerprint,
        "embedding": embedding_id,
        "embedding_paradigm": embedding_paradigm,
        "embedding_rationale": embedding_rationale,
        "classifier": classifier_name,
        **{metric: metrics.get(metric) for metric in metrics},
        "best_params": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
    }

    new_row = pd.DataFrame([row])
    if csv_path.exists():
        current = pd.read_csv(csv_path)
        required_columns = {"dataset_fingerprint", "embedding", "classifier"}
        if required_columns.issubset(current.columns):
            dedup_mask = ~(
                (current["dataset_fingerprint"] == dataset_fingerprint)
                & (current["embedding"] == embedding_id)
                & (current["classifier"] == classifier_name)
            )
            current = current.loc[dedup_mask].copy()
        combined = pd.concat([current, new_row], ignore_index=True)
    else:
        combined = new_row

    atomic_write_csv(combined, csv_path)


def describe_results_table(csv_path: Path) -> str:
    if not csv_path.exists():
        return "No cumulative results table exists yet."

    df = pd.read_csv(csv_path)
    if df.empty:
        return "The cumulative results table is empty."

    required_columns = {"embedding", "classifier", PRIMARY_METRIC}
    if not required_columns.issubset(df.columns):
        return "A previous results table exists, but it does not follow the current schema."

    pivot = df.pivot_table(
        index="embedding",
        columns="classifier",
        values=PRIMARY_METRIC,
        aggfunc="max",
    ).round(4)

    lines = [
        "F1-macro summary table by embedding and classifier:",
        pivot.to_string(),
    ]
    best_row = df.loc[df[PRIMARY_METRIC].idxmax()]
    lines.append(
        "Best recorded combination so far: "
        f"[{best_row['embedding']}] + [{best_row['classifier']}] "
        f"with F1-macro={best_row[PRIMARY_METRIC]:.4f}."
    )
    return "\n".join(lines)


def evaluate_all_classifiers(
    X: np.ndarray,
    y: list[int],
    *,
    embedding_id: str,
    embedding_config: dict,
    classifiers_grid: dict[str, dict],
    checkpoint: ExperimentCheckpoint,
    run_id: str,
    dataset_fingerprint: str,
    results_csv: Path,
    model_registry_file: Path,
    nested_cv_details_dir: Path,
) -> dict[str, Any]:
    best_local: dict[str, Any] | None = None
    trained_model_registry = {}
    if model_registry_file.exists():
        with open(model_registry_file, "r", encoding="utf-8") as handle:
            trained_model_registry = json.load(handle)

    y_array = np.asarray(y)
    classifier_names = list(classifiers_grid.keys())
    total = len(classifier_names)

    logger.info("[EVAL] Starting classifier sweep for embedding '%s'.", embedding_id)
    logger.info("[EVAL] Embedding family: %s.", embedding_config["paradigm"])
    logger.info("[PROGRESS] Total classifiers to evaluate: %s.", total)

    completed = 0
    elapsed_times: list[float] = []

    for classifier_name, classifier_config in classifiers_grid.items():
        t_start = datetime.now()

        if checkpoint.is_done(embedding_id, classifier_name):
            logger.info(
                "[CHECKPOINT] Skipping '%s' + '%s' because this combination is already stored in the checkpoint.",
                embedding_id,
                classifier_name,
            )
            stored_result = checkpoint.get_result(embedding_id, classifier_name)
            if stored_result is not None and (
                best_local is None
                or stored_result["metrics"].get(PRIMARY_METRIC, -1.0)
                > best_local["metrics"].get(PRIMARY_METRIC, -1.0)
            ):
                best_local = {
                    "classifier": classifier_name,
                    **stored_result,
                }
            completed += 1
            pct_done = completed / total * 100
            logger.info(
                "[PROGRESS] %s/%s (%.1f%% completado | %.1f%% restante) — '%s' omitido por checkpoint.",
                completed, total, pct_done, 100 - pct_done, classifier_name,
            )
            continue

        safe_embedding = embedding_id.replace("/", "__")
        artifacts = EvaluationArtifacts(
            model_path=model_registry_file.parent / f"{safe_embedding}__{classifier_name}.joblib",
            metadata_path=model_registry_file.parent / f"{safe_embedding}__{classifier_name}.json",
            nested_details_path=nested_cv_details_dir / f"{safe_embedding}__{classifier_name}.json",
        )

        result = evaluate_classifier_with_nested_cv(
            X,
            y_array,
            embedding_id=embedding_id,
            classifier_name=classifier_name,
            classifier_config=classifier_config,
            artifacts=artifacts,
        )

        elapsed = (datetime.now() - t_start).total_seconds()
        elapsed_times.append(elapsed)
        completed += 1

        pct_done = completed / total * 100
        remaining = total - completed
        avg_time = sum(elapsed_times) / len(elapsed_times)
        eta_seconds = avg_time * remaining
        eta_dt = datetime.now() + __import__("datetime").timedelta(seconds=eta_seconds)

        logger.info(
            "[PROGRESS] %s/%s (%.1f%% completado | %.1f%% restante) — '%s' terminado en %.1fs.",
            completed, total, pct_done, 100 - pct_done, classifier_name, elapsed,
        )
        if remaining > 0:
            logger.info(
                "[PROGRESS] Tiempo restante estimado: %s | Hora estimada de finalizacion: %s.",
                _fmt_seconds(eta_seconds),
                eta_dt.strftime("%H:%M:%S"),
            )

        checkpoint.record_result(embedding_id, classifier_name, result)
        logger.info(
            "[CHECKPOINT] Combination '%s' + '%s' stored successfully.",
            embedding_id,
            classifier_name,
        )

        append_result_row(
            csv_path=results_csv,
            run_id=run_id,
            dataset_fingerprint=dataset_fingerprint,
            embedding_id=embedding_id,
            embedding_paradigm=embedding_config["paradigm"],
            embedding_rationale=embedding_config["rationale"],
            classifier_name=classifier_name,
            metrics=result["metrics"],
            best_params=result["best_params"],
        )
        logger.info(
            "[SUMMARY] Results table updated with '%s' + '%s'.",
            embedding_id,
            classifier_name,
        )

        registry_key = f"{embedding_id}::{classifier_name}"
        trained_model_registry[registry_key] = {
            "model_path": result["model_path"],
            "metadata_path": result["metadata_path"],
        }
        atomic_write_json(model_registry_file, trained_model_registry)
        logger.info(
            "[MODEL] Model registry updated for '%s' + '%s'.",
            embedding_id,
            classifier_name,
        )

        if best_local is None or result["metrics"][PRIMARY_METRIC] > best_local["metrics"][PRIMARY_METRIC]:
            best_local = {
                "classifier": classifier_name,
                **result,
            }

    if best_local is not None:
        logger.info(
            "[SUMMARY] Best classifier for embedding '%s': '%s' with F1-macro %.4f.",
            embedding_id,
            best_local["classifier"],
            best_local["metrics"][PRIMARY_METRIC],
        )
        return best_local

    return {}
