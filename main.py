from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from config.experiment import CLASSIFIERS_GRID, EMBEDDING_MODELS, XAI_CONFIG, get_embedding_config, PRIMARY_METRIC
from config.paths import CHECKPOINTS_DIR
from src.utils.io_utils import stable_text_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Pipeline principal para clasificar asuntos cortos, evaluar embeddings y "
            "clasificadores, y generar artefactos XAI.\n\n"
            "Uso recomendado:\n"
            "  1. Corre todo el experimento: entrenamiento + seleccion + XAI.\n"
            "  2. Restringe embeddings o clasificadores cuando quieras pruebas rapidas.\n"
            "  3. Usa --embeddings-only si solo quieres llenar o refrescar la cache.\n"
            "  4. Usa --only-xai cuando ya existe un run previo con best_model.joblib."
        ),
        epilog=(
            "Ejemplos utiles:\n"
            "  python main.py\n"
            "  python main.py --embeddings-only\n"
            "  python main.py --embeddings sentence-transformers/all-mpnet-base-v2 BAAI/bge-m3\n"
            "  python main.py --classifiers logistic_regression\n"
            "  python main.py --run-id 20260329_103615 --only-xai\n\n"
            "Ayuda extendida disponible tambien con: --help, -h, --h, -help"
        ),
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=30, width=110),
        add_help=False,
    )
    parser.add_argument(
        "-h", "--help", "--h", "-help",
        action="help",
        help="Show this practical help message with workflows, examples, and argument guidance.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=False,
        default="data/pop_dataset_Full(Tiltan).csv",
        help=(
            "Path to the CSV dataset. Use this when you want to swap datasets without editing code. "
            "The file must contain one subject-like column and one label-like column."
        ),
    )
    parser.add_argument(
        "--subject-column",
        type=str,
        default=None,
        help=(
            "Explicit name of the text column to use as subject. "
            "Set this only if auto-detection fails or your CSV uses a non-standard header."
        ),
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help=(
            "Explicit name of the label column. "
            "Useful when the dataset uses a custom label header and you do not want auto-detection."
        ),
    )
    parser.add_argument(
        "--embeddings",
        nargs="+",
        default=None,
        help=(
            "Subset of embedding IDs to evaluate. "
            "If omitted, the pipeline runs every embedding defined in config.experiment."
        ),
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=None,
        help=(
            "Subset of classifier IDs to evaluate. "
            "Use this to shorten experiments or to compare a single downstream model."
        ),
    )
    parser.add_argument(
        "--skip-xai",
        action="store_true",
        help=(
            "Train and select the best model, but stop before generating explanation artefacts. "
            "Useful for fast benchmarking runs."
        ),
    )
    parser.add_argument(
        "--only-xai",
        action="store_true",
        help=(
            "Skip training and reuse an existing run directory to generate XAI only. "
            "Requires --run-id and a previously saved best model."
        ),
    )
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help=(
            "Compute or refresh the embedding cache only. "
            "No classifier training, no model selection, and no XAI will run."
        ),
    )
    parser.add_argument(
        "--n-xai",
        type=int,
        default=XAI_CONFIG["n_xai_examples"],
        help=(
            "Number of representative subjects to explain during the XAI phase. "
            "Ignored when --skip-xai or --embeddings-only is active."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Run identifier to reuse an existing results directory. "
            "Most important when running --only-xai against a previous experiment."
        ),
    )
    return parser.parse_args()


def _build_scope(args: argparse.Namespace) -> tuple[dict[str, dict], dict[str, dict]]:
    selected_embeddings = EMBEDDING_MODELS
    if args.embeddings:
        invalid = [item for item in args.embeddings if item not in EMBEDDING_MODELS]
        if invalid:
            raise ValueError(f"Unknown embeddings: {invalid}")
        selected_embeddings = {key: EMBEDDING_MODELS[key] for key in args.embeddings}

    selected_classifiers = CLASSIFIERS_GRID
    if args.classifiers:
        invalid = [item for item in args.classifiers if item not in CLASSIFIERS_GRID]
        if invalid:
            raise ValueError(f"Unknown classifiers: {invalid}")
        selected_classifiers = {key: CLASSIFIERS_GRID[key] for key in args.classifiers}

    return selected_embeddings, selected_classifiers


def _build_checkpoint_path(dataset_fingerprint: str, embeddings: dict[str, dict], classifiers: dict[str, dict]) -> Path:
    emb_slug = stable_text_hash("||".join(sorted(embeddings.keys())))[:12]
    clf_slug = stable_text_hash("||".join(sorted(classifiers.keys())))[:12]
    return CHECKPOINTS_DIR / f"checkpoint__{dataset_fingerprint}__{emb_slug}__{clf_slug}.json"


def _load_or_compute_embeddings(embedding_id: str, embedding_store: EmbeddingStore, texts: list[str], device, log):
    from src.embeddings.factory import get_embedder
    from src.utils.device import should_skip_model

    cached = embedding_store.load(embedding_id)
    if cached is not None:
        if cached.shape[0] == len(texts):
            log.info("[CACHE] Reusing cached embeddings for '%s'.", embedding_id)
            return cached
        log.warning(
            "[CACHE] Cached embeddings for '%s' have %s rows, but the current dataset has %s. Recomputing.",
            embedding_id,
            cached.shape[0],
            len(texts),
        )

    embedding_config = get_embedding_config(embedding_id)
    skip, reason = should_skip_model(embedding_config, device)
    if skip:
        log.warning("[EMBED] Embedding '%s' will be skipped because %s.", embedding_id, reason)
        return None

    log.info("[EMBED] No reusable cache found for '%s'. Embeddings will be computed now.", embedding_id)
    embedder = get_embedder(
        name=embedding_id,
        model_config=embedding_config,
        device=device,
        checkpoint_dir=embedding_store.get_model_dir(embedding_id),
    )
    try:
        embeddings = embedder.encode(texts)
    finally:
        embedder.release_resources()

    embedding_store.save(
        embedding_id,
        embeddings,
        {
            "embedding_id": embedding_id,
            "num_samples": len(texts),
            "shape": list(embeddings.shape),
        },
    )
    log.info("[CACHE] Persistent embedding cache updated for '%s'.", embedding_id)
    return embeddings


def _write_manifest(artifacts, args: argparse.Namespace, dataset_bundle, device, embeddings: dict, classifiers: dict) -> None:
    from src.utils.io_utils import atomic_write_json

    atomic_write_json(
        artifacts.manifest_file,
        {
            "run_id": artifacts.run_id,
            "data_path": str(Path(args.data).resolve()),
            "dataset_fingerprint": dataset_bundle.dataset_fingerprint,
            "selected_embeddings": list(embeddings.keys()),
            "selected_classifiers": list(classifiers.keys()),
            "device": str(device),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def _select_global_best(checkpoint: ExperimentCheckpoint) -> dict | None:
    return checkpoint.get_best_result()


def _only_xai_mode(args: argparse.Namespace, artifacts, device, log) -> None:
    from src.data_loader import load_dataset
    from src.xai_runner import run_xai

    if not artifacts.best_model_meta_file.exists() or not artifacts.best_model_file.exists():
        raise FileNotFoundError(
            "No persisted best model exists in this run directory. Execute the evaluation pipeline first."
        )

    from src.utils.io_utils import read_json

    log_section(log, "XAI-ONLY MODE")
    metadata = read_json(artifacts.best_model_meta_file, {})
    dataset_bundle = load_dataset(
        Path(args.data),
        subject_column=args.subject_column,
        label_column=args.label_column,
    )

    log.info("[XAI] Reusing persisted best model from '%s'.", artifacts.best_model_file)
    log.info("[XAI] Winning embedding: %s", metadata["embedding"])
    log.info("[XAI] Winning classifier: %s", metadata["classifier"])

    run_xai(
        embedding_id=metadata["embedding"],
        classifier_name=metadata["classifier"],
        model_path=artifacts.best_model_file,
        texts=dataset_bundle.texts,
        labels=dataset_bundle.labels,
        device=device,
        xai_lime_dir=artifacts.xai_lime_dir,
        xai_shap_dir=artifacts.xai_shap_dir,
        n_examples=args.n_xai,
    )
    log.info("[XAI] Explanation phase completed successfully.")


def run_pipeline(args: argparse.Namespace) -> None:
    from src.data_loader import load_dataset
    from src.embedding_store import EmbeddingStore
    from src.run_artifacts import create_run_artifacts
    from src.trainer import describe_results_table, evaluate_all_classifiers
    from src.utils.checkpoint import ExperimentCheckpoint
    from src.utils.device import log_device_info
    from src.utils.io_utils import atomic_write_json
    from src.utils.logging_setup import get_logger, log_section, log_subsection, setup_pipeline_logger
    from src.utils.eda import plot_subject_length_eda
    from src.xai_runner import run_xai

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts = create_run_artifacts(run_id)
    setup_pipeline_logger(artifacts.log_file)
    log = get_logger("main")

    log_section(log, "PHISHING SUBJECT XAI PIPELINE")

    selected_embeddings, selected_classifiers = _build_scope(args)

    log_subsection(log, "DATASET PREPARATION")
    dataset_bundle = load_dataset(
        Path(args.data),
        subject_column=args.subject_column,
        label_column=args.label_column,
    )

    log.info("[DATA] Dataset path: %s", Path(args.data))
    log.info("[DATA] Final number of instances: %s", dataset_bundle.size)
    log.info("[DATA] Dataset fingerprint: %s", dataset_bundle.dataset_fingerprint)

    log_subsection(log, "EXPLORATORY DATA ANALYSIS")
    plot_subject_length_eda(dataset_bundle.dataframe, artifacts.eda_dir)
    log.info("[EDA] Gráficos guardados en '%s'.", artifacts.eda_dir)

    log_subsection(log, "HARDWARE AND EXECUTION ENVIRONMENT")
    device = log_device_info(log)

    _write_manifest(artifacts, args, dataset_bundle, device, selected_embeddings, selected_classifiers)
    dataset_bundle.dataframe.to_csv(artifacts.prepared_dataset_file, index=False)
    log.info("[DATA] Prepared dataset exported to '%s'.", artifacts.prepared_dataset_file)
    log.info("[RUN] Run manifest saved to '%s'.", artifacts.manifest_file)

    checkpoint_path = _build_checkpoint_path(
        dataset_bundle.dataset_fingerprint,
        selected_embeddings,
        selected_classifiers,
    )
    checkpoint = ExperimentCheckpoint(checkpoint_path)

    log_subsection(log, "EXPERIMENT SCOPE")
    log.info("[RUN] Run identifier: %s", artifacts.run_id)
    log.info("[RUN] Run directory: %s", artifacts.root_dir)
    log.info("[RUN] Persistent checkpoint file: %s", checkpoint_path)
    log.info("[RUN] %s", checkpoint.summary())
    log.info(
        "[RUN] Planned search space: %s embeddings x %s classifiers = %s candidate combinations.",
        len(selected_embeddings),
        len(selected_classifiers),
        len(selected_embeddings) * len(selected_classifiers),
    )

    if args.only_xai:
        _only_xai_mode(args, artifacts, device, log)
        return

    embedding_store = EmbeddingStore(dataset_bundle.dataset_fingerprint)
    best_result = None

    for embedding_id in selected_embeddings:
        skip_reason = checkpoint.is_embedding_skipped(embedding_id)
        if skip_reason:
            log_section(log, f"EMBEDDING SKIPPED: {embedding_id}")
            log.warning("[CHECKPOINT] This embedding was previously marked as skipped: %s", skip_reason)
            continue

        embedding_config = EMBEDDING_MODELS[embedding_id]
        log_section(log, f"EMBEDDING EVALUATION: {embedding_id}")
        log.info("[EMBED] Representation family: %s", embedding_config["paradigm"])
        log.info("[EMBED] Selection rationale: %s", embedding_config["rationale"])

        embedding_matrix = _load_or_compute_embeddings(
            embedding_id,
            embedding_store,
            dataset_bundle.texts,
            device,
            log,
        )
        if embedding_matrix is None:
            checkpoint.mark_embedding_skipped(
                embedding_id,
                "the embedding could not be executed with the available hardware or token configuration",
            )
            log.warning("[CHECKPOINT] Embedding '%s' marked as skipped for future resumptions.", embedding_id)
            continue

        log.info(
            "[EMBED] Embedding matrix ready for '%s' with shape %s.",
            embedding_id,
            embedding_matrix.shape,
        )

        if args.embeddings_only:
            log.info("[EMBED] Embeddings-only mode is active. Classifier evaluation is skipped for '%s'.", embedding_id)
            continue

        local_best = evaluate_all_classifiers(
            embedding_matrix,
            dataset_bundle.labels,
            embedding_id=embedding_id,
            embedding_config=embedding_config,
            classifiers_grid=selected_classifiers,
            checkpoint=checkpoint,
            run_id=artifacts.run_id,
            dataset_fingerprint=dataset_bundle.dataset_fingerprint,
            results_csv=artifacts.results_csv,
            model_registry_file=artifacts.model_registry_file,
            nested_cv_details_dir=artifacts.nested_cv_details_dir,
        )

        if local_best and (
            best_result is None
            or local_best["metrics"].get(PRIMARY_METRIC, 0.0) > best_result["metrics"].get(PRIMARY_METRIC, 0.0)
        ):
            best_result = {
                "embedding": embedding_id,
                **local_best,
            }
            log.info(
                "[SUMMARY] New best result in this run: '%s' + '%s' with %s %.4f.",
                best_result["embedding"],
                best_result["classifier"],
                PRIMARY_METRIC,
                best_result["metrics"].get(PRIMARY_METRIC, 0.0),
            )

    if args.embeddings_only:
        log_section(log, "EMBEDDINGS-ONLY FINISHED")
        log.info("[RUN] Embedding cache preparation completed successfully.")
        log.info("[RUN] Cache root: %s", EmbeddingStore(dataset_bundle.dataset_fingerprint).dataset_dir)
        log.info("[RUN] Run directory: %s", artifacts.root_dir)
        log.info("[RUN] Log file: %s", artifacts.log_file)
        return

    global_best = _select_global_best(checkpoint)

    log_section(log, "EXPERIMENT SUMMARY")
    if global_best is not None:
        log.info("[SUMMARY] Best combination stored in the persistent checkpoint:")
        log.info("[SUMMARY] Embedding: %s", global_best["embedding"])
        log.info("[SUMMARY] Classifier: %s", global_best["classifier"])
        log.info("[SUMMARY] Primary Metric (%s): %.4f", PRIMARY_METRIC, global_best["metrics"].get(PRIMARY_METRIC, 0.0))
        log.info("[SUMMARY] Accuracy: %.4f", global_best["metrics"]["accuracy"])
        log.info("[SUMMARY] ROC-AUC: %.4f", global_best["metrics"]["roc_auc"])

    log.debug(describe_results_table(artifacts.results_csv))
    log.info("[SUMMARY] Cumulative results table updated: %s", artifacts.results_csv)

    if best_result is None and global_best is None:
        log.error("[SUMMARY] No valid model was obtained. The pipeline ends without usable results.")
        return

    winning = best_result if best_result is not None else global_best
    winning_model_path = Path(winning["model_path"])
    winning_metadata = {
        "embedding": winning["embedding"],
        "classifier": winning["classifier"],
        "metrics": winning["metrics"],
        "model_path": str(winning_model_path),
    }
    atomic_write_json(artifacts.best_model_meta_file, winning_metadata)

    if winning_model_path.exists():
        import shutil

        shutil.copy2(winning_model_path, artifacts.best_model_file)

    log.info("[MODEL] Best model metadata saved to '%s'.", artifacts.best_model_meta_file)
    log.info("[MODEL] Best model copied to '%s'.", artifacts.best_model_file)

    if args.skip_xai:
        log.info("[XAI] Explanation phase skipped by user request.")
        return

    log_section(log, "XAI PHASE")
    log.info("[XAI] Running explanations for the winning combination.")
    log.info("[XAI] Winning embedding: %s", winning["embedding"])
    log.info("[XAI] Winning classifier: %s", winning["classifier"])
    log.info("[XAI] Number of representative examples requested: %s", args.n_xai)

    run_xai(
        embedding_id=winning["embedding"],
        classifier_name=winning["classifier"],
        model_path=artifacts.best_model_file,
        texts=dataset_bundle.texts,
        labels=dataset_bundle.labels,
        device=device,
        xai_lime_dir=artifacts.xai_lime_dir,
        xai_shap_dir=artifacts.xai_shap_dir,
        n_examples=args.n_xai,
    )

    log_section(log, "PIPELINE FINISHED")
    log.info("[RUN] All requested stages completed successfully.")
    log.info("[RUN] Run directory: %s", artifacts.root_dir)
    log.info("[RUN] Log file: %s", artifacts.log_file)
    log.info("[RUN] Best model: %s", artifacts.best_model_file)
    log.info("[RUN] LIME outputs: %s", artifacts.xai_lime_dir)
    log.info("[RUN] SHAP outputs: %s", artifacts.xai_shap_dir)


if __name__ == "__main__":
    try:
        run_pipeline(parse_args())
    except Exception as exc:
        print(f"Fatal pipeline failure: {exc}", file=sys.stderr)
        raise
