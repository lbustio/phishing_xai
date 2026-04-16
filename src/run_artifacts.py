from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.paths import RUNS_DIR, TABLES_DIR


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    root_dir: Path
    log_file: Path
    manifest_file: Path
    checkpoint_file: Path
    prepared_dataset_file: Path
    best_model_file: Path
    best_model_meta_file: Path
    model_registry_file: Path
    results_csv: Path
    nested_cv_details_dir: Path
    xai_dir: Path
    xai_lime_dir: Path
    xai_shap_dir: Path
    eda_dir: Path


def create_run_artifacts(run_id: str) -> RunArtifacts:
    root_dir = RUNS_DIR / run_id
    nested_cv_details_dir = root_dir / "nested_cv_details"
    xai_dir = root_dir / "xai"
    xai_lime_dir = xai_dir / "lime"
    xai_shap_dir = xai_dir / "shap"
    eda_dir = root_dir / "eda"

    for directory in [
        root_dir,
        nested_cv_details_dir,
        xai_dir,
        xai_lime_dir,
        xai_shap_dir,
        eda_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    return RunArtifacts(
        run_id=run_id,
        root_dir=root_dir,
        log_file=root_dir / "run.log",
        manifest_file=root_dir / "run_manifest.json",
        checkpoint_file=root_dir / "experiment_checkpoint.json",
        prepared_dataset_file=root_dir / "prepared_dataset.csv",
        best_model_file=root_dir / "best_model.joblib",
        best_model_meta_file=root_dir / "best_model_meta.json",
        model_registry_file=root_dir / "trained_models_registry.json",
        results_csv=TABLES_DIR / "all_results.csv",
        nested_cv_details_dir=nested_cv_details_dir,
        xai_dir=xai_dir,
        xai_lime_dir=xai_lime_dir,
        xai_shap_dir=xai_shap_dir,
        eda_dir=eda_dir,
    )
