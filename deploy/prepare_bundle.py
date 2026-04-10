from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
DEPLOY_DIR = PROJECT_ROOT / "deploy"
BUNDLE_DIR = DEPLOY_DIR / "bundle"

DEFAULT_RUN_ID = "20260328_031334"
DEFAULT_EMBEDDING = "distilbert-base-uncased"
DEFAULT_CLASSIFIER = "svm_rbf"


def build_model_filename(embedding: str, classifier: str) -> str:
    return f"{embedding.replace('/', '__')}__{classifier}.joblib"


def build_metadata_filename(embedding: str, classifier: str) -> str:
    return f"{embedding.replace('/', '__')}__{classifier}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a lightweight deploy bundle for Render.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING)
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER)
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_id
    model_path = run_dir / build_model_filename(args.embedding, args.classifier)
    metadata_path = run_dir / build_metadata_filename(args.embedding, args.classifier)
    run_manifest_path = run_dir / "run_manifest.json"

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    if not run_manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found: {run_manifest_path}")

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(model_path, BUNDLE_DIR / "model.joblib")
    shutil.copy2(metadata_path, BUNDLE_DIR / "model_metadata.json")

    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    # Public deploy bundles must not disclose the original dataset filename or host path.
    run_manifest["data_path"] = "data/dataset.csv"
    (BUNDLE_DIR / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    bundle_manifest = {
        "run_id": args.run_id,
        "embedding": args.embedding,
        "classifier": args.classifier,
        "model_path": "model.joblib",
        "model_metadata_path": "model_metadata.json",
        "run_manifest_path": "run_manifest.json",
    }

    (BUNDLE_DIR / "deploy_bundle.json").write_text(
        json.dumps(bundle_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Deploy bundle prepared successfully:")
    print(f"  run_id: {args.run_id}")
    print(f"  embedding: {args.embedding}")
    print(f"  classifier: {args.classifier}")
    print(f"  bundle_dir: {BUNDLE_DIR}")


if __name__ == "__main__":
    main()
