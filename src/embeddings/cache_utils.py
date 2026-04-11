from __future__ import annotations

from pathlib import Path


def get_hf_model_cache_status(repo_name: str, cache_root: str | Path) -> str:
    """
    Determine whether a Hugging Face model appears to be cached locally.

    Returns one of:
    - "missing"
    - "incomplete"
    - "cached"
    """
    cache_root = Path(cache_root)
    safe_repo = f"models--{repo_name.replace('/', '--')}"
    model_path = cache_root / safe_repo

    if not model_path.exists():
        return "missing"

    blobs_path = model_path / "blobs"
    snapshots_path = model_path / "snapshots"

    has_blobs = blobs_path.exists() and any(blobs_path.iterdir())
    has_snapshots = snapshots_path.exists() and any(snapshots_path.iterdir())
    has_snapshot_files = snapshots_path.exists() and any(path.is_file() for path in snapshots_path.rglob("*"))

    if has_snapshots and (has_blobs or has_snapshot_files):
        return "cached"

    return "incomplete"
