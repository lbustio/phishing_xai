from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SECRETS_DIR = PROJECT_ROOT / "secrets"

CACHE_DIR = RESULTS_DIR / "cache"
RUNS_DIR = RESULTS_DIR / "runs"
TABLES_DIR = RESULTS_DIR / "tables"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

EMBEDDING_CACHE_DIR = CACHE_DIR / "embeddings"
HF_CACHE_HINT_DIR = CACHE_DIR / "huggingface"
FRONTEND_ANALYSES_DIR = RESULTS_DIR / "frontend_analyses"

for directory in [
    DATA_DIR,
    RESULTS_DIR,
    SECRETS_DIR,
    CACHE_DIR,
    RUNS_DIR,
    TABLES_DIR,
    CHECKPOINTS_DIR,
    EMBEDDING_CACHE_DIR,
    HF_CACHE_HINT_DIR,
    FRONTEND_ANALYSES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
