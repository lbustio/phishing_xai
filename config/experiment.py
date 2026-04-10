from __future__ import annotations
from copy import deepcopy

# ====================================================================
# CONFIGURACIÓN GENERAL Y MÉTRICAS
# ====================================================================
DEFAULT_RANDOM_SEED = 42
PRIMARY_METRIC = "f1_macro"
METRICS = [
    "f1_macro",
    "f1_weighted",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "roc_auc",
]

CV_SETTINGS = {
    "outer_folds": 5,
    "inner_folds": 3,
    "shuffle": True,
    "random_state": DEFAULT_RANDOM_SEED,
    "n_jobs": -1,
}

XAI_CONFIG = {
    "n_lime_features": 10,
    "n_lime_samples": 500,
    "lime_kernel_width": 0.1,
    "n_xai_examples": 20,
    "class_names": ["Legitimate", "Phishing"],
}

DATASET_CONFIG = {
    "default_subject_column_candidates": ["subject", "asunto", "title"],
    "default_label_column_candidates": ["label", "class", "target", "is_phishing"],
    "balance_classes": False,
}

def _embedding(
    paradigm: str,
    backend_type: str,
    rationale: str,
    *,
    query_prefix: str = "",
    instruction_mode: str = "prefix",
    batch_size: int = 32,
    trust_remote_code: bool = False,
    requires_hf_token: bool = False,
    skip_on_cpu: bool = False,
) -> dict:
    return {
        "paradigm": paradigm,
        "type": backend_type,
        "query_prefix": query_prefix,
        "instruction_mode": instruction_mode,
        "batch_size": batch_size,
        "trust_remote_code": trust_remote_code,
        "requires_hf_token": requires_hf_token,
        "skip_on_cpu": skip_on_cpu,
        "rationale": rationale,
    }

# ====================================================================
# EMBEDDING LIGERO Y CLASIFICADOR
# ====================================================================
EMBEDDING_MODELS = {
    "sentence-transformers/all-mpnet-base-v2": _embedding(
        "Sentence-level (contrastive)",
        "sentence_transformer",
        "Strong contrastive sentence embedding baseline with excellent general-purpose semantic performance on short texts.",
        batch_size=64,
    ),
    "Salesforce/SFR-Embedding-Mistral": _embedding(
        "Instruction-tuned",
        "large_llm",
        "Enterprise-grade Mistral-based embedding model.",
        batch_size=4,
        trust_remote_code=True
    ),
    "BAAI/bge-m3": _embedding(
        "Sentence-level (contrastive)",
        "sentence_transformer",
        "Multi-granular and multilingual contrastive embedding.",
        batch_size=32
    ),
    "distilbert-base-uncased": _embedding(
        "Transformer encoder",
        "hf_encoder",
        "Baseline encoder ligero derivado de BERT.",
        batch_size=128
    )
}

CLASSIFIERS_GRID = {
    "logistic_regression": {
        "module": "sklearn.linear_model",
        "class_name": "LogisticRegression",
        "use_scaler": True,
        "init_params": {
            "random_state": DEFAULT_RANDOM_SEED,
            "max_iter": 4000,
        },
        "grid_params": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__solver": ["lbfgs", "liblinear"],
            "classifier__class_weight": [None, "balanced"],
        },
        "note": "Mandatory linear baseline.",
    }
}

# ====================================================================
# FUNCIONES
# ====================================================================
def get_embedding_config(embedding_id: str) -> dict:
    config = deepcopy(EMBEDDING_MODELS[embedding_id])
    config["repo"] = embedding_id
    return config