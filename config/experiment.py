from __future__ import annotations

from copy import deepcopy

from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


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
    "lime_kernel_width": 0.25,
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


EMBEDDING_MODELS = {
    "distilbert-base-uncased": _embedding(
        "Transformer encoder",
        "hf_encoder",
        (
            "Lightweight encoder derived from BERT. Included as a low-cost baseline to test "
            "whether a generic Transformer encoder is sufficient for short phishing subjects."
        ),
        batch_size=64,
    ),
    "sentence-transformers/all-mpnet-base-v2": _embedding(
        "Sentence-level (contrastive)",
        "sentence_transformer",
        (
            "Strong contrastive sentence embedding baseline with excellent general-purpose "
            "semantic performance on short texts."
        ),
        batch_size=64,
    ),
    "hkunlp/instructor-large": _embedding(
        "Instruction-tuned",
        "sentence_transformer",
        (
            "Instruction-aware encoder that injects task information without fine-tuning, "
            "which is especially relevant for specialized security classification tasks."
        ),
        query_prefix="Represent the email subject for phishing classification: ",
        instruction_mode="pair",
        batch_size=32,
    ),
    "intfloat/e5-mistral-7b-instruct": _embedding(
        "Instruction-tuned",
        "large_llm",
        (
            "Large instruction-tuned embedding model used to test whether LLM-scale "
            "representations improve robustness on short adversarial subjects."
        ),
        query_prefix="Instruct: Classify this email subject as phishing or legitimate\nQuery: ",
        batch_size=8,
        skip_on_cpu=True,
    ),
    "Salesforce/SFR-Embedding-Mistral": _embedding(
        "Instruction-tuned",
        "large_llm",
        (
            "Enterprise-grade Mistral-based embedding model included as a strong industrial "
            "instruction-tuned representation baseline."
        ),
        query_prefix="Instruct: Classify this email subject as phishing or legitimate\nQuery: ",
        batch_size=8,
        skip_on_cpu=True,
    ),
    "BAAI/bge-m3": _embedding(
        "Sentence-level (contrastive)",
        "sentence_transformer",
        (
            "Multi-granular and multilingual contrastive embedding chosen for its robustness "
            "on heterogeneous short-text inputs."
        ),
        batch_size=64,
    ),
    "intfloat/e5-large-v2": _embedding(
        "Sentence-level (contrastive)",
        "sentence_transformer",
        (
            "Strong contrastive encoder frequently used as a robust text embedding baseline, "
            "especially for short query-like inputs."
        ),
        query_prefix="query: ",
        batch_size=32,
    ),
    "Alibaba-NLP/gte-Qwen2-7B-instruct": _embedding(
        "Instruction-tuned",
        "large_llm",
        (
            "Frontier instruction-tuned 7B embedding model included to evaluate whether "
            "state-of-the-art large-scale representations offer measurable gains."
        ),
        query_prefix="Instruct: Classify this email subject as phishing or legitimate\nQuery: ",
        batch_size=4,
        trust_remote_code=True,
        skip_on_cpu=True,
    ),
    "jinaai/jina-embeddings-v3": _embedding(
        "Sentence-level (contrastive)",
        "sentence_transformer",
        (
            "Modern multitask embedding model included as a practical high-performing sentence "
            "representation candidate."
        ),
        batch_size=32,
        trust_remote_code=True,
    ),
    "meta-llama/Meta-Llama-3.1-8B-Instruct": _embedding(
        "Instruction-tuned",
        "large_llm",
        (
            "Instruction-tuned generative LLM reused as a feature extractor to test whether "
            "a general-purpose model can compete as a phishing representation model."
        ),
        query_prefix="Classify this email subject as phishing or legitimate: ",
        batch_size=4,
        requires_hf_token=True,
        skip_on_cpu=True,
    ),
    "BAAI/bge-large-en-v1.5": _embedding(
        "Sentence-level (contrastive)",
        "sentence_transformer",
        (
            "English-focused contrastive encoder included to measure whether a strong "
            "English-only model outperforms multilingual alternatives."
        ),
        batch_size=32,
    ),
    "thenlper/gte-large": _embedding(
        "Sentence-level (contrastive)",
        "sentence_transformer",
        (
            "Mid-sized contrastive embedding model selected to study the quality-efficiency "
            "trade-off relative to both smaller and larger alternatives."
        ),
        batch_size=32,
    ),
    "Salesforce/SFR-Embedding-2_R": _embedding(
        "Instruction-tuned",
        "large_llm",
        (
            "Second-generation SFR embedding model included to measure within-family "
            "improvements under identical evaluation conditions."
        ),
        query_prefix="Instruct: Classify this email subject as phishing or legitimate\nQuery: ",
        batch_size=8,
        skip_on_cpu=True,
    ),
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
        "note": (
            "Mandatory linear baseline for dense embeddings. Fast, interpretable, and "
            "a strong default reference for text classification."
        ),
    },
    "linear_svc": {
        "module": "sklearn.calibration",
        "class_name": "CalibratedClassifierCV",
        "use_scaler": True,
        "init_params": {
            "estimator": LinearSVC(
                random_state=DEFAULT_RANDOM_SEED,
                dual="auto",
                max_iter=20000,
            ),
            "method": "sigmoid",
            "cv": 3,
        },
        "grid_params": {
            "classifier__estimator__C": [0.001, 0.01, 0.1, 1.0],
        },
        "note": (
            "Linear maximum-margin classifier with probability calibration. Included because "
            "linear SVMs are classic high-quality baselines in text classification."
        ),
    },
    "svm_rbf": {
        "module": "sklearn.svm",
        "class_name": "SVC",
        "use_scaler": True,
        "init_params": {
            "kernel": "rbf",
            "probability": True,
            "random_state": DEFAULT_RANDOM_SEED,
            "cache_size": 2000,
        },
        "grid_params": {
            "classifier__C": [0.1, 1.0, 10.0, 50.0],
            "classifier__gamma": ["scale", "auto"],
            "classifier__class_weight": [None, "balanced"],
        },
        "note": (
            "Non-linear kernel classifier used to test whether phishing and legitimate "
            "subjects are separated by curved decision boundaries in embedding space."
        ),
    },
    "sgd_classifier": {
        "module": "sklearn.linear_model",
        "class_name": "SGDClassifier",
        "use_scaler": True,
        "init_params": {
            "random_state": DEFAULT_RANDOM_SEED,
            "max_iter": 3000,
            "tol": 1e-3,
            "loss": "log_loss",
        },
        "grid_params": {
            "classifier__alpha": [1e-5, 1e-4, 1e-3],
            "classifier__penalty": ["l2", "l1", "elasticnet"],
            "classifier__class_weight": [None, "balanced"],
        },
        "note": (
            "Classic large-scale linear text classifier trained with stochastic optimization. "
            "Included as a fast and methodologically strong baseline."
        ),
    },
    "random_forest": {
        "module": "sklearn.ensemble",
        "class_name": "RandomForestClassifier",
        "use_scaler": False,
        "init_params": {
            "random_state": DEFAULT_RANDOM_SEED,
            "n_jobs": -1,
        },
        "grid_params": {
            "classifier__n_estimators": [200, 500],
            "classifier__max_depth": [None, 10, 20],
            "classifier__max_features": ["sqrt", "log2"],
            "classifier__class_weight": [None, "balanced"],
        },
        "note": (
            "Classical tree ensemble baseline for non-linear separability in dense embedding spaces."
        ),
    },
    "extra_trees": {
        "module": "sklearn.ensemble",
        "class_name": "ExtraTreesClassifier",
        "use_scaler": False,
        "init_params": {
            "random_state": DEFAULT_RANDOM_SEED,
            "n_jobs": -1,
        },
        "grid_params": {
            "classifier__n_estimators": [200, 500],
            "classifier__max_depth": [None, 10, 20],
            "classifier__max_features": ["sqrt", "log2"],
            "classifier__class_weight": [None, "balanced"],
        },
        "note": (
            "Highly randomized tree ensemble included as a classical alternative to random forests "
            "with different bias-variance behavior."
        ),
    },
    "mlp": {
        "module": "sklearn.neural_network",
        "class_name": "MLPClassifier",
        "use_scaler": True,
        "init_params": {
            "random_state": DEFAULT_RANDOM_SEED,
            "max_iter": 700,
            "early_stopping": True,
            "n_iter_no_change": 20,
            "validation_fraction": 0.1,
        },
        "grid_params": {
            "classifier__hidden_layer_sizes": [(256,), (512,), (256, 128)],
            "classifier__alpha": [1e-4, 1e-3, 1e-2],
            "classifier__learning_rate_init": [1e-3, 1e-4],
        },
        "note": (
            "Shallow neural classifier over embeddings, included to test whether a modest learned "
            "non-linear projection improves discrimination."
        ),
    },
    "knn": {
        "module": "sklearn.neighbors",
        "class_name": "KNeighborsClassifier",
        "use_scaler": True,
        "init_params": {
            "n_jobs": -1,
        },
        "grid_params": {
            "classifier__n_neighbors": [3, 5, 9, 15],
            "classifier__metric": ["cosine", "euclidean"],
            "classifier__weights": ["uniform", "distance"],
        },
        "note": (
            "Classic instance-based classifier kept as a geometric baseline to test whether phishing "
            "and legitimate subjects form coherent neighborhoods in embedding space."
        ),
    },
    "decision_tree": {
        "module": "sklearn.tree",
        "class_name": "DecisionTreeClassifier",
        "use_scaler": False,
        "init_params": {
            "random_state": DEFAULT_RANDOM_SEED,
        },
        "grid_params": {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [None, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__class_weight": [None, "balanced"],
        },
        "note": (
            "Directly interpretable tree baseline included to complement post-hoc explanations "
            "with a native rule-based model."
        ),
    },
    "gaussian_nb": {
        "module": "sklearn.naive_bayes",
        "class_name": "GaussianNB",
        "use_scaler": False,
        "init_params": {},
        "grid_params": {
            "classifier__var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
        },
        "note": (
            "Simple probabilistic baseline for dense continuous embeddings. Useful as a fast "
            "reference point even if it is unlikely to be the top performer."
        ),
    },
    "lda": {
        "module": "sklearn.discriminant_analysis",
        "class_name": "LinearDiscriminantAnalysis",
        "use_scaler": True,
        "init_params": {},
        "grid_params": {
            "classifier__solver": ["lsqr"],
            "classifier__shrinkage": [None, "auto"],
        },
        "note": (
            "Linear probabilistic classifier that offers a useful midpoint between simple linear "
            "models and stronger distributional assumptions."
        ),
    },
}


def get_embedding_config(model_id: str) -> dict:
    config = deepcopy(EMBEDDING_MODELS[model_id])
    config["repo"] = model_id
    return config
