# Technical README

This document is the operator manual for `phishing_xai`. It focuses on how to install, configure, run, inspect, and maintain the project as it is currently implemented.

For the repository overview, read [`../README.md`](../README.md). For the methodological explanation, read [`README_THEORETICAL.md`](README_THEORETICAL.md).

## 1. Scope of This Document

This file explains:

- environment setup
- local directory expectations
- secrets handling
- dataset handling
- command-line usage
- outputs and artifact layout
- demo application usage
- common operational caveats

It does not attempt to justify the modeling choices in depth. That is covered in [`README_THEORETICAL.md`](README_THEORETICAL.md).

## 2. Runtime Requirements

The project is written in Python and is currently organized around a Conda environment described in [`../environment.yml`](../environment.yml).

The declared dependencies include:

- Python `3.10`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `joblib`
- `psutil`
- `pytorch`
- `numba` (required by SHAP for JIT compilation)
- `shap`
- `fastapi`
- `uvicorn[standard]`
- `pydantic`
- `requests`
- `transformers`
- `sentence-transformers`
- `huggingface_hub`
- `accelerate`
- `lime`
- `langdetect`
- `groq`
- `tenacity`
- `jinja2`

The current environment file also defines:

- `XAI_SECRETS_DIR=./secrets`

## 3. Installation

Create the environment:

```bash
conda env create -f environment.yml
conda activate xai
```

If you need gated Hugging Face models, set one of the expected tokens:

```bash
export HF_TOKEN=your_token_here
```

On Windows PowerShell:

```powershell
$env:HF_TOKEN="your_token_here"
```

The code may also read `HUGGING_FACE_HUB_TOKEN` through the Hugging Face stack, depending on the model backend and library behavior.

If an embedding model is already present under `results/cache/huggingface/`, the current embedding factory attempts to load it in local-only mode to avoid unnecessary Hub checks and repeat downloads.

## 4. Local Directory Contract

The project expects these root-level directories to exist:

- `config/`
- `data/`
- `console/`
- `results/`
- `secrets/`
- `src/`

The paths are centralized in [`../config/paths.py`](../config/paths.py). That module creates required directories automatically at import time, including:

- `results/cache/`
- `results/cache/huggingface/`
- `results/frontend_analyses/`
- `results/runs/`
- `results/tables/`
- `results/checkpoints/`
- `results/temp_demo_xai/`

### Important Git Behavior

The repository intentionally does not version these contents:

- dataset files inside `data/`
- the full `results/` tree
- `secrets/`

That means a fresh clone gives you the expected directory structure, but not the local data, local results, or local tokens.

## 5. Dataset Handling

The default dataset path used in examples is:

```text
data/dataset.csv
```

This file is not shipped with the repository. Collaborators must place a local dataset in `data/` and may either name it `dataset.csv` or pass its real path explicitly with `--data`.

### 5.1 Required Dataset Shape

The loader expects a CSV with:

- one subject-like column
- one label-like column

Subject column auto-detection candidates:

- `subject`
- `asunto`
- `title`

Label column auto-detection candidates:

- `label`
- `class`
- `target`
- `is_phishing`

If auto-detection is not suitable, you can force the column names through the CLI.

### 5.2 Label Mapping

The current code maps the following values to phishing (`1`):

- `phishing`
- `phish`
- `spam`
- `malicious`
- `1`
- `true`
- `yes`

It maps the following values to legitimate (`0`):

- `legitimate`
- `legit`
- `ham`
- `benign`
- `0`
- `false`
- `no`

Rows with unrecognized labels are dropped.

### 5.3 Subject Cleaning

The loader currently:

- strips leading and trailing whitespace
- removes control characters
- collapses repeated whitespace
- removes rows that become empty after cleaning

The dataset fingerprint is computed after this preprocessing stage.

## 6. Secrets Handling

The repository expects a dedicated `secrets/` directory for local tokens.

Expected files:

- `secrets/groq.txt`
- `secrets/huggingface.txt`

Each file should contain only the raw token string.

The current project uses secrets in two distinct ways:

- Hugging Face tokens for gated model access and some explanation flows
- Groq token for the natural-language explanation layer in the demo flow

### 6.0 Groq API Key Details

The demo explanation layer currently expects a Groq API key when natural-language reasoning and synthetic email-body generation are enabled.

Minimal setup:

```text
secrets/groq.txt
```

Contents:

```text
your_groq_api_key_here
```

The runtime reads that file and uses the token as the `api_key` for the Groq client used by the explanation layer.

Operational implications:

- if `secrets/groq.txt` exists and contains a valid key, the demo can request the natural-language reasoning and synthetic body generation layer through Groq
- if the file is missing, empty, or invalid, the explanation layer may fall back to reduced behavior or fail, depending on the code path being exercised
- this key should never be committed to Git

If you prefer an environment-variable workflow in your shell session, keep the file-based contract as the repository default and generate or synchronize `secrets/groq.txt` locally before running the demo. The current implementation is documented around the local secrets directory, not around a required `GROQ_API_KEY` environment variable.

Language handling in the current demo is implemented explicitly:

- the explanatory reasoning is always requested in Spanish
- the synthetic email body is requested in the detected language of the analyzed subject
- arbitrary subject languages rely on `langdetect`; if it is missing, the code falls back to a simple Spanish-vs-English heuristic

The `secrets/` directory is ignored by Git and should remain local.

## 6.1 Frontend Analysis Persistence

The PowerToy now persists each frontend analysis under:

```text
results/frontend_analyses/<analysis_id>/
```

Each directory is created when the frontend analysis result is finalized. The backend writes structured textual artifacts immediately. The browser then uploads visual/export artifacts when they are available.

Typical contents:

- `analysis_result.json`
- `metadata.json`
- `processing_log.txt`
- `reasoning.txt`
- `keywords.json`
- `synthetic_email_body.txt`
- `semantic_map.json`
- `semantic_scatter.png`
- `forensic_report.pdf`

This artifact space is separate from `results/runs/`. Training runs document model development. Frontend analyses document operator-facing case studies and reportable examples produced through the PowerToy flow.

## 7. Main Pipeline Entry Point

The main entry point is [`../main.py`](../main.py).

The current CLI supports:

- `--data`
- `--subject-column`
- `--label-column`
- `--embeddings`
- `--classifiers`
- `--skip-xai`
- `--only-xai`
- `--embeddings-only`
- `--n-xai`
- `--run-id`
- `--help`

### 7.1 Basic Commands

Run the full experiment:

```bash
python main.py
```

Show CLI help:

```bash
python main.py --help
```

Run embeddings only:

```bash
python main.py --embeddings-only
```

Run a subset of embeddings:

```bash
python main.py --embeddings sentence-transformers/all-mpnet-base-v2 BAAI/bge-m3
```

Run a subset of classifiers:

```bash
python main.py --classifiers logistic_regression linear_svc
```

Skip the XAI phase:

```bash
python main.py --skip-xai
```

Generate XAI only for an existing run:

```bash
python main.py --run-id 20260329_103615 --only-xai
```

### 7.2 CLI Semantics

`--data`

- Overrides the dataset path without editing source code.

`--subject-column`

- Explicitly selects the text column.

`--label-column`

- Explicitly selects the label column.

`--embeddings`

- Restricts the sweep to a subset of embedding IDs defined in [`../config/experiment.py`](../config/experiment.py).

`--classifiers`

- Restricts the sweep to a subset of classifier IDs defined in [`../config/experiment.py`](../config/experiment.py).

`--skip-xai`

- Runs data preparation, embedding generation, model evaluation, and model selection, but does not generate explanation artifacts.

`--only-xai`

- Skips training and attempts to reuse a previous run.
- Requires a valid `--run-id` whose run directory already contains the necessary persisted artifacts.

`--embeddings-only`

- Computes or refreshes the embedding cache and stops before classifier training.

`--n-xai`

- Controls how many examples are passed into the XAI stage.

`--run-id`

- Forces artifacts into a known run directory or reuses one when combined with `--only-xai`.

## 8. Current Active Search Space

The current experiment configuration in [`../config/experiment.py`](../config/experiment.py) defines:

- `13` embeddings
- `11` classifiers
- `5` outer folds
- `3` inner folds

This can be computationally expensive, especially when large embeddings are enabled.

### 8.1 Embedding List

- `distilbert-base-uncased`: low-cost generic Transformer baseline.
- `sentence-transformers/all-mpnet-base-v2`: strong general-purpose contrastive sentence baseline.
- `hkunlp/instructor-large`: task-aware instruction encoder without fine-tuning.
- `intfloat/e5-mistral-7b-instruct`: large instruction-tuned embedding model for LLM-scale robustness testing.
- `Salesforce/SFR-Embedding-Mistral`: strong industrial instruction-tuned baseline.
- `BAAI/bge-m3`: multilingual multi-granular contrastive model for heterogeneous short text.
- `intfloat/e5-large-v2`: robust contrastive baseline for query-like inputs.
- `Alibaba-NLP/gte-Qwen2-7B-instruct`: frontier 7B instruction-tuned representation model.
- `jinaai/jina-embeddings-v3`: practical modern multitask sentence embedding candidate.
- `meta-llama/Meta-Llama-3.1-8B-Instruct`: general-purpose LLM reused as a feature extractor.
- `BAAI/bge-large-en-v1.5`: English-focused contrastive comparison point.
- `thenlper/gte-large`: mid-sized contrastive model for quality-efficiency comparison.
- `Salesforce/SFR-Embedding-2_R`: second-generation SFR model for within-family comparison.

### 8.2 Classifier List

- `logistic_regression`: mandatory linear dense-embedding baseline.
- `linear_svc`: classic linear maximum-margin text-classification reference.
- `svm_rbf`: non-linear kernel model for curved decision boundaries.
- `sgd_classifier`: scalable stochastic linear baseline.
- `random_forest`: standard tree-ensemble non-linear baseline.
- `extra_trees`: randomized tree-ensemble comparison point.
- `mlp`: shallow neural baseline over embeddings.
- `knn`: geometric neighborhood baseline in embedding space.
- `decision_tree`: directly interpretable rule-based baseline.
- `gaussian_nb`: simple probabilistic baseline for continuous embeddings.
- `lda`: linear probabilistic midpoint between simple linear and stronger distributional models.

## 9. Outputs and Artifacts

Each run creates a directory under:

```text
results/runs/<run_id>/
```

Typical run outputs include:

- `run.log`
- `run_manifest.json`
- `prepared_dataset.csv`
- `best_model.joblib`
- `best_model_meta.json`
- `trained_models_registry.json`
- `nested_cv_details/*.json`
- `xai/lime/*`
- `xai/shap/*`

Global outputs include:

- `results/cache/embeddings/`
- `results/cache/huggingface/`
- `results/checkpoints/`
- `results/tables/all_results.csv`

### 9.1 Important Note About Version Control

These artifacts are intentionally not committed to Git because they may contain:

- large binary models
- model cache files
- repeated experiment outputs
- local evaluation history
- data-derived artifacts

## 10. Analyzer Application

The repository includes a FastAPI-based console under [`../console/api.py`](../console/api.py).

The console is not an independent training system. It is a consumer of previously generated artifacts.

### 10.1 What The Analyzer Needs

The console expects:

- a populated `results/tables/all_results.csv`
- one or more completed run directories under `results/runs/`
- the required local secrets for the natural-language explanation layer

### 10.2 Running The Analyzer

Start the API server:

```bash
python console/api.py
```

Then open:

```text
http://localhost:8000
```

### 10.3 What The Demo Does

The demo:

- selects a previously trained model from historical results
- loads the corresponding embedding model and classifier
- computes phishing probability for a user-supplied subject
- computes leave-one-out keyword attribution
- generates a natural-language explanation via a three-level LLM engine (Groq → HuggingFace Router → local Qwen)
- generates a synthetic email body in the detected language of the subject
- reconstructs a semantic map from the real dataset and real embeddings of the selected run
- projects the semantic space to 2D with PCA for interactive visualization
- exposes per-point hover inspection with cosine similarity and cosine distance to the analyzed subject
- reports nearest-neighbor and centroid distances by class
- injects a semantic reading note into the reasoning panel
- reconstructs the same semantic view inside the history-detail modal
- can export a PDF that includes the semantic reading and the scatter snapshot when available
- stores local demo cache and local history files
- stores per-analysis frontend artifacts under `results/frontend_analyses/`
- provides a history sidebar with:
  - a per-card delete button (trash icon, revealed on hover) calling `DELETE /api/history/{index}`
  - a **Borrar todo** button fixed at the bottom of the sidebar calling `DELETE /api/history`
  - full forensic detail modal on card click

### 10.4 Demo Semantics and Interaction

The current semantic panel in the PowerToy is based on:

- the dataset referenced by the selected run manifest
- the embedding model selected by the demo
- cached embeddings reused from `results/cache/embeddings/` when available
- a `PCA` projection to two dimensions for display only

Important interpretation rule:

- the 2D coordinates are only for visualization
- the reported similarity and distance values are computed in the original embedding space with cosine similarity / cosine distance

The browser-side chart is rendered with Plotly loaded from a CDN at runtime. It is not a Python dependency in `environment.yml`.

### 10.4.1 LLM Explanation Engine

The natural-language reasoning layer uses a three-level fallback chain:

1. **Groq** (`llama-3.3-70b-versatile`): primary engine. Active when `secrets/groq.txt` contains a valid key.
2. **HuggingFace Router** (`meta-llama/Llama-3.1-8B-Instruct`): secondary engine. Active when `secrets/huggingface.txt` is available and Groq is not.
3. **Local Qwen** (`Qwen/Qwen2.5-1.5B-Instruct`): offline fallback loaded via `transformers.pipeline`. No API key required.

All three engines apply `tenacity`-based exponential-backoff retry logic on transient failures.

## 10.5 Lightweight Public Deploy

The repository also includes a simplified public-demo package under [`../deploy/`](../deploy/).

That package is intentionally narrower than the full local PowerToy. In particular, it omits:

- the semantic scatter based on the private local dataset
- history reconstruction
- PDF export
- other dataset-dependent artifacts that should remain local

The deploy package includes:

- a simplified FastAPI app
- a sanitized model bundle
- a lightweight frontend
- a Render service template

The operational guide for that public-demo package is in [`../deploy/README.md`](../deploy/README.md).

## 11. Logging

The training pipeline logs extensively to console and file.

The logs are intended to answer:

- which dataset was loaded
- which columns were used
- how many rows were dropped and why
- which device was selected
- which embeddings were reused from cache
- which embeddings were skipped
- which hyperparameters won inside each fold
- which model won globally
- which XAI artifacts were generated

This logging behavior is useful both for debugging and for later reconstruction of the experiment narrative.

## 12. Known Operational Caveats

- Large models may be impractical on CPU-only hardware.
- The full configuration can take a long time.
- Some explanation stages are materially slower than plain inference.
- The demo depends on previous results and is not meaningful before at least one completed run.
- The current repository has no automated tests, so configuration changes should be validated carefully.

## 13. Recommended Operating Practice

For collaborators using this codebase, the safest routine is:

1. Verify the dataset path and column names.
2. Start with a restricted run using a subset of embeddings and classifiers.
3. Confirm that artifacts are being written as expected.
4. Expand toward the full search space.
5. Preserve run directories and result tables locally for auditability.

## 14. Files Worth Reading Alongside This Document

- [`../main.py`](../main.py)
- [`../config/experiment.py`](../config/experiment.py)
- [`../config/paths.py`](../config/paths.py)
- [`../src/data_loader.py`](../src/data_loader.py)
- [`../src/trainer.py`](../src/trainer.py)
- [`../src/xai_runner.py`](../src/xai_runner.py)
