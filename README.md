# Phishing Subject XAI Pipeline

Explainable phishing detection pipeline for short texts, focused on email or message subjects only.

This project evaluates multiple Hugging Face embedding models, combines them with classical machine learning classifiers, estimates performance with nested cross-validation, caches embeddings persistently, and explains the best-performing combination with word-ablation attribution and SHAP.

## Overview

The goal of this repository is to build a robust and reproducible experimental pipeline for phishing detection from short text subjects, while keeping the system suitable for academic reporting and later paper writing.

The pipeline is designed to:

- compare several embedding paradigms, from lightweight encoders to instruction-tuned large models
- compare multiple downstream classifiers under a controlled tuning protocol
- support CPU and GPU execution with the same codebase
- resume work after interruptions without recomputing everything
- cache downloaded models and computed embeddings
- log every relevant action to both console and file using descriptive, paper-friendly messages
- generate local and global XAI outputs for the best model
- expose a demo app for real-time forensic inspection

## Experimental Design

The pipeline follows this workflow:

1. Load and clean the dataset.
2. Build a reproducible dataset fingerprint after preprocessing.
3. Compute or reuse cached embeddings for each selected embedding model.
4. Evaluate each embedding with multiple classifiers.
5. Tune classifier hyperparameters with `GridSearchCV`.
6. Estimate generalization with nested cross-validation.
7. Refit the best hyperparameter setting on the full dataset.
8. Save models, metadata, fold-level reports, cumulative tables, and logs.
9. Run word-ablation attribution and SHAP on representative examples from the best-performing combination.

## Key Methodological Decisions

### Nested Cross-Validation

Classifier selection is not evaluated with a single optimistic cross-validation loop. Instead, the repository uses:

- an inner CV loop for hyperparameter selection
- an outer CV loop for robust performance estimation

This makes the reported results much more defensible in a research setting.

### Persistent Embedding Cache

Embeddings are cached using the fingerprint of the cleaned dataset plus the embedding identifier. If the same prepared dataset is used again, embeddings are reused instead of recomputed.

### Crash Recovery

Completed embedding-classifier combinations are recorded in a persistent experiment checkpoint. If the run is interrupted, the next execution can continue from the last completed state.

### Hardware Agnosticism

The code automatically selects the best available device:

- `cuda` if available
- `mps` on Apple Silicon
- `cpu` otherwise

Large 7B embedding models can be skipped automatically on CPU to avoid impractical runs.

## Embedding Models

The repository currently includes the following embedding candidates:

- `distilbert-base-uncased`
- `sentence-transformers/all-mpnet-base-v2`
- `hkunlp/instructor-large`
- `intfloat/e5-mistral-7b-instruct`
- `Salesforce/SFR-Embedding-Mistral`
- `BAAI/bge-m3`
- `intfloat/e5-large-v2`
- `Alibaba-NLP/gte-Qwen2-7B-instruct`
- `jinaai/jina-embeddings-v3`
- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `BAAI/bge-large-en-v1.5`
- `thenlper/gte-large`
- `Salesforce/SFR-Embedding-2_R`

Each embedding entry stores:

- exact Hugging Face repository identifier
- representation paradigm
- selection rationale
- runtime hints such as batch size, remote code, token requirements, and CPU skip policy

The rationale field is intended to support the corresponding model-selection table in an academic paper.

## Classifiers

The current downstream classifiers are:

- Logistic Regression
- Linear SVM
- RBF SVM
- Random Forest
- Gradient Boosting
- MLP
- KNN

Each classifier includes:

- constructor settings
- whether feature scaling is applied
- a full hyperparameter grid
- a short methodological note

## Repository Structure

```text
.
|-- config/
|   |-- experiment.py
|   `-- paths.py
|-- data/
|-- demo_app/
|   |-- api.py
|   `-- static/
|-- results/
|   |-- cache/
|   |-- checkpoints/
|   |-- runs/
|   `-- tables/
|-- secrets/
|   |-- groq.txt
|   `-- huggingface.txt
|-- src/
|   |-- embeddings/
|   |-- utils/
|   |-- xai/
|   |-- data_loader.py
|   |-- embedding_store.py
|   |-- run_artifacts.py
|   |-- trainer.py
|   `-- xai_runner.py
|-- main.py
`-- main_emb.py   # Compatibility wrapper for embeddings-only mode
```

## Installation

Create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate xai
```

If you want to run gated Hugging Face embedding models such as Llama, set a valid token:

```bash
export HF_TOKEN=your_token_here
```

On Windows PowerShell:

```powershell
$env:HF_TOKEN="your_token_here"
```

## Secrets

Runtime secrets for the demo app must live in the dedicated `secrets/` directory at the project root.

Expected files:

- `secrets/groq.txt`
- `secrets/huggingface.txt`

Each file should contain only the raw token, with no extra quotes or JSON wrapper.

By default the application reads secrets from `./secrets`. You can override that location with the `XAI_SECRETS_DIR` environment variable. The provided `environment.yml` already defines that variable.

Notes:

- `secrets/huggingface.txt` is used by the natural-language XAI demo layer.
- `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` are still used by the embedding pipeline when downloading gated Hugging Face models.

## Dataset Expectations

The pipeline expects a CSV file containing:

- one subject column
- one label column

By default, the loader tries to infer common column names such as:

- subject-like: `subject`, `asunto`, `title`
- label-like: `label`, `class`, `target`, `is_phishing`

If needed, you can specify them explicitly with CLI arguments.

Labels are mapped to binary classes:

- `0`: legitimate
- `1`: phishing

Several common textual label variants are already supported.

## Usage

### Quick Commands

Full experiment:

```bash
python main.py
```

Show the extended CLI help:

```bash
python main.py --help
```

Alternative help aliases:

```bash
python main.py --h
python main.py -help
```

Run only the embedding cache stage:

```bash
python main.py --embeddings-only
```

Legacy wrapper equivalent:

```bash
python main_emb.py
```

Reuse a previous run and generate XAI only:

```bash
python main.py --run-id 20260327_123456 --only-xai
```

### CLI Parameters

`--data`
: CSV path to load. Use it when you want to switch datasets without touching the code.

`--subject-column`
: Explicit subject/text column name. Useful when your CSV header is not one of the auto-detected defaults.

`--label-column`
: Explicit label column name. Use it when the dataset stores labels under a custom name.

`--embeddings`
: One or more embedding IDs to evaluate. If omitted, the pipeline evaluates all embeddings defined in `config.experiment`.

`--classifiers`
: One or more classifier IDs to evaluate. Good for faster ablations or targeted comparisons.

`--skip-xai`
: Runs the experiment and selects the best model, but stops before generating explanation artefacts.

`--only-xai`
: Skips training and evaluation and reuses a previous run to generate XAI only. In practice this should be used together with `--run-id`.

`--embeddings-only`
: Computes or refreshes the embedding cache only. No classifier training, no model selection, and no XAI generation.

`--n-xai`
: Number of representative texts to explain during the XAI phase. This matters only when XAI is enabled.

`--run-id`
: Reuses an existing run directory. Most useful with `--only-xai`, but it can also help you keep outputs grouped under a known identifier.

### Example Recipes

Run the full pipeline on another dataset:

```bash
python main.py --data data/your_dataset.csv
```

Force the subject and label columns:

```bash
python main.py --data data/your_dataset.csv --subject-column subject --label-column label
```

Evaluate only two embeddings:

```bash
python main.py --embeddings sentence-transformers/all-mpnet-base-v2 BAAI/bge-m3
```

Evaluate only one classifier:

```bash
python main.py --classifiers logistic_regression
```

Prepare embeddings only for a subset of models:

```bash
python main.py --embeddings-only --embeddings sentence-transformers/all-mpnet-base-v2 BAAI/bge-m3
```

Run training without explanations:

```bash
python main.py --skip-xai
```

## XAI PowerToy (Demo App)

The repository includes a web-based demonstration tool called the **PowerToy**, designed for non-technical users to explore the model's decision-making process in real-time.

### Features

- **Real-Time Forensic Console**: Displays the analysis steps (Embeddings, LIME, LLM) as they happen via a streaming backend.
- **Master XAI Reasoning**: An AI-generated synthesis that integrates model confidence and word-level impacts into a human-readable narrative.
- **Risk Impact Visualization**: Highlights critical keywords within the subject line and displays their percentage contribution to the phishing risk.
- **Malicious Email Simulation**: Generates a plausible email body to demonstrate how the detected subject would look in a real attack scenario.

### Running the PowerToy

1. Ensure the `xai` conda environment is active.
2. Launch the FastAPI server:
   ```bash
   python demo_app/api.py
   ```
3. Open your browser at `http://localhost:8000`.

## Outputs

Each run creates a dedicated directory under `results/runs/<run_id>/`.

Typical outputs include:

- `run.log`: descriptive execution log
- `run_manifest.json`: run-level metadata
- `prepared_dataset.csv`: cleaned dataset used by the experiment
- `best_model.joblib`: copied best model for the run
- `best_model_meta.json`: metadata for the best model
- `trained_models_registry.json`: registry of trained embedding-classifier models
- `nested_cv_details/*.json`: per-combination fold-level evaluation records
- `xai/lime/*`: LIME plots and explanation data
- `xai/shap/*`: SHAP plots and explanation data

Global reusable outputs include:

- `results/cache/embeddings/`: persistent embedding cache
- `results/checkpoints/`: persistent experiment checkpoints
- `results/tables/all_results.csv`: cumulative result table across runs

## Logging

The logging system is intentionally verbose and descriptive. It is designed to answer questions such as:

- what was loaded and from where
- which device was used
- which embedding was skipped and why
- which hyperparameters were selected in each fold
- which model won and with what metrics
- which XAI artifacts were generated

This makes the logs useful not only for debugging, but also for reconstructing the experimental narrative later.

## XAI

The best embedding-classifier pair is explained with:

- LIME for local word-level contributions
- SHAP for local and aggregated token-level contributions

Representative examples are selected from a mix of:

- true positives
- true negatives
- false positives
- false negatives

This avoids explaining only the easy cases.

## Reproducibility

The repository includes several mechanisms to improve reproducibility:

- centralized experiment configuration
- stable dataset fingerprint after preprocessing
- persistent embedding cache
- persistent checkpointing
- run-specific manifests and artifacts
- cumulative results table
- explicit hyperparameter grids
- fixed random seeds in the evaluation protocol

## Current Limitations

- The pipeline currently works on subject text only, not full email bodies.
- Large embedding models may still be expensive even on GPU.
- SHAP can be slow because it repeatedly queries the full text-to-embedding-to-classifier pipeline.
- The repository assumes a classical ML downstream stage, not end-to-end fine-tuning.

## Recommended Next Steps

- add automated tests for configuration, dataset loading, and checkpoint behavior
- add export utilities for LaTeX tables
- add a paper-ready experiment summary generator
- add support for train/validation/test protocols in addition to nested CV
- add optional dimensionality reduction and calibration analysis

## Citation

If you use this repository in academic work, cite the paper that accompanies the experiments once available.

## License

Add your preferred open-source license here.
