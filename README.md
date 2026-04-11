# phishing_xai

<p align="center">
  <img src="demo_app/static/logo-corporate.svg" alt="phishing_xai logo" width="220">
</p>

<p align="center">
  <img src="demo_app/static/logo-shield.svg" alt="phishing shield icon" width="72">
</p>

<p align="center">
  Experimental framework for explainable phishing detection on short email subjects.
</p>

Explainable phishing detection pipeline for short email subjects.

This repository implements an experimental framework that compares multiple text embedding models and multiple classical classifiers for binary phishing detection using subject lines only. It is designed for reproducible experimentation, careful model comparison, and post hoc explanation of the selected model.

The project does not claim to solve phishing detection in the general case. Its scope is narrower and explicit:

- input text is the subject line only
- the downstream task is binary classification: `legitimate` vs `phishing`
- the current modeling strategy is `embedding -> classical classifier -> XAI`
- the codebase prioritizes reproducibility and inspectability over minimal runtime

## Documentation Map

This repository uses three complementary documentation entry points:

- [`README.md`](README.md): repository overview, current configuration, and first-run orientation
- [`docs/README_TECHNICAL.md`](docs/README_TECHNICAL.md): operational guide for installation, execution, artifacts, and maintenance
- [`docs/README_THEORETICAL.md`](docs/README_THEORETICAL.md): methodological and theoretical explanation of the experimental design

The three files are intentionally different in role. The root README is the GitHub-facing overview. The technical README is the operator manual. The theoretical README explains why the pipeline is structured the way it is and what assumptions it makes.

## Why This Project Exists

Short email subjects are operationally important because they are often the first visible cue a user receives, and they may already contain urgency, authority cues, credential prompts, account warnings, or payment requests. At the same time, subject lines are short, noisy, and context-poor, which makes both modeling and explanation difficult.

This repository exists to support a disciplined evaluation of that problem under a controlled setup:

- compare embedding families rather than relying on a single encoder
- compare classical classifiers rather than assuming one downstream model is sufficient
- separate model selection from model assessment through nested cross-validation
- persist intermediate results so large sweeps can be resumed
- produce explanation artifacts for the selected model instead of stopping at accuracy alone

## Current Experimental Configuration

The active configuration is defined in [`config/experiment.py`](config/experiment.py). At the time of writing, the repository is configured with:

- `13` embedding models
- `11` classifiers
- nested cross-validation with `5` outer folds and `3` inner folds
- `f1_macro` as the primary model-selection metric

### Embeddings

The current embedding set includes:

- `distilbert-base-uncased`: low-cost BERT-derived baseline used to test whether a generic lightweight encoder is already sufficient for short subjects.
- `sentence-transformers/all-mpnet-base-v2`: strong general-purpose contrastive baseline for short-text semantics.
- `hkunlp/instructor-large`: instruction-aware encoder included to test whether task-aware prompting helps without fine-tuning.
- `intfloat/e5-mistral-7b-instruct`: large instruction-tuned embedding model used to test LLM-scale robustness on adversarial short text.
- `Salesforce/SFR-Embedding-Mistral`: industrial instruction-tuned baseline included as a strong enterprise-grade representation model.
- `BAAI/bge-m3`: multilingual and multi-granular contrastive model chosen for robustness on heterogeneous short inputs.
- `intfloat/e5-large-v2`: well-established contrastive baseline for query-like short texts.
- `Alibaba-NLP/gte-Qwen2-7B-instruct`: frontier 7B instruction-tuned embedding model included to test whether newer large-scale representations add measurable gains.
- `jinaai/jina-embeddings-v3`: modern multitask sentence embedding model included as a practical high-performing candidate.
- `meta-llama/Meta-Llama-3.1-8B-Instruct`: general-purpose instruction-tuned LLM reused as a feature extractor to test transferability.
- `BAAI/bge-large-en-v1.5`: English-focused contrastive encoder included to compare English-only strength against multilingual alternatives.
- `thenlper/gte-large`: mid-sized contrastive model included to study the quality-efficiency trade-off.
- `Salesforce/SFR-Embedding-2_R`: second-generation SFR model included to measure within-family improvement under the same evaluation protocol.

These embeddings span three broad families:

- transformer encoder baselines
- sentence-transformer / contrastive encoders
- instruction-tuned large language model embeddings

Some large models are marked `skip_on_cpu=True` in the configuration. That means the pipeline may omit them automatically on CPU-only hardware to avoid impractical runs.

### Classifiers

The active classifier set is:

- `logistic_regression`: mandatory linear baseline for dense embeddings; fast, interpretable, and a strong default reference.
- `linear_svc`: linear maximum-margin baseline included because linear SVMs remain a classic strong text-classification reference.
- `svm_rbf`: non-linear kernel model used to test whether embedding-space separation is curved rather than linear.
- `sgd_classifier`: scalable stochastic linear baseline included as a fast and methodologically solid reference.
- `random_forest`: tree ensemble baseline for non-linear separability in dense embedding spaces.
- `extra_trees`: randomized tree ensemble included to compare against random forests with a different bias-variance profile.
- `mlp`: shallow neural classifier included to test whether a modest learned non-linear projection improves discrimination.
- `knn`: neighborhood-based baseline used to test whether phishing and legitimate subjects form coherent local regions in embedding space.
- `decision_tree`: directly interpretable tree baseline included to complement post hoc explanations with a native rule-based model.
- `gaussian_nb`: simple probabilistic baseline for dense continuous embeddings, useful as a fast reference even if not expected to win.
- `lda`: linear probabilistic classifier included as a midpoint between simple linear models and stronger distributional assumptions.

This set was chosen to cover a range of modeling assumptions:

- linear discriminative models
- margin-based models
- probabilistic baselines
- tree-based learners
- neighborhood methods
- a shallow neural baseline

Each classifier has an explicit hyperparameter grid in [`config/experiment.py`](config/experiment.py). Those grids are part of the documented experiment design, not an implementation detail.

## Repository Structure

```text
.
|-- config/
|   |-- experiment.py
|   |-- paths.py
|   `-- __init__.py
|-- data/
|   `-- .gitkeep
|-- demo_app/
|   |-- api.py
|   `-- static/
|-- deploy/
|   |-- app.py
|   |-- bundle/
|   |-- static/
|   `-- render.yaml
|-- docs/
|   |-- README_TECHNICAL.md
|   `-- README_THEORETICAL.md
|-- results/
|   |-- frontend_analyses/
|   `-- runs/
|-- secrets/
|-- src/
|   |-- embeddings/
|   |-- utils/
|   |-- xai/
|   |-- data_loader.py
|   |-- embedding_store.py
|   |-- run_artifacts.py
|   |-- trainer.py
|   `-- xai_runner.py
|-- .gitignore
|-- environment.yml
|-- main.py
`-- README.md
```

Important version-control notes:

- the `data/` directory exists in the repository, but dataset files are intentionally ignored and must be provided locally
- the `results/` directory is intentionally ignored because it can become very large
- `secrets/` is intentionally ignored

The repository also contains a simplified public-demo deploy package under [`deploy/`](deploy/) for Render. That package is intentionally narrower than the full local PowerToy and is designed for lightweight public hosting.

## What The Pipeline Does

At a high level, the pipeline performs the following steps:

1. Load a CSV dataset from disk.
2. Detect or receive the subject column and label column.
3. Clean and normalize the subject text.
4. Map labels to binary values.
5. Build a reproducible dataset fingerprint after preprocessing.
6. Compute or reuse cached embeddings for each selected embedding model.
7. Evaluate each embedding with each selected classifier under nested cross-validation.
8. Refit the best hyperparameter setting on the full dataset.
9. Persist artifacts, tables, metadata, and logs.
10. Run XAI for the selected winning combination unless the user disables that stage.

## Reproducibility and Persistence

The project includes several persistence mechanisms:

- dataset fingerprinting after preprocessing
- embedding cache under `results/cache/embeddings/`
- Hugging Face model cache under `results/cache/huggingface/`
- experiment checkpointing under `results/checkpoints/`
- run-specific artifacts under `results/runs/<run_id>/`
- cumulative tabulation under `results/tables/all_results.csv`

This matters because the experiment space can be large. Without these mechanisms, reruns after interruption would be unnecessarily expensive.

When a Hugging Face embedding model is already present in the local cache, the current codebase attempts to reuse it in local-only mode instead of requerying the Hub on each run.

## XAI Scope

The repository includes three explanation layers in the current codebase:

- word ablation / leave-one-out attribution
- SHAP-based explanation
- natural-language explanation generation for the demo and explanation outputs

The current demo application also includes a semantic visualization layer:

- 2D PCA projection built from real cached embeddings of the underlying dataset
- class-colored scatter plot for phishing and legitimate subjects
- interactive inspection of local neighbors, per-point similarity, and class centroids
- a strongly highlighted marker for the subject currently being analyzed
- dashed lines from the analyzed subject to each class centroid, colored by class (red to phishing centroid, green to legitimate centroid)
- a neighbor slider and local-neighborhood emphasis in the interactive view
- a semantic reading note appended to the reasoning panel
- the same semantic view reconstructed inside the history-detail modal
- PDF export that includes the semantic reading and, when available, a print-oriented rendering of the scatter

The XAI stage is intended to help inspect model behavior, not to prove causal linguistic structure. The explanations are post hoc and should be read as diagnostic evidence about model behavior under the current pipeline, not as a guarantee of semantic truth.

## Frontend Analysis Artifacts

The PowerToy is not treated as an ephemeral demo only. Each frontend analysis is now assigned a dedicated artifact folder under:

```text
results/frontend_analyses/<analysis_id>/
```

For each analysis, the system persists the structured result and related reportable artifacts generated by the frontend/backend flow. Depending on what has already been rendered or exported, the folder can include:

- `analysis_result.json`
- `metadata.json`
- `processing_log.txt`
- `reasoning.txt`
- `keywords.json`
- `synthetic_email_body.txt`
- `semantic_map.json`
- `semantic_scatter.png`
- `forensic_report.pdf`

This is intentionally separate from training runs under `results/runs/`. Training artifacts document model development. Frontend analysis artifacts document operator-facing case studies and reportable examples derived from the deployed PowerToy flow.

## PowerToy Behavior

The FastAPI PowerToy under `demo_app/` is meant to be an artifact-generating inspection surface, not only a transient demo.

For each analyzed subject, the current interface can provide:

- leave-one-out keyword attribution in the browser
- a semantic scatter plot based on real dataset subjects and real embeddings
- hover inspection of individual points with similarity and distance to the analyzed subject
- nearest-neighbor and centroid summaries by class
- a semantic reading note attached to the textual reasoning
- a reconstructed history-detail view for prior analyses
- PDF export with the corresponding semantic interpretation

When the semantic map is available, the interface is intended to answer two different questions:

- what did the classifier predict?
- where does this subject sit relative to real phishing and legitimate subjects from the selected run?

The semantic plot should still be interpreted as an inspection aid, not as the classifier itself.

## Quick Start

Create the environment:

```bash
conda env create -f environment.yml
conda activate xai
```

Run the full pipeline:

```bash
python main.py
```

Show the CLI help:

```bash
python main.py --help
```

Run only a subset of embeddings:

```bash
python main.py --embeddings sentence-transformers/all-mpnet-base-v2 BAAI/bge-m3
```

Run only a subset of classifiers:

```bash
python main.py --classifiers logistic_regression linear_svc
```

Run training without the XAI stage:

```bash
python main.py --skip-xai
```

More detailed operational guidance is in [`docs/README_TECHNICAL.md`](docs/README_TECHNICAL.md).

## Inputs and Assumptions

The default dataset path used in the repository examples is:

```text
data/dataset.csv
```

The repository does not distribute any dataset file. The `data/` directory is kept so collaborators can place a local dataset there without changing project structure. Any documentation reference to `data/dataset.csv` should be read as a placeholder path, not as the name of a distributed dataset.

The loader expects:

- one subject-like text column
- one label-like column

Supported default subject-like names:

- `subject`
- `asunto`
- `title`

Supported default label-like names:

- `label`
- `class`
- `target`
- `is_phishing`

Supported label values include:

- phishing-like: `phishing`, `phish`, `spam`, `malicious`, `1`, `true`, `yes`
- legitimate-like: `legitimate`, `legit`, `ham`, `benign`, `0`, `false`, `no`

## Current Limitations

The project should be understood with the following limitations in mind:

- it operates on subject lines only, not full email bodies
- some large embedding models may be skipped on CPU hardware
- the run space can become expensive with the full 13 x 11 configuration
- XAI outputs are post hoc explanations of the trained pipeline, not direct causal explanations
- the repository currently has no automated test suite
- the demo application depends on previously generated run artifacts and result tables
- the demo semantic map is a visual projection for interpretation; distances shown in the UI are computed in the original embedding space, not in 2D alone

## Demo Behavior

The FastAPI PowerToy under `demo_app/` currently provides:

- leave-one-out keyword attribution in the browser
- a semantic scatter plot based on real dataset subjects and real embeddings
- hover inspection of individual points with similarity and distance to the analyzed subject
- nearest-neighbor and centroid summaries by class
- a visible, explicitly labeled marker for the analyzed subject
- dashed centroid lines in the scatter: one red line to the phishing centroid and one green line to the legitimate centroid
- natural-language reasoning in Spanish
- synthetic email-body generation in the same detected language as the analyzed subject
- per-analysis artifact persistence under `results/frontend_analyses/`

The semantic map should be read as an interpretive aid. The classifier decision is still produced by the trained pipeline; the 2D projection is a visualization of the embedding space, not the classifier itself.

## What This Repository Is For

This repository is suitable for:

- controlled experimentation
- comparative benchmarking across embeddings and classifiers
- production of artifacts for analysis, inspection, and reporting
- forensic inspection of model behavior on short subjects

This repository is not, in its current state, a drop-in production phishing defense system.

## Recommended Reading Order

For a new collaborator, the most useful reading order is:

1. [`README.md`](README.md)
2. [`docs/README_TECHNICAL.md`](docs/README_TECHNICAL.md)
3. [`docs/README_THEORETICAL.md`](docs/README_THEORETICAL.md)
4. [`config/experiment.py`](config/experiment.py)
5. [`main.py`](main.py)

## License

This repository is released under the [MIT License](LICENSE).
