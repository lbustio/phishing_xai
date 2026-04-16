# Contributing

This document is a short operational guide for collaborators working on `phishing_xai`.

It focuses on repository hygiene and local workflow. For full setup and execution details, see:

- [`README.md`](README.md)
- [`docs/README_TECHNICAL.md`](docs/README_TECHNICAL.md)
- [`docs/README_THEORETICAL.md`](docs/README_THEORETICAL.md)

## Scope

Contributions are welcome in areas such as:

- experiment configuration
- embedding and classifier evaluation
- XAI analysis and visualization
- logging and reproducibility improvements
- documentation
- demo application improvements
- tests and validation utilities

## Local Setup

Create the environment defined in [`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml
conda activate xai
```

## Repository Rules

Please keep the following local-only assets out of version control:

- dataset files inside `data/`
- all generated outputs under `results/`
- any tokens or credentials under `secrets/`
- cache files, notebooks checkpoints, and Python bytecode

The repository is already configured to ignore these paths. Do not force-add them.

## Data Handling

The `data/` directory is intentionally kept in the repository structure, but dataset contents are not distributed.

When working with datasets:

- place the CSV locally under `data/` or pass a custom path with `--data`
- do not commit non-distributable data
- do not commit derivative exports if they expose restricted source data

## Secrets Handling

The `secrets/` directory is local-only.

Expected files may include:

- `secrets/groq.txt`
- `secrets/huggingface.txt`

Never commit tokens, screenshots of tokens, or copied credential values into issues, pull requests, logs, or documentation.

## Results and Artifacts

The `results/` tree can become very large and is intentionally ignored by Git.

That includes:

- embedding cache
- run logs
- checkpoints
- trained models
- SHAP and LIME outputs
- cumulative results tables generated locally

If a result needs to be preserved in Git, summarize it in documentation or export a small derived table intentionally and review whether it is safe to publish.

## Coding Expectations

When changing the codebase:

- keep configuration explicit
- preserve reproducibility mechanisms
- prefer clear, auditable logic over hidden automation
- document non-obvious assumptions
- avoid changing experiment semantics silently

If you change the active experimental configuration in [`config/experiment.py`](config/experiment.py), also update the documentation so the repository description stays aligned with the code.

## Recommended Validation

Before publishing a change, it is a good idea to:

1. run a small subset experiment
2. verify that outputs land in the expected `results/` paths
3. confirm that no ignored local files were accidentally staged
4. review documentation links and configuration counts if they changed

Example quick check:

```bash
python main.py --embeddings sentence-transformers/all-mpnet-base-v2 --classifiers logistic_regression --skip-xai
```

## Pull Request Hygiene

A good contribution should make it easy to answer:

- what changed
- why it changed
- whether experiment behavior changed
- whether documentation was updated
- whether local-only artifacts stayed out of Git

## Questions

If a change affects:

- dataset assumptions
- evaluation protocol
- XAI semantics
- artifact formats
- repository hygiene rules

please document that explicitly in the change description.
