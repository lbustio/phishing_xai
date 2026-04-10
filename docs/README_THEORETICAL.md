# Theoretical README

This document explains the methodological logic of `phishing_xai`. It is meant for collaborators who need to understand what problem the repository addresses, what assumptions it makes, how the experiment is structured, and how to interpret the outputs responsibly.

For installation and execution details, read [`README_TECHNICAL.md`](README_TECHNICAL.md). For the repository-facing overview, read [`../README.md`](../README.md).

## 1. Problem Definition

The repository addresses binary phishing detection from subject lines only.

Input:

- a short text string corresponding to an email subject or closely related short message title

Output:

- a binary prediction: `legitimate` or `phishing`

This is a deliberately constrained problem statement. The codebase does not claim to model the full email, headers, sender infrastructure, HTML content, attachment behavior, or user context.

## 2. Why Subject-Only Detection Matters

A subject line can already contain strong attack cues:

- urgency
- account warnings
- billing prompts
- password-reset language
- reward or payment bait
- institutional impersonation cues

At the same time, subject lines are difficult objects for modeling because they are:

- short
- noisy
- often ambiguous
- easy to paraphrase
- frequently missing the broader context that would disambiguate intent

That makes the task suitable for a comparative representation study: if the text is short and sparse, the quality of the embedding layer matters a great deal.

## 3. Modeling Philosophy

The repository uses a modular architecture:

```text
subject text -> embedding model -> classical classifier -> explanation layer
```

This design reflects several practical goals:

- compare representation quality separately from classifier choice
- avoid entangling model architecture and task learning into a single opaque end-to-end system
- preserve the ability to inspect and swap each stage independently
- support a wide range of classical classifiers over a shared embedding space

The project therefore treats embeddings as reusable feature extractors and downstream classifiers as the task-specific decision layer.

## 4. Why Multiple Embeddings

The repository is configured to compare `13` embeddings spanning several families:

- lightweight transformer encoders
- contrastive sentence encoders
- instruction-tuned embedding models
- large LLM-based embedding backends

This matters because short phishing subjects can benefit from different representation properties:

- lexical sensitivity
- semantic similarity structure
- instruction alignment
- robustness to short, imperative, or adversarial phrasing

The repository does not assume in advance that one family is universally superior. The experiment is set up to test that empirically.

## 5. Why Multiple Classifiers

The current configuration includes `11` classifiers:

- `logistic_regression`
- `linear_svc`
- `svm_rbf`
- `sgd_classifier`
- `random_forest`
- `extra_trees`
- `mlp`
- `knn`
- `decision_tree`
- `gaussian_nb`
- `lda`

These are not redundant for methodological purposes. They cover distinct assumptions:

- linear decision boundaries
- maximum-margin separation
- non-linear kernel separation
- stochastic large-scale linear fitting
- tree-based partitioning
- instance-based neighborhood classification
- simple probabilistic generative assumptions
- shallow learned non-linearity

This breadth is useful because the embedding layer may produce spaces whose separability is:

- approximately linear for some models
- curved or clustered for others
- neighborhood-structured rather than globally linear

## 6. Why Nested Cross-Validation

The repository uses nested cross-validation with:

- `5` outer folds
- `3` inner folds

This is a central design choice.

### 6.1 The Problem With A Single CV Loop

If hyperparameters are tuned and reported on the same resampling loop, performance estimates become optimistic. The model-selection procedure leaks into the model-assessment procedure.

### 6.2 What Nested CV Fixes

The current design separates:

- inner loop: hyperparameter selection
- outer loop: external performance estimation

This produces a more defensible estimate of generalization for comparative experiments.

### 6.3 Why The Repository Uses 5x3 Instead of 10-Fold

The current implementation uses `5x3`, not `10x10` or `10-fold` single-loop CV. This is a practical compromise between:

- methodological rigor
- runtime cost
- the already large Cartesian product of embeddings and classifiers

The design choice is explicit and visible in [`../config/experiment.py`](../config/experiment.py).

## 7. Why `f1_macro` Is The Primary Metric

The primary metric is:

- `f1_macro`

This is appropriate when the two classes should both matter in evaluation and when plain accuracy could hide asymmetric behavior.

The repository also records:

- `f1_weighted`
- `accuracy`
- `precision_macro`
- `recall_macro`
- `roc_auc`

The logic is:

- use `f1_macro` for model selection
- retain other metrics for fuller interpretation

## 8. Data Processing Assumptions

The loader:

- reads a CSV
- detects subject and label columns
- drops rows with missing subject or label
- normalizes labels to binary form
- cleans control characters and redundant whitespace
- removes empty subjects after cleaning

The dataset fingerprint is computed after preprocessing. This is important because the fingerprint is meant to identify the actual experimental data seen by the model, not the raw file before cleaning.

## 9. Reproducibility Logic

The repository includes several reproducibility mechanisms:

- centralized configuration
- deterministic random seed settings
- stable dataset fingerprinting
- persistent embedding cache
- persistent experiment checkpoints
- run manifests
- saved hyperparameter selections
- fold-level nested-CV details

This is not merely for convenience. In a large search space, reruns after interruption can otherwise change the practical experiment trajectory, waste compute, or make later reconstruction difficult.

## 10. Embedding Cache and Checkpointing

The repository distinguishes two kinds of persistence:

### 10.1 Embedding Cache

Embeddings are cached by:

- dataset fingerprint
- embedding identifier

If the same prepared dataset and embedding model are reused, the pipeline can skip recomputation.

### 10.2 Experiment Checkpoint

Classifier evaluations are checkpointed by:

- dataset fingerprint
- selected embedding set
- selected classifier set

This lets the pipeline resume after interruption and avoids recomputing completed embedding-classifier combinations.

## 11. Explanation Strategy

The repository includes multiple explanation layers.

### 11.1 Word Ablation / Leave-One-Out Attribution

The word-ablation layer perturbs the subject by removing or altering local textual elements and observes the change in the model output. This is useful because the final model is not natively token-based in the classifier stage; it is a classifier over embeddings. Ablation provides a practical way to estimate local contribution at the word level.

### 11.2 SHAP

The SHAP layer provides another post hoc perspective on how the pipeline responds to local text perturbations and how contributions aggregate across examples.

### 11.3 Natural-Language Explanation

The natural-language explanation layer translates algorithmic outputs into a more readable narrative. This is useful for demos and interpretation support, but it should not be mistaken for an independent source of truth. It is downstream of the underlying attribution signals.

In the current demo implementation, the explanation and body-generation policies are intentionally separated:

- the reasoning text is always generated in Spanish
- the synthetic email body is generated in the detected language of the analyzed subject

This is a presentation rule for operator usability. It does not change the underlying classifier output.

### 11.4 Interactive Semantic Projection

The current PowerToy also includes an interactive semantic map:

- the points correspond to real subjects from the dataset used by the selected run
- the coordinates come from a 2D PCA projection of real embeddings
- phishing and legitimate examples are shown as different color regions
- the analyzed subject is inserted into the same projected space as a separate marker
- nearest neighbors and class centroids are exposed to the user as interpretive cues
- the semantic reading is summarized in prose inside the reasoning panel
- the same semantic evidence can be reconstructed later from the history-detail view

This view is useful because it gives an operator a geometric intuition for local structure in the embedding space. However, it must be interpreted carefully.

- the map is not the classifier itself
- the 2D layout is a projection, not the full feature space
- the scientifically relevant similarity values should be those computed in the original embedding space

For that reason, the current demo reports cosine similarity and cosine distance relative to the analyzed subject while using the 2D panel only as a visual aid.

The repository also treats frontend analyses as reportable artifacts rather than transient UI events. Each PowerToy analysis can therefore leave behind a persistent case folder under `results/frontend_analyses/`, which is separate from the training-run artifacts under `results/runs/`.

## 12. Important Interpretation Caveat

The project produces explanations of model behavior, not explanations of the world.

That distinction matters:

- if a keyword receives high attribution, that means the trained pipeline used it strongly in that instance
- it does not automatically mean the keyword is inherently malicious in every context
- it does not prove causal attack semantics
- if two points appear close in the 2D semantic map, that is an interpretive cue, not by itself proof of classifier causality

In other words, the XAI outputs are diagnostic and analytical, not ontological proof.

## 13. Representativeness of XAI Examples

Earlier documentation language often describes representative examples as a stratified mix of true positives, true negatives, false positives, and false negatives. The current implementation in [`../src/xai_runner.py`](../src/xai_runner.py) does not yet do that. It currently samples representative indices randomly with a fixed seed.

That means the current behavior is:

- reproducible
- simple
- not yet error-profile-aware

This is important to document honestly because the theoretical intention and the implemented selection policy are not identical.

## 14. Why Classical Classifiers Over Embeddings

This design has several methodological advantages:

- it decouples representation learning from classification
- it allows direct comparison across downstream decision rules
- it makes hyperparameter grids explicit and inspectable
- it permits some models with more straightforward explanations than end-to-end deep classifiers

The trade-off is that the system does not fine-tune embeddings on the dataset task itself. That may leave performance on the table relative to carefully tuned end-to-end approaches, but it improves modularity and comparability.

## 15. Why Include Simple Baselines

The inclusion of models such as:

- `gaussian_nb`
- `lda`
- `decision_tree`

is methodologically useful even if they are unlikely to dominate the leaderboard.

Simple baselines serve several roles:

- they expose whether the task is already separable under simple assumptions
- they prevent overclaiming gains from more complex models
- they give interpretability anchors
- they help detect whether a sophisticated pipeline is truly necessary

## 16. Why Include Stronger Non-Linear Baselines

The inclusion of models such as:

- `svm_rbf`
- `random_forest`
- `extra_trees`
- `mlp`

tests a different hypothesis: whether the embedding space contains discriminative information that is not recoverable with a purely linear boundary.

This is especially relevant if phishing subjects cluster in local pockets or exhibit feature interactions that linear models cannot recover cleanly.

## 17. Operational Honesty About Constraints

The current repository should be described soberly:

- it is a serious experimental framework
- it is not a finished production security appliance
- it supports comparative modeling and explanation
- it depends on local datasets and local artifacts that are not distributed in Git
- it currently lacks automated tests

Those statements do not weaken the project. They make its claims more credible and auditable.

## 18. What The Repository Can Support Well

The current framework is well suited for:

- comparative embedding studies
- classifier ablation studies
- reproducible experiment tracking
- artifact production for later reporting
- inspection of subject-level phishing decisions

## 19. What It Does Not Yet Fully Cover

The current framework does not fully cover:

- full-email phishing detection
- header-based or infrastructure-based detection
- end-to-end fine-tuning of foundation models
- calibration analysis as a first-class experiment axis
- formal robustness evaluation under adversarial paraphrasing
- automated regression testing of the codebase

## 20. Suggested Reading Inside The Codebase

For methodological understanding, the most relevant files are:

- [`../config/experiment.py`](../config/experiment.py)
- [`../src/data_loader.py`](../src/data_loader.py)
- [`../src/trainer.py`](../src/trainer.py)
- [`../src/embedding_store.py`](../src/embedding_store.py)
- [`../src/utils/checkpoint.py`](../src/utils/checkpoint.py)
- [`../src/xai_runner.py`](../src/xai_runner.py)

## 21. Summary

The repository is best understood as a reproducible comparative framework for explainable phishing detection from short subjects. Its main strengths are:

- explicit experiment configuration
- systematic comparison across embeddings and classifiers
- nested cross-validation
- persistence of heavy intermediate computations
- explanation outputs beyond raw metrics

Its main limits are equally explicit:

- subject-only scope
- computational cost for the full configuration
- post hoc rather than intrinsic explanation for most winning pipelines
- missing automated tests

Those boundaries should be kept visible whenever the project is presented, documented, or submitted for evaluation.
