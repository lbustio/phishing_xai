"""
src/xai/lime_explainer.py
=========================
LIME (Local Interpretable Model-agnostic Explanations) for the phishing detector.

How LIME works here
-------------------
LIME explains why the model classified a specific email subject as phishing or
legitimate.  It does this by:

  1. Taking the original subject string  (e.g. "URGENT: Verify your PayPal account").
  2. Creating hundreds of perturbed versions by randomly masking individual words.
  3. Running each perturbed version through the full pipeline:
       perturbed_text → embedding model → trained classifier → P(phishing)
  4. Fitting a simple local linear model to approximate which words
     PUSHED the prediction towards phishing (positive weight)
     or towards legitimate (negative weight).

The result is a word-level feature importance bar chart that is directly
interpretable: "the word 'URGENT' increased the phishing probability by X%".

This is much more faithful to the actual model than gradient-based methods
because LIME treats the entire pipeline as a black box.

Output artefacts
----------------
For each explained instance:
  results/xai/lime/lime_instance_{idx}_{label}.png  – Individual bar chart
  results/xai/lime/lime_aggregate_importance.png    – Aggregated word importance
  results/xai/lime/lime_explanations.json           – Raw explanation data
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for headless servers)
import matplotlib.pyplot as plt
import numpy as np

from config.experiment import XAI_CONFIG

logger = logging.getLogger("phishing_xai.xai.lime")


class LIMEExplainer:
    """
    Wrapper around lime.lime_text.LimeTextExplainer adapted for the phishing
    detection pipeline.

    Parameters
    ----------
    output_dir   : Path  – Directory where explanation artefacts are saved.
    class_names  : list  – Labels for the two classes (default from config).
    n_features   : int   – Maximum number of features (words) to highlight.
    n_samples    : int   – Number of perturbed samples LIME generates internally.
    kernel_width : float – Width of the exponential similarity kernel.
    """

    def __init__(
        self,
        output_dir:  Path,
        class_names: Optional[List[str]] = None,
        n_features:  int   = XAI_CONFIG["n_lime_features"],
        n_samples:   int   = XAI_CONFIG["n_lime_samples"],
        kernel_width: float = XAI_CONFIG["lime_kernel_width"],
    ) -> None:
        self.output_dir  = output_dir
        self.class_names = class_names or XAI_CONFIG["class_names"]
        self.n_features  = n_features
        self.n_samples   = n_samples
        self.kernel_width = kernel_width

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_explainer(self):
        """Lazily instantiate LimeTextExplainer (defers import error if not installed)."""
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError:
            raise ImportError(
                "The 'lime' package is required for LIME explanations. "
                "Install with: pip install lime"
            )
        return LimeTextExplainer(
            class_names  = self.class_names,
            kernel_width = self.kernel_width,
        )

    def explain_instance(
        self,
        text:        str,
        predict_fn:  Callable,
        instance_idx: int,
        true_label:  Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Explain a single email subject prediction.

        Parameters
        ----------
        text         : str      – The raw email subject string.
        predict_fn   : Callable – Function that takes List[str] → np.ndarray
                                  of shape (n, 2) with class probabilities.
        instance_idx : int      – Index in the dataset (for filename generation).
        true_label   : int, opt – Ground-truth label (0 or 1), for plot annotation.

        Returns
        -------
        dict  – {'text', 'prediction', 'confidence', 'features': [(word, weight), ...]}
        """
        explainer = self._get_explainer()

        logger.info(
            f"  LIME explaining instance {instance_idx}: '{text[:80]}...' "
            if len(text) > 80 else
            f"  LIME explaining instance {instance_idx}: '{text}'"
        )

        def sensitive_predict_fn(texts):
            probs = predict_fn(texts)
            # Logit scaling to avoid saturation in high-confidence models
            # This makes LIME sensitive to the difference between 0.999 and 0.9999
            eps = 1e-7
            probs = np.clip(probs, eps, 1 - eps)
            logits = np.log(probs / (1 - probs))
            # Rescale logits to [0, 1] range to satisfy LIME's internal expectations if needed,
            # but actually LIME just needs a signal. We'll use a sigmoid-like squash to keep them bounded.
            return 1 / (1 + np.exp(-logits)) # Back to probs but with expanded gradients

        exp = explainer.explain_instance(
            text_instance = text,
            classifier_fn = sensitive_predict_fn,
            num_features  = self.n_features,
            num_samples   = self.n_samples,
            top_labels    = 2,
        )

        # Extract probabilities for this instance
        probs    = predict_fn([text])[0]
        pred_cls = int(np.argmax(probs))
        conf     = float(probs[pred_cls])
        pred_lbl = self.class_names[pred_cls]
        true_lbl = self.class_names[true_label] if true_label is not None else "unknown"

        logger.info(
            f"    Prediction: {pred_lbl} (confidence={conf:.3f}), "
            f"True label: {true_lbl}"
        )

        # Feature list: [(word, weight_for_phishing_class)]
        features = exp.as_list(label=1)   # Label 1 = Phishing
        for word, weight in features:
            direction = "PHISHING ↑" if weight > 0 else "LEGITIMATE ↓"
            logger.info(
                f"    Word '{word}': {direction}  (weight={weight:+.4f})"
            )

        # ── Save individual plot ───────────────────────────────────────────────
        label_tag = "phish" if pred_cls == 1 else "legit"
        plot_path = self.output_dir / f"lime_instance_{instance_idx:04d}_{label_tag}.png"
        self._save_instance_plot(exp, text, pred_lbl, conf, true_lbl, plot_path)

        return {
            "text":        text,
            "instance_idx": instance_idx,
            "prediction":  pred_cls,
            "confidence":  conf,
            "true_label":  true_label,
            "features":    features,
        }

    def explain_batch(
        self,
        texts:      List[str],
        predict_fn: Callable,
        labels:     Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Explain multiple instances and produce an aggregate word importance plot.

        Parameters
        ----------
        texts      : List[str]      – Email subject strings to explain.
        predict_fn : Callable       – Predict-proba wrapper.
        labels     : List[int], opt – Ground-truth labels for annotation.

        Returns
        -------
        List[dict]  – One explanation dict per instance.
        """
        n = len(texts)
        logger.info(
            f"Starting LIME explanation of {n} instances "
            f"(n_features={self.n_features}, n_samples={self.n_samples})..."
        )

        explanations = []
        for i, text in enumerate(texts):
            true_lbl = labels[i] if labels else None
            result = self.explain_instance(text, predict_fn, i, true_lbl)
            explanations.append(result)

        # ── Aggregate feature importance ───────────────────────────────────────
        self._save_aggregate_plot(explanations)
        self._save_explanations_json(explanations)

        logger.info(
            f"LIME explanations complete. "
            f"Artefacts saved to: {self.output_dir}"
        )
        return explanations

    # ── Private plot helpers ───────────────────────────────────────────────────

    def _save_instance_plot(self, exp, text, pred_lbl, conf, true_lbl, path: Path) -> None:
        """Save a bar chart of feature importances for one LIME explanation."""
        features = exp.as_list(label=1)
        if not features:
            return

        words   = [f[0] for f in features]
        weights = [f[1] for f in features]
        colors  = ["#e74c3c" if w > 0 else "#2ecc71" for w in weights]

        fig, ax = plt.subplots(figsize=(9, max(4, len(words) * 0.45)))
        bars = ax.barh(range(len(words)), weights, color=colors, edgecolor="white")
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("LIME Weight (positive → Phishing, negative → Legitimate)", fontsize=10)
        ax.set_title(
            f"LIME Explanation\n"
            f'Subject: "{text[:70]}{"..." if len(text)>70 else ""}"\n'
            f"Prediction: {pred_lbl} ({conf:.1%}) | True: {true_lbl}",
            fontsize=10, pad=10,
        )
        plt.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"    LIME individual plot saved: {path.name}")

    def _save_aggregate_plot(self, explanations: List[Dict]) -> None:
        """
        Aggregate word importances across all explained instances and plot
        the top-N most influential words globally.

        Positive aggregate weight → word promotes PHISHING classification.
        Negative aggregate weight → word promotes LEGITIMATE classification.
        """
        from collections import defaultdict

        word_weights: Dict[str, List[float]] = defaultdict(list)
        for exp in explanations:
            for word, weight in exp.get("features", []):
                word_weights[word.lower()].append(weight)

        if not word_weights:
            return

        # Mean weight per word
        word_mean = {w: float(np.mean(ws)) for w, ws in word_weights.items()}
        word_std  = {w: float(np.std(ws))  for w, ws in word_weights.items()}

        # Select top-N by absolute mean importance
        top_n = min(20, len(word_mean))
        sorted_words = sorted(word_mean.keys(), key=lambda w: abs(word_mean[w]), reverse=True)[:top_n]
        sorted_words = sorted(sorted_words, key=lambda w: word_mean[w])

        means  = [word_mean[w] for w in sorted_words]
        stds   = [word_std[w]  for w in sorted_words]
        colors = ["#e74c3c" if m > 0 else "#2ecc71" for m in means]

        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.45)))
        ax.barh(range(len(sorted_words)), means, xerr=stds,
                color=colors, edgecolor="white", capsize=3)
        ax.set_yticks(range(len(sorted_words)))
        ax.set_yticklabels(sorted_words, fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(
            "Mean LIME Weight across explained instances\n"
            "(red = phishing indicator, green = legitimate indicator)",
            fontsize=10
        )
        ax.set_title(
            f"Aggregate LIME Word Importance\n"
            f"(Top {top_n} words across {len(explanations)} explained instances)",
            fontsize=11, pad=10,
        )
        plt.tight_layout()
        path = self.output_dir / "lime_aggregate_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Aggregate LIME importance plot saved: {path.name}")

    def _save_explanations_json(self, explanations: List[Dict]) -> None:
        """Serialise raw explanation data to JSON for downstream analysis."""
        # Convert numpy types to native Python for JSON serialisation
        clean = []
        for exp in explanations:
            clean.append({
                k: (v.item() if hasattr(v, "item") else v)
                for k, v in exp.items()
            })
        path = self.output_dir / "lime_explanations.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(clean, fh, indent=2, ensure_ascii=False)
        logger.info(f"Raw LIME explanation data saved: {path.name}")
