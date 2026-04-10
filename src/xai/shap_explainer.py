"""
src/xai/shap_explainer.py
=========================
SHAP (SHapley Additive exPlanations) for the phishing detector.

How SHAP works here
-------------------
SHAP computes the marginal contribution of each word to the model's prediction
by averaging over all possible orderings of words (Shapley values from game
theory).  Unlike LIME, SHAP values satisfy three mathematical guarantees:
  - Efficiency:  contributions sum to the output difference from the baseline.
  - Symmetry:    words with equal contributions receive equal SHAP values.
  - Linearity:   contributions are additive across features.

Pipeline integration
--------------------
Because the model is a black-box pipeline (text → embedding → classifier),
we use shap.Explainer with a Text masker.  The masker removes words one at a
time (or in groups) and feeds the modified texts through the full pipeline to
observe how the prediction changes.

This is functionally equivalent to a KernelSHAP with a text perturbation
strategy – it is the correct approach for explain-by-ablation on text input.

Output artefacts per instance
------------------------------
  results/xai/shap/shap_waterfall_{idx}_{label}.png  – Per-instance waterfall
  results/xai/shap/shap_bar_aggregate.png             – Global feature bar chart
  results/xai/shap/shap_beeswarm.png                  – Distribution of SHAP values
  results/xai/shap/shap_values.json                   – Raw SHAP values + metadata

Note on performance
-------------------
The Text masker re-encodes and re-classifies many perturbed versions of each
subject.  This is computationally expensive (~30–120 s per instance depending
on the embedding model and CPU/GPU speed).  Limit n_xai_examples accordingly.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config.experiment import XAI_CONFIG

logger = logging.getLogger("phishing_xai.xai.shap")


class SHAPExplainer:
    """
    SHAP Text explainer wrapper for the phishing detection pipeline.

    Parameters
    ----------
    output_dir  : Path       – Directory for SHAP artefacts.
    class_names : List[str]  – Class label strings.
    """

    def __init__(
        self,
        output_dir:  Path,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.output_dir  = output_dir
        self.class_names = class_names or XAI_CONFIG["class_names"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def explain_instance(
        self,
        text:         str,
        predict_fn:   Callable,
        instance_idx: int,
        true_label:   Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for a single email subject.

        Parameters
        ----------
        text         : str      – Raw email subject.
        predict_fn   : Callable – Function (List[str]) → np.ndarray (n, 2).
                                  Returns P(legitimate), P(phishing) per text.
        instance_idx : int      – Index for artefact naming.
        true_label   : int, opt – Ground truth label (for annotation only).

        Returns
        -------
        dict  – {'text', 'shap_values', 'words', 'prediction', 'confidence'}
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "The 'shap' package is required. Install with: pip install shap"
            )

        logger.info(
            f"  SHAP explaining instance {instance_idx}: "
            f"'{text[:80]}{'...' if len(text)>80 else ''}'"
        )

        # ── Wrap predict_fn to return only P(phishing) ────────────────────────
        # shap.Explainer with a scalar output is simpler to visualise and
        # avoids multi-output SHAP issues with text maskers.
        def phishing_prob(texts_list: List[str]) -> np.ndarray:
            probs = predict_fn(texts_list)
            return probs[:, 1]   # Column 1 = P(phishing)

        # ── Build explainer with Text masker ──────────────────────────────────
        # The regex r"\W+" splits on non-word characters, treating each word
        # as an independent feature.  This aligns with how LIME tokenises text.
        masker   = shap.maskers.Text(tokenizer=r"\W+")
        explainer = shap.Explainer(phishing_prob, masker)

        try:
            shap_values = explainer([text])
        except Exception as exc:
            logger.error(
                f"  SHAP failed on instance {instance_idx} ({exc}). "
                "This instance will be skipped in SHAP analysis."
            )
            return {"text": text, "error": str(exc)}

        # ── Extract values and token names ────────────────────────────────────
        sv    = shap_values[0]              # SHAPExplanation for this instance
        data  = sv.data                    # Token strings (numpy array)
        vals  = sv.values                  # SHAP values per token

        # Get prediction info
        probs    = predict_fn([text])[0]
        pred_cls = int(np.argmax(probs))
        conf     = float(probs[pred_cls])
        pred_lbl = self.class_names[pred_cls]
        true_lbl = self.class_names[true_label] if true_label is not None else "unknown"

        logger.info(
            f"    Prediction: {pred_lbl} (confidence={conf:.3f}), True: {true_lbl}"
        )

        # Log top contributing words
        if vals is not None and data is not None:
            sorted_pairs = sorted(zip(vals, data), key=lambda x: abs(x[0]), reverse=True)
            for weight, word in sorted_pairs[:min(5, len(sorted_pairs))]:
                direction = "↑ PHISHING" if weight > 0 else "↓ LEGITIMATE"
                logger.info(f"    SHAP '{word}': {direction}  ({weight:+.4f})")

        # ── Save waterfall plot ────────────────────────────────────────────────
        label_tag = "phish" if pred_cls == 1 else "legit"
        wf_path   = self.output_dir / f"shap_waterfall_{instance_idx:04d}_{label_tag}.png"
        self._save_waterfall(shap_values, wf_path, text, pred_lbl, conf, true_lbl)

        result = {
            "text":        text,
            "instance_idx": instance_idx,
            "prediction":  pred_cls,
            "confidence":  conf,
            "true_label":  true_label,
            "words":       data.tolist() if hasattr(data, "tolist") else list(data),
            "shap_values": vals.tolist() if hasattr(vals, "tolist") else list(vals),
        }
        return result

    def explain_batch(
        self,
        texts:      List[str],
        predict_fn: Callable,
        labels:     Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute SHAP values for multiple instances and produce aggregate plots.

        Parameters
        ----------
        texts      : List[str]  – Email subjects to explain.
        predict_fn : Callable   – Predict-proba wrapper.
        labels     : List[int]  – Ground-truth labels (optional).

        Returns
        -------
        List[dict]  – SHAP results per instance.
        """
        n = len(texts)
        logger.info(
            f"Starting SHAP explanation of {n} instance(s)."
        )

        results = []
        for i, text in enumerate(texts):
            true_lbl = labels[i] if labels else None
            r = self.explain_instance(text, predict_fn, i, true_lbl)
            results.append(r)

        # ── Global plots ───────────────────────────────────────────────────────
        self._save_global_bar_plot(results)
        self._save_beeswarm(results)
        self._save_shap_json(results)

        logger.info(
            f"SHAP explanations complete. Artefacts saved to: {self.output_dir}"
        )
        return results

    # ── Private plot helpers ───────────────────────────────────────────────────

    def _save_waterfall(self, shap_values, path: Path, text, pred_lbl, conf, true_lbl) -> None:
        """Save a SHAP waterfall plot for one instance."""
        try:
            import shap
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], show=False, max_display=15)
            plt.title(
                f"SHAP Waterfall — Phishing Probability\n"
                f'Subject: "{text[:65]}{"..." if len(text)>65 else ""}"\n'
                f"Prediction: {pred_lbl} ({conf:.1%}) | True: {true_lbl}",
                fontsize=9, pad=8,
            )
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.debug(f"    SHAP waterfall saved: {path.name}")
        except Exception as exc:
            logger.warning(f"    Could not save SHAP waterfall: {exc}")
            plt.close("all")

    def _save_global_bar_plot(self, results: List[Dict]) -> None:
        """
        Aggregate SHAP values across all explained instances and plot global
        word importance as a horizontal bar chart.
        """
        from collections import defaultdict

        word_vals: Dict[str, List[float]] = defaultdict(list)
        for r in results:
            if "error" in r or not r.get("words"):
                continue
            for word, val in zip(r["words"], r["shap_values"]):
                word_vals[word.lower()].append(val)

        if not word_vals:
            return

        word_mean = {w: float(np.mean(vs)) for w, vs in word_vals.items()}
        top_n = min(20, len(word_mean))
        sorted_words = sorted(word_mean.keys(), key=lambda w: abs(word_mean[w]), reverse=True)[:top_n]
        sorted_words = sorted(sorted_words, key=lambda w: word_mean[w])

        means  = [word_mean[w] for w in sorted_words]
        colors = ["#e74c3c" if m > 0 else "#2ecc71" for m in means]

        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.45)))
        ax.barh(range(len(sorted_words)), means, color=colors, edgecolor="white")
        ax.set_yticks(range(len(sorted_words)))
        ax.set_yticklabels(sorted_words, fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(
            "Mean |SHAP value| (contribution to phishing probability)\n"
            "Red = phishing indicator | Green = legitimate indicator",
            fontsize=10
        )
        ax.set_title(
            f"Global SHAP Word Importance\n"
            f"(Top {top_n} words averaged across {len(results)} explained instances)",
            fontsize=11, pad=10,
        )
        plt.tight_layout()
        path = self.output_dir / "shap_bar_aggregate.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Global SHAP bar chart saved: {path.name}")

    def _save_beeswarm(self, results: List[Dict]) -> None:
        """
        Create a simplified dot-plot showing the distribution of SHAP values
        for the most important words across all explained instances.
        """
        from collections import defaultdict

        word_vals: Dict[str, List[float]] = defaultdict(list)
        for r in results:
            if "error" in r or not r.get("words"):
                continue
            for word, val in zip(r["words"], r["shap_values"]):
                word_vals[word.lower()].append(val)

        if not word_vals:
            return

        #word_mean = {w: float(np.mean(abs(v) for v in vs)) for w, vs in word_vals.items()}
        word_mean = {w: float(np.mean([abs(v) for v in vs])) for w, vs in word_vals.items()}
        
        top_n = min(15, len(word_mean))
        top_words = sorted(word_mean.keys(), key=lambda w: word_mean[w], reverse=True)[:top_n]

        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.5)))
        for i, word in enumerate(reversed(top_words)):
            vals = word_vals[word]
            jitter = np.random.default_rng(42).uniform(-0.25, 0.25, len(vals))
            c = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]
            ax.scatter(vals, [i + j for j in jitter], c=c, alpha=0.7, s=30)

        ax.set_yticks(range(top_n))
        ax.set_yticklabels(list(reversed(top_words)), fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (contribution to phishing probability)", fontsize=10)
        ax.set_title(
            "SHAP Value Distribution per Word\n"
            "(Each dot = one explained instance; red = phishing, green = legitimate)",
            fontsize=11, pad=10,
        )
        plt.tight_layout()
        path = self.output_dir / "shap_beeswarm.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"SHAP beeswarm plot saved: {path.name}")

    def _save_shap_json(self, results: List[Dict]) -> None:
        """Persist raw SHAP values and metadata to JSON."""
        clean = []
        for r in results:
            entry = {}
            for k, v in r.items():
                if hasattr(v, "tolist"):
                    entry[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    entry[k] = v.item()
                else:
                    entry[k] = v
            clean.append(entry)
        path = self.output_dir / "shap_values.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(clean, fh, indent=2, ensure_ascii=False)
        logger.info(f"Raw SHAP values saved: {path.name}")
