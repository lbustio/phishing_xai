from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from config.experiment import XAI_CONFIG

logger = logging.getLogger("phishing_xai.xai.lime")


def _build_contextual_ablation_features(
    text: str,
    predict_fn: Callable[[List[str]], np.ndarray],
    *,
    max_features: int,
    span_lengths: Sequence[int] = (1,),
) -> tuple[List[Dict[str, Any]], float]:
    token_matches = list(re.finditer(r"\b[\w'-]+\b", text, flags=re.UNICODE))
    words = [match.group(0) for match in token_matches]
    if not words:
        return [], 0.0

    def remove_span(start_idx: int, end_idx: int) -> str:
        start = token_matches[start_idx].start()
        end = token_matches[end_idx].end()
        new_text = (text[:start] + " " + text[end:]).strip()
        return re.sub(r"\s+", " ", new_text)

    candidates: List[Dict[str, Any]] = []
    variants = [text]
    for span_len in span_lengths:
        if span_len <= 0:
            continue
        for start_idx in range(len(words) - span_len + 1):
            end_idx = start_idx + span_len - 1
            phrase_tokens = words[start_idx:end_idx + 1]
            candidates.append(
                {
                    "span": (start_idx, end_idx),
                    "phrase": " ".join(phrase_tokens),
                    "tokens": phrase_tokens,
                    "length": span_len,
                }
            )
            variants.append(remove_span(start_idx, end_idx))

    probabilities = np.asarray(predict_fn(variants), dtype=float)
    base_prob = float(probabilities[0, 1])

    single_span_abs: Dict[tuple[int, int], float] = {}
    for idx, candidate in enumerate(candidates, start=1):
        delta = base_prob - float(probabilities[idx, 1])
        candidate["delta"] = delta
        candidate["abs_delta"] = abs(delta)
        if candidate["length"] == 1:
            single_span_abs[candidate["span"]] = candidate["abs_delta"]

    for candidate in candidates:
        start_idx, end_idx = candidate["span"]
        constituent = [
            single_span_abs.get((token_idx, token_idx), 0.0)
            for token_idx in range(start_idx, end_idx + 1)
        ]
        constituent_mean = float(np.mean(constituent)) if constituent else 0.0

        if candidate["length"] == 1:
            overlapping_multi = [
                other["abs_delta"]
                for other in candidates
                if other["length"] > 1 and other["span"][0] <= start_idx <= other["span"][1]
            ]
            contextual_support = max(overlapping_multi) if overlapping_multi else candidate["abs_delta"]
            support_ratio = contextual_support / max(candidate["abs_delta"], 1e-9)
            candidate["adjusted_score"] = candidate["abs_delta"] / max(1.0, support_ratio)
        else:
            interaction_gain = candidate["abs_delta"] / max(constituent_mean, 1e-9)
            phrase_bonus = 1.0 + 0.18 * (candidate["length"] - 1)
            candidate["adjusted_score"] = candidate["abs_delta"] * phrase_bonus * min(interaction_gain, 2.5)

        candidate["signed_score"] = candidate["adjusted_score"] if candidate["delta"] >= 0 else -candidate["adjusted_score"]

    candidates.sort(key=lambda item: item["adjusted_score"], reverse=True)

    selected: List[Dict[str, Any]] = []
    used_spans: List[tuple[int, int]] = []
    for candidate in candidates:
        start_idx, end_idx = candidate["span"]
        overlaps = any(not (end_idx < used_start or start_idx > used_end) for used_start, used_end in used_spans)
        if overlaps or candidate["adjusted_score"] <= 0:
            continue
        selected.append(candidate)
        used_spans.append(candidate["span"])
        if len(selected) >= max_features:
            break

    return selected, base_prob


def compute_contextual_ablation_keywords(
    text: str,
    predict_fn: Callable[[List[str]], np.ndarray],
    *,
    max_features: int = 6,
    min_share: float = 0.04,
    span_lengths: Sequence[int] = (1,),
) -> List[Dict[str, Any]]:
    selected, _ = _build_contextual_ablation_features(
        text,
        predict_fn,
        max_features=max_features,
        span_lengths=span_lengths,
    )
    total_score = sum(item["adjusted_score"] for item in selected) or 1.0
    return [
        {
            "word": item["phrase"],
            "impact": round(item["adjusted_score"] / total_score * 100, 1),
            "delta_pp": round(item["delta"] * 100, 2),
            "positive": item["delta"] > 0,
        }
        for item in selected
        if item["adjusted_score"] / total_score > min_share
    ]


def resolve_guardrailed_verdict(
    phishing_probability: float,
    keywords: Sequence[Dict[str, Any]],
    *,
    base_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Conservative verdict policy based on model probability plus attribution consistency.

    The underlying classifier remains the primary signal. We only override a
    borderline legitimate verdict when the attribution pattern shows multiple
    independent phishing-driving spans with aggregate support that rivals or
    exceeds the legitimacy-driving evidence.
    """
    if phishing_probability >= base_threshold:
        return {
            "is_phishing": True,
            "probability": float(phishing_probability),
            "decision_source": "model_threshold",
        }

    positive = [item for item in keywords if item.get("positive", True)]
    negative = [item for item in keywords if not item.get("positive", True)]
    total_positive = float(sum(float(item.get("impact", 0.0)) for item in positive))
    total_negative = float(sum(float(item.get("impact", 0.0)) for item in negative))
    max_positive = max((float(item.get("impact", 0.0)) for item in positive), default=0.0)
    max_negative = max((float(item.get("impact", 0.0)) for item in negative), default=0.0)

    # Guardrail: only fires for truly borderline cases (model already near 50%).
    # A clear legitimate verdict (< 38%) must never be overridden by attribution alone —
    # attribution shares are relative percentages, not absolute probabilities.
    borderline_alert = (
        phishing_probability >= 0.38
        and len(positive) >= 2
        and total_positive >= 55.0
        and total_positive >= total_negative * 1.5
        and max_positive >= 25.0
    )

    if borderline_alert:
        adjusted_probability = max(
            float(phishing_probability),
            min(0.62, 0.50 + max(0.0, total_positive - total_negative) / 300.0 + max_positive / 400.0),
        )
        return {
            "is_phishing": True,
            "probability": adjusted_probability,
            "decision_source": "attribution_guardrail",
        }

    return {
        "is_phishing": False,
        "probability": float(phishing_probability),
        "decision_source": "model_threshold",
    }


def compute_counterfactual_flip(
    text: str,
    predict_fn: Callable[[List[str]], np.ndarray],
    base_prob: float,
    threshold: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """Find the single word whose removal most shifts the verdict (ideally flipping it)."""
    token_matches = list(re.finditer(r"\b[\w'-]+\b", text, flags=re.UNICODE))
    words = [m.group(0) for m in token_matches]
    if not words:
        return None

    variants: List[str] = []
    for m in token_matches:
        ablated = re.sub(r"\s+", " ", (text[:m.start()] + " " + text[m.end():]).strip())
        variants.append(ablated)

    probs = np.asarray(predict_fn(variants), dtype=float)[:, 1]
    is_phishing = base_prob >= threshold

    best_idx = int(np.argmin(probs)) if is_phishing else int(np.argmax(probs))
    new_prob = float(probs[best_idx])
    flips = (is_phishing and new_prob < threshold) or (not is_phishing and new_prob >= threshold)

    return {
        "word": words[best_idx],
        "original_prob": round(base_prob * 100, 1),
        "new_prob": round(new_prob * 100, 1),
        "flips_verdict": flips,
        "original_label": "phishing" if is_phishing else "legítimo",
        "new_label": "legítimo" if is_phishing else "phishing",
    }


class LIMEExplainer:
    """Deterministic leave-one-out token attribution.

    We keep the historical class name to avoid changing downstream code and
    output filenames, but the underlying method is now word ablation instead of
    classical LIME. This is more stable for text -> embedding -> classifier
    pipelines where sampled perturbations frequently collapse to zero weights.
    """

    def __init__(
        self,
        output_dir: Path,
        class_names: Optional[List[str]] = None,
        n_features: int = XAI_CONFIG["n_lime_features"],
        n_samples: int = XAI_CONFIG["n_lime_samples"],
        kernel_width: float = XAI_CONFIG["lime_kernel_width"],
    ) -> None:
        self.output_dir = output_dir
        self.class_names = class_names or XAI_CONFIG["class_names"]
        self.n_features = n_features
        self.n_samples = n_samples
        self.kernel_width = kernel_width
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in text.split() if token]

    def _compute_features(
        self,
        text: str,
        predict_fn: Callable[[List[str]], np.ndarray],
    ) -> tuple[List[tuple[str, float]], float]:
        candidates, base_prob = _build_contextual_ablation_features(
            text,
            predict_fn,
            max_features=self.n_features,
        )
        features = [(item["phrase"], float(item["signed_score"])) for item in candidates]
        features.sort(key=lambda item: abs(item[1]), reverse=True)
        return features, base_prob

    def explain_instance(
        self,
        text: str,
        predict_fn: Callable,
        instance_idx: int,
        true_label: Optional[int] = None,
    ) -> Dict[str, Any]:
        logger.info(
            "  Leave-one-out explaining instance %s: '%s'",
            instance_idx,
            f"{text[:80]}..." if len(text) > 80 else text,
        )

        probs = np.asarray(predict_fn([text]), dtype=float)[0]
        pred_cls = int(np.argmax(probs))
        conf = float(probs[pred_cls])
        pred_lbl = self.class_names[pred_cls]
        true_lbl = self.class_names[true_label] if true_label is not None else "unknown"

        logger.info(
            "    Prediction: %s (confidence=%.3f), True label: %s",
            pred_lbl,
            conf,
            true_lbl,
        )

        features, base_prob = self._compute_features(text, predict_fn)
        for word, weight in features:
            direction = "PHISHING UP" if weight > 0 else "LEGITIMATE UP"
            logger.info(
                "    Feature '%s': %s  (delta_p_phishing=%+.4f)",
                word,
                direction,
                weight,
            )

        label_tag = "phish" if pred_cls == 1 else "legit"
        plot_path = self.output_dir / f"lime_instance_{instance_idx:04d}_{label_tag}.png"
        self._save_instance_plot(features, text, pred_lbl, conf, true_lbl, base_prob, plot_path)

        return {
            "text": text,
            "instance_idx": instance_idx,
            "prediction": pred_cls,
            "confidence": conf,
            "true_label": true_label,
            "features": features,
            "base_phishing_probability": base_prob,
            "method": "leave_one_out",
        }

    def explain_batch(
        self,
        texts: List[str],
        predict_fn: Callable,
        labels: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        logger.info(
            "Starting word-ablation explanation of %s instances (top_features=%s)...",
            len(texts),
            self.n_features,
        )

        explanations = []
        for index, text in enumerate(texts):
            true_lbl = labels[index] if labels else None
            explanations.append(self.explain_instance(text, predict_fn, index, true_lbl))

        self._save_aggregate_plot(explanations)
        self._save_explanations_json(explanations)

        logger.info("Word-ablation explanations complete. Artefacts saved to: %s", self.output_dir)
        return explanations

    def _save_instance_plot(
        self,
        features: List[tuple[str, float]],
        text: str,
        pred_lbl: str,
        conf: float,
        true_lbl: str,
        base_prob: float,
        path: Path,
    ) -> None:
        if not features:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        words = [item[0] for item in features]
        weights = [item[1] for item in features]
        colors = ["#e74c3c" if weight > 0 else "#2ecc71" for weight in weights]

        fig, ax = plt.subplots(figsize=(9, max(4, len(words) * 0.45)))
        ax.barh(range(len(words)), weights, color=colors, edgecolor="white")
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Delta P(phishing) after removing the token", fontsize=10)
        ax.set_title(
            f"Token Ablation Explanation\n"
            f'Subject: "{text[:70]}{"..." if len(text) > 70 else ""}"\n'
            f"Prediction: {pred_lbl} ({conf:.1%}) | True: {true_lbl} | Base P(phishing): {base_prob:.1%}",
            fontsize=10,
            pad=10,
        )
        plt.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _save_aggregate_plot(self, explanations: List[Dict[str, Any]]) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from collections import defaultdict

        word_weights: Dict[str, List[float]] = defaultdict(list)
        for explanation in explanations:
            for word, weight in explanation.get("features", []):
                word_weights[word.lower()].append(weight)

        if not word_weights:
            return

        word_mean = {word: float(np.mean(weights)) for word, weights in word_weights.items()}
        word_std = {word: float(np.std(weights)) for word, weights in word_weights.items()}

        top_n = min(20, len(word_mean))
        sorted_words = sorted(word_mean, key=lambda word: abs(word_mean[word]), reverse=True)[:top_n]
        sorted_words = sorted(sorted_words, key=lambda word: word_mean[word])

        means = [word_mean[word] for word in sorted_words]
        stds = [word_std[word] for word in sorted_words]
        colors = ["#e74c3c" if mean > 0 else "#2ecc71" for mean in means]

        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.45)))
        ax.barh(range(len(sorted_words)), means, xerr=stds, color=colors, edgecolor="white", capsize=3)
        ax.set_yticks(range(len(sorted_words)))
        ax.set_yticklabels(sorted_words, fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(
            "Mean delta P(phishing) after removing the token\n"
            "(red = token raises phishing risk, green = token lowers it)",
            fontsize=10,
        )
        ax.set_title(
            f"Aggregate Token Ablation Importance\n"
            f"(Top {top_n} tokens across {len(explanations)} explained instances)",
            fontsize=11,
            pad=10,
        )
        plt.tight_layout()
        path = self.output_dir / "lime_aggregate_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Aggregate explanation plot saved: %s", path.name)

    def _save_explanations_json(self, explanations: List[Dict[str, Any]]) -> None:
        clean = []
        for explanation in explanations:
            serialized: Dict[str, Any] = {}
            for key, value in explanation.items():
                if isinstance(value, list):
                    serialized[key] = [
                        [item[0], item[1].item() if hasattr(item[1], "item") else item[1]]
                        if isinstance(item, tuple) else item
                        for item in value
                    ]
                else:
                    serialized[key] = value.item() if hasattr(value, "item") else value
            clean.append(serialized)

        path = self.output_dir / "lime_explanations.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(clean, fh, indent=2, ensure_ascii=False)
        logger.info("Raw explanation data saved: %s", path.name)
