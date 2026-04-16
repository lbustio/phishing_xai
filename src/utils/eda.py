from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("phishing_xai.eda")

_STYLE = "seaborn-v0_8-whitegrid"
_DPI = 300
_CLASS_LABELS = {0: "Legitimate", 1: "Phishing"}
_PALETTE = {0: "#2196F3", 1: "#F44336"}
_HATCH = {0: "///", 1: "xxx"}  # distinguishable in grayscale/print


def _lengths_by_class(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    lengths = df["subject"].str.len()
    legit = lengths[df["label"] == 0]
    phish = lengths[df["label"] == 1]
    return lengths, legit, phish


def _export(fig: plt.Figure, base: Path, csv_df: pd.DataFrame) -> None:
    """Save PNG, EPS and CSV with the same base name."""
    fig.savefig(base.with_suffix(".png"), dpi=_DPI, bbox_inches="tight")
    fig.savefig(base.with_suffix(".eps"), format="eps", bbox_inches="tight")
    plt.close(fig)
    csv_df.to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8")
    logger.info(
        "[EDA] Archivos exportados: %s (.png / .eps / .csv)",
        base.name,
    )


def _histogram_kde(legit: pd.Series, phish: pd.Series, base: Path) -> None:
    bins = np.linspace(0, max(legit.max(), phish.max()), 61)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    leg_counts, _ = np.histogram(legit, bins=bins)
    phi_counts, _ = np.histogram(phish, bins=bins)
    leg_density = leg_counts / (leg_counts.sum() * np.diff(bins))
    phi_density = phi_counts / (phi_counts.sum() * np.diff(bins))

    csv_df = pd.DataFrame({
        "bin_start": bins[:-1].round(2),
        "bin_end": bins[1:].round(2),
        "count_legitimate": leg_counts,
        "count_phishing": phi_counts,
        "density_legitimate": leg_density.round(6),
        "density_phishing": phi_density.round(6),
    })

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for label_id, series, density in [
            (0, legit, leg_density),
            (1, phish, phi_density),
        ]:
            ax.bar(
                bin_centers,
                density,
                width=np.diff(bins),
                color=_PALETTE[label_id],
                hatch=_HATCH[label_id],
                edgecolor="white",
                linewidth=0.3,
                label=_CLASS_LABELS[label_id],
            )
            series.plot.kde(ax=ax, color=_PALETTE[label_id], linewidth=2)
        ax.set_xlabel("Subject Length (characters)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Subject Length Distribution by Class", fontsize=13, fontweight="bold")
        ax.legend(title="Class", fontsize=11)
        ax.set_xlim(left=0)

    _export(fig, base, csv_df)


def _histogram_kde_log(legit: pd.Series, phish: pd.Series, base: Path) -> None:
    min_val = max(1, min(legit.min(), phish.min()))
    max_val = max(legit.max(), phish.max())
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 61)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    leg_counts, _ = np.histogram(legit, bins=bins)
    phi_counts, _ = np.histogram(phish, bins=bins)
    leg_density = leg_counts / (leg_counts.sum() * np.diff(bins))
    phi_density = phi_counts / (phi_counts.sum() * np.diff(bins))

    csv_df = pd.DataFrame({
        "bin_start": bins[:-1].round(4),
        "bin_end": bins[1:].round(4),
        "count_legitimate": leg_counts,
        "count_phishing": phi_counts,
        "density_legitimate": leg_density.round(8),
        "density_phishing": phi_density.round(8),
    })

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for label_id, counts, density in [
            (0, leg_counts, leg_density),
            (1, phi_counts, phi_density),
        ]:
            ax.bar(
                bin_centers,
                density,
                width=np.diff(bins),
                color=_PALETTE[label_id],
                hatch=_HATCH[label_id],
                edgecolor="white",
                linewidth=0.3,
                label=_CLASS_LABELS[label_id],
            )
        ax.set_xscale("log")
        ax.set_xlabel("Subject Length (characters, log scale)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Subject Length Distribution by Class (Log Scale)", fontsize=13, fontweight="bold")
        ax.legend(title="Class", fontsize=11)

    _export(fig, base, csv_df)


def _boxplot(legit: pd.Series, phish: pd.Series, base: Path) -> None:
    def _five_num(s: pd.Series, label: str) -> dict:
        return {
            "class": label,
            "min": int(s.min()),
            "q1": float(s.quantile(0.25)),
            "median": float(s.median()),
            "q3": float(s.quantile(0.75)),
            "max": int(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
        }

    csv_df = pd.DataFrame([
        _five_num(legit, "Legitimate"),
        _five_num(phish, "Phishing"),
    ])

    plot_df = pd.DataFrame({
        "Length": pd.concat([legit, phish]),
        "Class": (["Legitimate"] * len(legit)) + (["Phishing"] * len(phish)),
    })

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.boxplot(
            data=plot_df,
            x="Class",
            y="Length",
            palette={"Legitimate": _PALETTE[0], "Phishing": _PALETTE[1]},
            width=0.5,
            flierprops={"marker": "o", "markersize": 2},
            ax=ax,
        )
        ax.set_xlabel("Class", fontsize=12)
        ax.set_ylabel("Subject Length (characters)", fontsize=12)
        ax.set_title("Subject Length Boxplot by Class", fontsize=13, fontweight="bold")

    _export(fig, base, csv_df)


def _cdf(legit: pd.Series, phish: pd.Series, base: Path) -> None:
    # Build a common length grid for the CSV
    max_len = max(legit.max(), phish.max())
    grid = np.arange(0, max_len + 1)

    def _cdf_at_grid(series: pd.Series) -> np.ndarray:
        sorted_vals = np.sort(series)
        cdf_vals = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        return np.interp(grid, sorted_vals, cdf_vals, left=0.0, right=1.0)

    leg_cdf = _cdf_at_grid(legit)
    phi_cdf = _cdf_at_grid(phish)

    # Downsample CSV to max 2000 rows to keep file size reasonable
    step = max(1, len(grid) // 2000)
    csv_df = pd.DataFrame({
        "length_chars": grid[::step],
        "cdf_legitimate": leg_cdf[::step].round(6),
        "cdf_phishing": phi_cdf[::step].round(6),
    })

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(grid, leg_cdf, color=_PALETTE[0], linewidth=2, label=_CLASS_LABELS[0])
        ax.plot(grid, phi_cdf, color=_PALETTE[1], linewidth=2, label=_CLASS_LABELS[1])
        ax.axvline(x=100, color="gray", linestyle="--", linewidth=1, label="100 chars")
        ax.set_xlabel("Subject Length (characters)", fontsize=12)
        ax.set_ylabel("Cumulative Proportion", fontsize=12)
        ax.set_title("Cumulative Distribution Function (CDF) by Class", fontsize=13, fontweight="bold")
        ax.legend(title="Class", fontsize=11)
        ax.set_xlim(left=0)

    _export(fig, base, csv_df)


def _descriptive_stats(legit: pd.Series, phish: pd.Series, base: Path) -> None:
    stats_def = [
        ("Mean", legit.mean(), phish.mean()),
        ("Median", legit.median(), phish.median()),
        ("Std. Dev.", legit.std(), phish.std()),
        ("p75", legit.quantile(0.75), phish.quantile(0.75)),
        ("p95", legit.quantile(0.95), phish.quantile(0.95)),
    ]
    labels = [s[0] for s in stats_def]
    leg_vals = [s[1] for s in stats_def]
    phi_vals = [s[2] for s in stats_def]

    csv_df = pd.DataFrame({
        "statistic": labels,
        "legitimate": [round(v, 2) for v in leg_vals],
        "phishing": [round(v, 2) for v in phi_vals],
    })

    x = np.arange(len(labels))
    width = 0.35

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        bars_leg = ax.bar(x - width / 2, leg_vals, width, label="Legitimate",
                          color=_PALETTE[0], hatch=_HATCH[0], edgecolor="white")
        bars_phi = ax.bar(x + width / 2, phi_vals, width, label="Phishing",
                          color=_PALETTE[1], hatch=_HATCH[1], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Characters", fontsize=12)
        ax.set_title("Descriptive Statistics of Subject Length by Class", fontsize=13, fontweight="bold")
        ax.legend(title="Class", fontsize=11)
        for bar in list(bars_leg) + list(bars_phi):
            ax.annotate(
                f"{bar.get_height():.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=9,
            )

    _export(fig, base, csv_df)


def plot_subject_length_eda(df: pd.DataFrame, out_dir: Path) -> None:
    """Genera todos los gráficos EDA de longitud de asunto en PNG, EPS y CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lengths, legit, phish = _lengths_by_class(df)

    _histogram_kde(legit, phish,        out_dir / "subject_length_distribution")
    _histogram_kde_log(legit, phish,    out_dir / "subject_length_distribution_logscale")
    _boxplot(legit, phish,              out_dir / "subject_length_boxplot")
    _cdf(legit, phish,                  out_dir / "subject_length_cdf")
    _descriptive_stats(legit, phish,    out_dir / "subject_length_descriptive_stats")

    logger.info("[EDA] Todos los gráficos exportados en '%s' (PNG + EPS + CSV).", out_dir)
