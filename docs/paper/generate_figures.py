#!/usr/bin/env python3
"""Generate publication-grade SVG and PNG figures for DeepMzyme from verified benchmark results."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.2,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_PMM_D = "#d95f02"      # Burnt orange (PMM-D deployment)
COLOR_PMM_R = "#e6ab02"      # Gold/amber (PMM-R leaked table)
COLOR_PMM_PUB = "#7570b3"    # Muted purple (PMM published)
COLOR_DMZ_SINGLE = "#2b8cbe" # Medium blue (DeepMzyme single/5-fold)
COLOR_DMZ_ENS = "#08589e"    # Deep navy (DeepMzyme ensembles)
COLOR_GVP = "#66a61e"        # Green (Only-GVP)
COLOR_ESM = "#1b9e77"        # Teal (Only-ESM-C)


def save_both(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.svg")
    fig.savefig(OUT_DIR / f"{stem}.png")
    plt.close(fig)


def make_fig1_benchmark_overview() -> None:
    """2-panel figure: (A) 5-Fold Grouped CV (BA & Macro-Precision), (B) Reference Test (Full 1,488 & In-Scope 1,346)."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.0, 5.8), gridspec_kw={"width_ratios": [1.0, 1.08], "wspace": 0.58}
    )

    # --- Panel A: 5-Fold Cross-Validation ---
    cv_labels = [
        "PMM-D\n(Deployment, 89 feat.)",
        "PMM-R\n(Leaked valence_3a, 93 feat.)",
        "PinMyMetal Published*\n(Ungrouped CV + Leak)",
        "DeepMzyme Baseline\n(5-Fold B4, s42)",
        "DeepMzyme Trunk-LR\n(5-Fold lr=1e-4, s42)",
        "DeepMzyme P 5-Class\n(15-Ckpt, 3 Seeds)",
        "DeepMzyme Greedy CV\n(30-Ckpt, R3)",
    ]
    cv_ba = [62.96, 67.16, 72.70, 74.34, 75.51, 75.82, 77.99]
    cv_mp = [68.47, 71.24, 75.08, 75.74, 77.12, 77.13, 81.14]

    y = np.arange(len(cv_labels))
    h = 0.36
    bars_ba = ax1.barh(
        y - h / 2, cv_ba, height=h, color="#08589e", label="Balanced Accuracy (Mean Recall %)", edgecolor="white"
    )
    bars_mp = ax1.barh(
        y + h / 2, cv_mp, height=h, color="#6baed6", label="Macro-Precision (PMM Fig. 2a Metric %)", edgecolor="white"
    )

    ax1.axvline(
        75.08, color=COLOR_PMM_PUB, linestyle="--", linewidth=1.2, alpha=0.85, label="PMM Published Fig. 2a (75.08% MP)"
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels(cv_labels)
    ax1.invert_yaxis()
    ax1.set_xlim(50, 86.5)
    ax1.set_xlabel("5-Fold Cross-Validation Score (%)")
    ax1.set_title("A. 5-Fold Cross-Validation (7,398 Sites, ≥90% Seq-ID)", pad=10)
    ax1.grid(axis="x", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.48, -0.15), ncol=1, frameon=True, facecolor="white")

    for b in bars_ba:
        w = b.get_width()
        ax1.text(
            w + 0.4, b.get_y() + b.get_height() / 2, f"{w:.1f}%",
            va="center", ha="left", fontsize=8.5, fontweight="bold", color="#08306b",
        )
    for b in bars_mp:
        w = b.get_width()
        ax1.text(
            w + 0.4, b.get_y() + b.get_height() / 2, f"{w:.1f}%",
            va="center", ha="left", fontsize=8.2, color="#2171b5",
        )

    # --- Panel B: Reference Test Set ---
    test_labels = [
        "PMM-D (Deployment, 89 feat.)",
        "PMM-R (Leaked valence_3a, 93 feat.)",
        "PinMyMetal Published (Fig. 2b)*",
        "DeepMzyme Single Refit (1 Ckpt)",
        "DeepMzyme Baseline (5 Ckpts, s42)",
        "DeepMzyme Trunk-LR (5 Ckpts, lr=1e-4)",
        "DeepMzyme Unified (5 Ckpts, P5+lr1e4)",
        "DeepMzyme P 5-Class (15 Ckpts, R1)",
        "DeepMzyme CV Top-5 (25 Ckpts, R2b)",
    ]
    full_test = [58.27, 63.41, 67.85, 67.55, 69.02, 70.75, 71.72, 71.10, 72.10]
    elig_test = [59.48, 63.81, np.nan, 69.33, 70.75, 72.25, 73.08, 72.68, 73.64]

    y2 = np.arange(len(test_labels))
    bars_full = ax2.barh(
        y2 - h / 2, full_test, height=h, color="#08589e", label="Full Reference Test (1,488 sites)", edgecolor="white"
    )
    bars_elig = ax2.barh(
        y2 + h / 2, [0 if np.isnan(v) else v for v in elig_test],
        height=h, color="#4eb3d3", label="In-Scope Protein Sites (1,346 sites)", edgecolor="white",
    )

    ax2.axvline(
        67.85, color=COLOR_PMM_PUB, linestyle="--", linewidth=1.2, alpha=0.85, label="PMM Published Fig. 2b (67.85% Full)"
    )
    ax2.set_yticks(y2)
    ax2.set_yticklabels(test_labels)
    ax2.invert_yaxis()
    ax2.set_xlim(50, 79.5)
    ax2.set_xlabel("Test Balanced Accuracy (%)")
    ax2.set_title("B. Held-Out Reference Test Set (Full 1,488 & In-Scope 1,346)", pad=10)
    ax2.grid(axis="x", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.48, -0.15), ncol=1, frameon=True, facecolor="white")

    for b in bars_full:
        w = b.get_width()
        ax2.text(
            w + 0.35, b.get_y() + b.get_height() / 2, f"{w:.2f}%",
            va="center", ha="left", fontsize=8.3, fontweight="bold", color="#08306b",
        )
    for b, val in zip(bars_elig, elig_test):
        if not np.isnan(val):
            ax2.text(
                val + 0.35, b.get_y() + b.get_height() / 2, f"{val:.2f}%",
                va="center", ha="left", fontsize=8.0, color="#1d6f8a",
            )

    fig.suptitle(
        "DeepMzyme Outperforms PinMyMetal Across Matched 5-Fold CV and Reference Test Benchmarks",
        fontsize=13, fontweight="bold", y=0.99,
    )
    save_both(fig, "fig1_benchmark_overview")


def make_fig2_per_metal_recall_audit() -> None:
    """Grouped bar chart of per-metal test recalls (Mn, Cu, Zn, Class VIII) highlighting the TECH-028 Mn drop."""
    fig, ax = plt.subplots(figsize=(10.0, 5.2))

    metals = ["Manganese (Mn)", "Copper (Cu)", "Zinc (Zn)", "Class VIII (Fe/Co/Ni)"]
    x = np.arange(len(metals))
    w = 0.19

    pmm_r = [79.0, 40.6, 85.2, 48.8]
    pmm_d = [61.7, 50.0, 60.7, 60.7]
    dmz_1 = [55.7, 70.3, 85.1, 59.1]
    dmz_ens = [65.3, 73.4, 89.0, 60.7]

    b1 = ax.bar(x - 1.5 * w, pmm_r, width=w, color=COLOR_PMM_R, label="PMM-R (With Leaked valence_3a: 63.4% BA)", edgecolor="white")
    b2 = ax.bar(x - 0.5 * w, pmm_d, width=w, color=COLOR_PMM_D, label="PMM-D (Deployment Features: 58.3% BA)", edgecolor="white")
    b3 = ax.bar(x + 0.5 * w, dmz_1, width=w, color=COLOR_DMZ_SINGLE, label="DeepMzyme Single Model (67.6% BA)", edgecolor="white")
    b4 = ax.bar(x + 1.5 * w, dmz_ens, width=w, color=COLOR_DMZ_ENS, label="DeepMzyme CV Ensemble (R2b Top-5: 72.1% BA)", edgecolor="white")

    for bars in (b1, b2, b3, b4):
        for bar in bars:
            h_val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h_val + 1.0, f"{h_val:.1f}%",
                ha="center", va="bottom", fontsize=8.3, fontweight="bold",
            )

    ax.annotate(
        "Removing leaked valence_3a\ndrops PMM Mn recall by -17.4 pp\n(79.0% → 61.7%)",
        xy=(0 - 0.5 * w, 65.5),
        xytext=(0.48, 84.0),
        arrowprops={"arrowstyle": "->", "color": "#b30000", "lw": 1.5},
        fontsize=8.8,
        fontweight="bold",
        color="#990000",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff5f0", "edgecolor": "#cb181d", "alpha": 0.95},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metals, fontweight="bold")
    ax.set_ylim(0, 114)
    ax.set_ylabel("Held-Out Test Recall (%) — Full 1,488 Sites")
    ax.set_title("Per-Metal Recall & Impact of the Upstream Bond-Valence Leak (valence_3a)", pad=12)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True, facecolor="white")

    fig.tight_layout()
    save_both(fig, "fig2_per_metal_recall_audit")


def make_fig3_modality_optimization_ablations() -> None:
    """Horizontal bar progression showing unimodal -> L2 tuned -> multimodal baseline -> differential trunk LR -> CV ensembles."""
    fig, ax = plt.subplots(figsize=(10.4, 4.5))

    variants = [
        "Only-GVP (3D Structure Only, wd=1e-4)",
        "Only-ESM-C 600M (Sequence Only, wd=1e-4)",
        "Only-GVP (3D Structure Only, L2 wd=1.0)",
        "DeepMzyme Late Fusion (Baseline lr=3e-4)",
        "DeepMzyme Late Fusion (Differential Trunk lr=1e-4)",
        "DeepMzyme 15-Model P 5-Class Ensemble (3 Seeds)",
        "DeepMzyme Multi-Family CV Ensembles (R1–R3)",
    ]
    scores = [69.66, 72.33, 73.25, 74.28, 75.45, 75.82, 76.61]
    colors = [COLOR_GVP, COLOR_ESM, "#80b1d3", COLOR_DMZ_SINGLE, "#1d91c0", "#225ea8", COLOR_DMZ_ENS]
    annotations = [
        "Unregularized 3D GNN baseline",
        "Pretrained 600M PLM baseline",
        "+3.59 pp from heavy L2 weight decay",
        "+1.95 pp vs. ESM-C (5/5 folds) / +4.62 pp vs. GVP",
        "+3.12 pp vs. ESM-C / +2.20 pp vs. GVP (wd=1.0)",
        "Pooled OOF BA (+1.48 pp over 5-fold baseline)",
        "Nested CV up to 76.61% (77.99% pooled OOF)",
    ]

    y = np.arange(len(variants))
    bars = ax.barh(y, scores, height=0.58, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(variants)
    ax.invert_yaxis()
    ax.set_xlim(65, 92.5)
    ax.set_xlabel("5-Fold Grouped Cross-Validation Balanced Accuracy (%)")
    ax.set_title("Controlled Modality, Regularization & Optimization Ablations (7,398 Sites)", pad=10)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    for bar, note in zip(bars, annotations):
        w = bar.get_width()
        ax.text(
            w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%  —  {note}",
            va="center", ha="left", fontsize=8.6, fontweight="bold", color="#1a1a1a",
        )

    fig.tight_layout()
    save_both(fig, "fig3_modality_optimization_ablations")


def make_fig4_class_viii_resolution() -> None:
    """Summary of native 5-class and 6-class resolution inside Class VIII (Fe, Co, Ni)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"wspace": 0.28})

    # Panel A: Native 5-class recalls (Mn, Cu, Zn, Fe, Co+Ni)
    classes_5 = ["Mn\n(n=2,528)", "Zn\n(n=2,032)", "Cu\n(n=343)", "Fe (Resolved)\n(n=1,737)", "Co+Ni\n(n=758)"]
    recalls_5 = [75.8, 80.4, 72.0, 60.6, 27.4]
    c5 = ["#377eb8", "#4daf4a", "#984ea3", "#e41a1c", "#ff7f00"]
    b5 = ax1.bar(classes_5, recalls_5, width=0.55, color=c5, edgecolor="white")
    ax1.axhline(65.2, color="#222222", linestyle="--", linewidth=1.1, label="Native 5-Class Macro BA (65.2%)")
    ax1.set_ylim(0, 95)
    ax1.set_ylabel("5-Fold Grouped CV Recall (%)")
    ax1.set_title("A. Native 5-Class Resolution (Fe vs. Co+Ni)", pad=10)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)
    for b in b5:
        v = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=8.8)

    # Panel B: Native 6-class Class VIII breakdown & Pairwise Co-Ni AUROC
    metrics_6 = ["Fe Recall\n(n=1,737)", "Ni Recall\n(n=475)", "Co Recall\n(n=283, 3.8%)", "Pairwise Co–Ni\nAUROC (×100)"]
    vals_6 = [61.3, 29.5, 4.9, 68.0]
    c6 = ["#e41a1c", "#ff7f00", "#fdb462", "#08589e"]
    b6 = ax2.bar(metrics_6, vals_6, width=0.52, color=c6, edgecolor="white")
    ax2.set_ylim(0, 95)
    ax2.set_ylabel("Score (%) / AUROC × 100")
    ax2.set_title("B. Native 6-Class Resolution (Fe / Co / Ni)", pad=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    for idx, b in enumerate(b6):
        v = b.get_height()
        label = "0.680 AUROC\n[0.582, 0.770]" if idx == 3 else f"{v:.1f}%"
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, label, ha="center", va="bottom", fontweight="bold", fontsize=8.5)

    save_both(fig, "fig4_class_viii_resolution")


if __name__ == "__main__":
    make_fig1_benchmark_overview()
    make_fig2_per_metal_recall_audit()
    make_fig3_modality_optimization_ablations()
    make_fig4_class_viii_resolution()
    print(f"Generated 4 SVG/PNG figure pairs in {OUT_DIR}")
