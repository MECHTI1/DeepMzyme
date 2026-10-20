#!/usr/bin/env python3
"""Generate publication-grade figures (SVG + 300-DPI PNG) for DeepMzyme.

All values and confidence intervals are strictly taken from the verified
matched comparison (`docs/campaigns/phase2_unified_2026-10/pmm_matched/README.md`),
the release audit (`docs/campaigns/phase2_unified_2026-10/release_audit/audit_summary.json`),
and the Phase 2 unified summary (`docs/campaigns/phase2_unified_2026-10/phase2_unified_summary.json`).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.8,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.8,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "legend.fontsize": 8.2,
    "figure.titlesize": 12.2,
    "figure.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
})


def save_both(fig: plt.Figure, stem: str) -> None:
    svg_path = OUT_DIR / f"{stem}.svg"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig1_benchmark_overview() -> None:
    """Two-panel horizontal comparison with identical pre-declared model rows in CV (A) and Test (B)."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14.4, 6.3), sharey=True, gridspec_kw={"wspace": 0.14}
    )

    models = [
        "PMM-D (Deployment, 89 feat.)",
        "PMM-R (With valence_3a leak, 93 feat.)",
        "PMM-H (HistGB Ablation, 89 feat.)",
        "PinMyMetal Published* (Ungrouped + Leak)",
        "DeepMzyme Single Refit (1 Ckpt, a0007846)",
        "DeepMzyme Baseline B4 (5 Ckpts, R4 s42)",
        "DeepMzyme Trunk-LR 1e-4 (5 Ckpts, s42)",
        "DeepMzyme R1 Top-1 (P 5-Class, 15 Ckpts)",
        "DeepMzyme R1 Top-4 All Seeds (50 Ckpts)",
    ]
    # Panel A: 5-Fold CV Pooled BA (%) and Macro-Precision (%)
    cv_ba = [62.96, 67.16, 71.22, 72.70, 74.34, 74.34, 75.51, 75.82, 76.46]
    cv_mp = [68.47, 71.24, 75.51, 75.08, 75.74, 75.74, 77.12, 77.13, 78.36]

    # Panel B: Full Test (1,488) BA + 95% CI, and In-Scope Eligible (1,346) BA
    test_full = [58.27, 63.41, 68.83, 67.85, 67.55, 69.02, 70.75, 71.10, 71.75]
    test_ci_lo = [53.53, 59.64, 64.26, np.nan, 62.93, 64.47, 66.29, 66.65, 67.22]
    test_ci_hi = [62.85, 67.25, 73.04, np.nan, 71.65, 73.05, 74.77, 75.14, 75.80]
    test_elig = [59.48, 63.81, 69.49, np.nan, 69.33, 70.75, 72.25, 72.68, 73.48]

    y = np.arange(len(models))
    h = 0.36

    # Panel A bars
    b1_ba = ax1.barh(y - h / 2, cv_ba, height=h, color="#0b559f", label="5-Fold CV Balanced Accuracy (%)")
    b1_mp = ax1.barh(y + h / 2, cv_mp, height=h, color="#6baed6", label="5-Fold CV Macro-Precision (PMM Fig. 2a %)")
    ax1.axvline(62.96, color="#d95f02", linestyle=":", linewidth=1.2, alpha=0.9, label="Matched PMM-D CV BA (62.96%)")
    ax1.axvline(75.08, color="#756bb1", linestyle="--", linewidth=1.1, alpha=0.85, label="PMM Published Fig. 2a MP (75.08%)")
    ax1.axhline(3.5, color="#888888", linestyle="-", linewidth=0.8, alpha=0.5)

    ax1.set_yticks(y)
    ax1.set_yticklabels(models, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlim(50, 84.5)
    ax1.set_xlabel("5-Fold Cross-Validation Score (%)")
    ax1.set_title("A. 5-Fold Grouped CV (7,398 Sites, ≥90% Seq-ID)", pad=10)
    ax1.grid(axis="x", linestyle=":", alpha=0.5)

    for bar in b1_ba:
        w = bar.get_width()
        ax1.text(w + 0.35, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", ha="left", fontsize=8.1, fontweight="bold", color="#08306b")
    for bar in b1_mp:
        w = bar.get_width()
        ax1.text(w + 0.35, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", ha="left", fontsize=7.9, color="#2171b5")

    # Panel B bars + 95% CI whiskers on Full Test
    xerr_lo = [v - lo if not np.isnan(lo) else 0.0 for v, lo in zip(test_full, test_ci_lo)]
    xerr_hi = [hi - v if not np.isnan(hi) else 0.0 for v, hi in zip(test_full, test_ci_hi)]
    b2_full = ax2.barh(
        y - h / 2, test_full, height=h, color="#0b559f",
        xerr=[xerr_lo, xerr_hi], error_kw={"ecolor": "#1a1a1a", "elinewidth": 1.0, "capsize": 2.5, "alpha": 0.75},
        label="Full Test (1,488 sites, ±95% CI)",
    )
    elig_vals = [v if not np.isnan(v) else 0.0 for v in test_elig]
    b2_elig = ax2.barh(y + h / 2, elig_vals, height=h, color="#41b6c4", label="In-Scope Protein Test (1,346 sites)")
    ax2.axvline(58.27, color="#d95f02", linestyle=":", linewidth=1.2, alpha=0.9, label="Matched PMM-D Full Test (58.27%)")
    ax2.axvline(67.85, color="#756bb1", linestyle="--", linewidth=1.1, alpha=0.85, label="PMM Published Fig. 2b Test (67.85%)")
    ax2.axhline(3.5, color="#888888", linestyle="-", linewidth=0.8, alpha=0.5)

    ax2.set_xlim(48, 82.5)
    ax2.set_xlabel("Held-Out Test Balanced Accuracy (%)")
    ax2.set_title("B. Held-Out Reference Test Set (Full 1,488 & In-Scope 1,346)", pad=10)
    ax2.grid(axis="x", linestyle=":", alpha=0.5)

    for idx, bar in enumerate(b2_full):
        w = bar.get_width()
        hi = test_ci_hi[idx]
        text_x = (hi if not np.isnan(hi) else w) + 0.35
        ax2.text(text_x, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", ha="left", fontsize=8.1, fontweight="bold", color="#08306b")
    for idx, bar in enumerate(b2_elig):
        if np.isnan(test_elig[idx]):
            continue
        w = bar.get_width()
        ax2.text(w + 0.35, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", ha="left", fontsize=7.9, color="#0c7c8a")

    ax1.legend(loc="lower center", bbox_to_anchor=(0.48, -0.24), ncol=2, frameon=True, fontsize=7.9)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.52, -0.24), ncol=2, frameon=True, fontsize=7.9)
    fig.suptitle("DeepMzyme vs. PinMyMetal: Pre-Registered Matched Comparison Across 5-Fold CV and Reference Test", y=0.98)
    save_both(fig, "fig1_benchmark_overview")


def make_fig2_per_metal_recall_audit() -> None:
    """Two-panel per-metal recall comparison across 5-Fold CV (A) and Held-Out Test (B) using strict CV models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.2), sharey=True, gridspec_kw={"wspace": 0.10})

    metals = ["Manganese\n(Mn)", "Copper\n(Cu)", "Zinc\n(Zn)", "Class VIII\n(Fe/Co/Ni)"]
    x = np.arange(len(metals))
    w = 0.20

    # Panel A: 5-Fold Grouped CV (7,398 sites) - exact verified recalls from audit_summary.json & PMM refits
    cv_pmm_r = [81.6, 42.6, 82.0, 62.5]
    cv_pmm_d = [77.2, 42.3, 66.2, 66.1]
    cv_dmz_b4 = [75.6, 73.5, 81.4, 66.9]
    cv_dmz_r1 = [78.6, 74.6, 84.4, 65.7]

    # Panel B: Held-Out Full Test (1,488 sites) - exact verified recalls
    test_pmm_r = [79.0, 40.6, 85.2, 48.8]
    test_pmm_d = [61.7, 50.0, 60.7, 60.7]
    test_dmz_1 = [55.7, 70.3, 85.1, 59.1]
    test_dmz_r1 = [62.3, 71.9, 89.2, 61.1]

    colors = ["#e6ab02", "#d95f02", "#2b8cbe", "#08589e"]
    labels_cv = [
        "PMM-R (With Leaked valence_3a: 67.2% BA)",
        "PMM-D (Deployment Features: 63.0% BA)",
        "DeepMzyme Baseline B4 (5 Ckpts: 74.3% BA)",
        "DeepMzyme R1 Top-1 (P 5-Class, 15 Ckpts: 75.8% BA)",
    ]
    labels_test = [
        "PMM-R (With Leaked valence_3a: 63.4% BA)",
        "PMM-D (Deployment Features: 58.3% BA)",
        "DeepMzyme Single Refit (1 Ckpt: 67.6% BA)",
        "DeepMzyme R1 Top-1 (P 5-Class, 15 Ckpts: 71.1% BA)",
    ]

    for idx, (series, col, lab) in enumerate(zip([cv_pmm_r, cv_pmm_d, cv_dmz_b4, cv_dmz_r1], colors, labels_cv)):
        offset = (idx - 1.5) * w
        bars = ax1.bar(x + offset, series, width=w * 0.90, color=col, edgecolor="white", label=lab)
        for b in bars:
            v = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, v + 1.1, f"{v:.1f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(metals, fontweight="bold")
    ax1.set_ylim(0, 114)
    ax1.set_ylabel("Per-Class Recall (%)")
    ax1.set_title("A. 5-Fold Grouped Cross-Validation (7,398 Sites)", pad=10)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=True)

    for idx, (series, col, lab) in enumerate(zip([test_pmm_r, test_pmm_d, test_dmz_1, test_dmz_r1], colors, labels_test)):
        offset = (idx - 1.5) * w
        bars = ax2.bar(x + offset, series, width=w * 0.90, color=col, edgecolor="white", label=lab)
        for b in bars:
            v = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2, v + 1.1, f"{v:.1f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")

    ax2.annotate(
        "Removing leaked valence_3a\ndrops PMM Mn recall by -17.4 pp\n(79.0% → 61.7%, 95% CI [11.1, 24.1])",
        xy=(0 - 0.5 * w, 65.5),
        xytext=(0.12, 95.5),
        arrowprops={"arrowstyle": "->", "color": "#b30000", "lw": 1.6},
        fontsize=8.1,
        fontweight="bold",
        color="#990000",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff5f0", "edgecolor": "#cb181d", "alpha": 0.95},
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(metals, fontweight="bold")
    ax2.set_title("B. Held-Out Reference Test Set (1,488 Sites)", pad=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=True)

    fig.suptitle("Per-Metal Recall Breakdown & Impact of the Upstream Bond-Valence Leak (TECH-028)", y=0.99)
    save_both(fig, "fig2_per_metal_recall_audit")


def make_fig3_modality_optimization_ablations() -> None:
    """Two-panel strict ablation figure: (A) Strictly matched s42 modality/ensemble/regularization (CV + Test), (B) 3-seed Late Fusion funnel."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 5.5), gridspec_kw={"wspace": 0.34, "width_ratios": [1.26, 1.0]})

    # Panel A: Strictly Matched (s42, Recipe B) Modality, Multi-Target Ensemble & Regularization Audit
    variants_a = [
        "Only-GVP 4c\n(wd=1e-4, 5 ckpts)",
        "Only-GVP 4c\n(Heavy L2 wd=1.0, 5 ckpts)",
        "Only-GVP 15-Ckpt\n(4c+5c+6c Ens, 15 ckpts)",
        "Only-ESM-C 4c\n(wd=1e-4, 5 ckpts)",
        "Only-ESM-C 6c\n(wd=1e-4, 5 ckpts)",
        "Only-ESM-C 15-Ckpt\n(4c+5c+6c Ens, 15 ckpts)",
        "Post-Hoc 0.5 ESM+0.5 GVP\n(Unimodal Avg 4c, 10 ckpts)",
        "Late Fusion 4c (B4)\n(Matched Baseline, 5 ckpts)",
        "Late Fusion 6c (B6)\n(Matched Baseline, 5 ckpts)",
        "Late Fusion 15-Ckpt\n(4c+5c+6c Ens, 15 ckpts)",
        "Late Fusion Trunk-LR*\n(GVP lr=1e-4, 5 ckpts)",
    ]
    cv_a = [69.71, 72.94, np.nan, 72.38, 72.32, np.nan, 73.79, 74.34, 74.98, 75.95, 75.51]
    test_a = [66.37, 65.02, 66.82, 69.56, 70.15, 69.99, 70.68, 70.75, 72.07, 71.37, 72.25]
    notes_a = [
        "CV 69.71% | Test 66.37%",
        "CV 72.94% | Test 65.02% (-1.35 pp!)",
        "Test 66.82% (64.70% Full)",
        "CV 72.38% | Test 69.56%",
        "CV 72.32% | Test 70.15%",
        "Test 69.99% (68.34% Full)",
        "CV 73.79% | Test 70.68% (10 ckpts)",
        "CV 74.34% | Test 70.75% (+1.19 vs ESM)",
        "CV 74.98% | Test 72.07% (+1.92 vs ESM)",
        "CV 75.95% | Test 71.37% (+1.38 vs ESM)",
        "CV 75.51% | Test 72.25% (LF-only opt*)",
    ]

    y_a = np.arange(len(variants_a))
    h_a = 0.36
    b_cv = ax1.barh(y_a - h_a / 2, [0 if np.isnan(v) else v for v in cv_a], height=h_a * 0.92, color="#2b8cbe", edgecolor="white", label="5-Fold Pooled OOF CV BA (%)")
    b_ts = ax1.barh(y_a + h_a / 2, test_a, height=h_a * 0.92, color="#41b6c4", edgecolor="white", label="In-Scope Test BA (1,346 sites, %)")
    ax1.set_yticks(y_a)
    ax1.set_yticklabels(variants_a, fontsize=7.7, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlim(60, 91.5)
    ax1.set_xlabel("Balanced Accuracy (%) — Strictly Matched Seed 42 (except *LF-only Trunk-LR)")
    ax1.set_title("A. Strictly Matched Modality, Target & Ensemble Audit (s42)", pad=10)
    ax1.grid(axis="x", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, fontsize=7.6)

    for idx, note in enumerate(notes_a):
        w = max(0 if np.isnan(cv_a[idx]) else cv_a[idx], test_a[idx])
        col = "#b30000" if idx == 1 else "#1a1a1a"
        ax1.text(w + 0.25, y_a[idx], note, va="center", ha="left", fontsize=7.3, fontweight="bold", color=col)

    # Panel B: Controlled 3-Seed (s42, 43, 44; 15 Checkpoints) Loss Recipe x Target Granularity (Within Late Fusion)
    families_b = [
        "Recipe P\n6-Class (P6)",
        "Recipe B\n4-Class (B4)",
        "Recipe P\n4-Class (P4)",
        "Recipe P\n5-Class (P5, R1)",
    ]
    mean_seed_cv = [73.31, 73.87, 74.24, 73.97]
    pooled_15_cv = [75.22, 75.58, 75.62, 75.82]
    inscope_test = [71.02, 71.83, 72.13, 72.68]
    test_lo = [66.44, 67.45, 67.68, 68.37]
    test_hi = [75.02, 75.81, 76.06, 76.56]

    x_b = np.arange(len(families_b))
    wb = 0.26
    b_ms = ax2.bar(x_b - wb, mean_seed_cv, width=wb * 0.90, color="#6baed6", label="3-Seed Mean Fold CV BA (%)")
    b_p15 = ax2.bar(x_b, pooled_15_cv, width=wb * 0.90, color="#08589e", label="15-Ckpt Pooled OOF CV BA (%)")
    yerr_lo = [v - lo for v, lo in zip(inscope_test, test_lo)]
    yerr_hi = [hi - v for v, hi in zip(inscope_test, test_hi)]
    b_tst = ax2.bar(
        x_b + wb, inscope_test, width=wb * 0.90, color="#41b6c4",
        yerr=[yerr_lo, yerr_hi], error_kw={"ecolor": "#1a1a1a", "elinewidth": 1.0, "capsize": 2.5},
        label="In-Scope Test BA (1,346 sites, ±95% CI)",
    )

    ax2.set_xticks(x_b)
    ax2.set_xticklabels(families_b, fontsize=8.4, fontweight="bold")
    ax2.set_ylim(62, 81.5)
    ax2.set_ylabel("Balanced Accuracy (%)")
    ax2.set_title("B. Within-Late-Fusion 3-Seed (s42/43/44) Recipe × Granularity", pad=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True, fontsize=7.9)

    for b in b_ms:
        v = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#2171b5")
    for b in b_p15:
        v = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#08306b")
    for idx, b in enumerate(b_tst):
        v = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2 + 0.03, test_hi[idx] + 0.35, f"{v:.2f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#0c7c8a")

    fig.suptitle("Strictly Matched Unimodal vs. Bimodal Audit & Downstream Late-Fusion Optimization", y=0.99)
    save_both(fig, "fig3_modality_optimization_ablations")


def make_fig4_class_viii_resolution() -> None:
    """Three-panel figure separating Native 5-Class recalls (A), Native 6-Class recalls (B), and Pairwise Co-Ni AUROC (C)."""
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(13.8, 4.4), gridspec_kw={"wspace": 0.28, "width_ratios": [1.35, 1.0, 0.72]}
    )

    # Panel A: Native 5-Class Resolution
    classes_5 = ["Mn\n(n=2,528)", "Zn\n(n=2,032)", "Cu\n(n=343)", "Fe\n(n=1,737)", "Co+Ni\n(n=758)"]
    recalls_5 = [75.8, 80.4, 72.0, 60.6, 27.4]
    c5 = ["#377eb8", "#4daf4a", "#984ea3", "#e41a1c", "#ff7f00"]
    b5 = ax1.bar(classes_5, recalls_5, width=0.54, color=c5, edgecolor="white")
    ax1.axhline(65.2, color="#111111", linestyle="--", linewidth=1.1, label="Discovery s42 Macro BA (65.2%)")
    ax1.axhline(63.8, color="#555555", linestyle="-.", linewidth=1.1, label="Confirm. s43/44 Macro BA (63.8%)")
    ax1.axhline(20.0, color="#999999", linestyle=":", linewidth=1.0, label="5-Class Random Chance (20.0%)")
    ax1.set_ylim(0, 96)
    ax1.set_ylabel("5-Fold Grouped CV Recall (%)")
    ax1.set_title("A. Native 5-Class (Fe vs. Co+Ni)", pad=10)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, fontsize=7.7)
    for idx, b in enumerate(b5):
        v = b.get_height()
        if idx == 3:
            # Place Fe 60.6% inside the top of the bar in bold white so it doesn't collide with the 63.8%/65.2% lines
            ax1.text(b.get_x() + b.get_width() / 2, v - 6.0, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=8.3, color="white")
        else:
            ax1.text(b.get_x() + b.get_width() / 2, v + 1.3, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=8.3)

    # Panel B: Native 6-Class Sub-Class VIII Recalls
    classes_6 = ["Fe Recall\n(n=1,737)", "Ni Recall\n(n=475)", "Co Recall\n(n=283, 3.8%)"]
    recalls_6 = [61.3, 29.5, 4.9]
    c6 = ["#e41a1c", "#ff7f00", "#fdb462"]
    b6 = ax2.bar(classes_6, recalls_6, width=0.50, color=c6, edgecolor="white")
    ax2.axhline(55.8, color="#111111", linestyle="--", linewidth=1.1, label="Discovery s42 6-Class BA (55.8%)")
    ax2.axhline(54.5, color="#555555", linestyle="-.", linewidth=1.1, label="Confirm. s43/44 6-Class BA (54.5%)")
    ax2.axhline(16.7, color="#999999", linestyle=":", linewidth=1.0, label="6-Class Random Chance (16.7%)")
    ax2.set_ylim(0, 96)
    ax2.set_ylabel("5-Fold Grouped CV Recall (%)")
    ax2.set_title("B. Native 6-Class (Fe / Ni / Co)", pad=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True, fontsize=7.7)
    for b in b6:
        v = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.3, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=8.3)

    # Panel C: Dedicated Pairwise Co vs. Ni AUROC axis (0.0 - 1.0)
    auroc_items = ["Random\nBaseline", "DeepMzyme 6-Class\nCo–Ni AUROC"]
    auroc_vals = [0.500, 0.680]
    yerr_lo = [0.0, 0.680 - 0.582]
    yerr_hi = [0.0, 0.770 - 0.680]
    b_auc = ax3.bar(
        auroc_items, auroc_vals, width=0.46, color=["#bdbdbd", "#08589e"], edgecolor="white",
        yerr=[yerr_lo, yerr_hi], error_kw={"ecolor": "#111111", "elinewidth": 1.3, "capsize": 4.0},
    )
    ax3.axhline(0.50, color="#666666", linestyle=":", linewidth=1.1)
    ax3.set_ylim(0.0, 0.96)
    ax3.set_ylabel("Pairwise ROC-AUC (Co vs. Ni)")
    ax3.set_title("C. Pairwise Co–Ni AUROC", pad=10)
    ax3.grid(axis="y", linestyle=":", alpha=0.5)
    ax3.text(b_auc[0].get_x() + b_auc[0].get_width() / 2, 0.515, "0.500", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#555555")
    ax3.text(b_auc[1].get_x() + b_auc[1].get_width() / 2, 0.785, "0.680\n[0.582, 0.770]", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#08306b")

    fig.suptitle("Native 5-Class and 6-Class Resolution Within Class VIII (Fe, Co, Ni) on 7,398 Sites", y=1.01)
    save_both(fig, "fig4_class_viii_resolution")


if __name__ == "__main__":
    make_fig1_benchmark_overview()
    make_fig2_per_metal_recall_audit()
    make_fig3_modality_optimization_ablations()
    make_fig4_class_viii_resolution()
    print(f"Generated 4 SVG/PNG figure pairs in {OUT_DIR}")
