"""Generate every figure of the paper (PDF) from the result CSVs.

Run after the experiment scripts:
    python experiments/make_figures.py
Figures are written to ``figures/`` and copied verbatim into the
paper's ``images/`` folder by the build script.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import (
    FIGURES, GRID, INK, INK2, MODEL_COLORS, MUTED, PALETTE, SERIES,
    load_csv, new_fig, reliability_curve, save_fig, set_style,
)

BLUE = PALETTE["blue"]
AQUA = PALETTE["aqua"]
YELLOW = PALETTE["yellow"]
VIOLET = PALETTE["violet"]
RED = PALETTE["red"]
GREEN = PALETTE["green"]
ORANGE = PALETTE["orange"]

SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# ---------------------------------------------------------------------------
# 1. Weight link functions
# ---------------------------------------------------------------------------

def fig_weight_functions():
    fig, ax = new_fig(3.45, 2.2)
    D = np.linspace(0, 1, 200)
    curves = [
        (r"linear $1-\bar D$", 1 - D, AQUA, "-"),
        (r"exponential $e^{-\gamma \bar D}$", np.exp(-D), YELLOW, "-"),
        (r"inverse $1/(1+\gamma \bar D)$", 1 / (1 + D), VIOLET, "-"),
        (r"harmonic $1/(1+\gamma\sum_j D_{ij}^2)$", 1 / (1 + 3 * D**2), BLUE, "-"),
    ]
    for label, w, c, ls in curves:
        ax.plot(D, w, color=c, ls=ls, label=label)
    ax.set_xlabel("dependence level")
    ax.set_ylabel("weight $w$")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", handlelength=1.6)
    save_fig(fig, "fig_weight_functions")


# ---------------------------------------------------------------------------
# 2-3. Dependence heatmaps + weight profile (breast cancer)
# ---------------------------------------------------------------------------

def fig_dependence_heatmaps():
    from sklearn.datasets import load_breast_cancer
    from abdnb.dependence import class_conditional_dependence, dependence_matrix

    data = load_breast_cancer()
    X, y = data.data, data.target
    marg = dependence_matrix(X, "spearman")
    mats, pooled = class_conditional_dependence(X, y, "spearman")
    fig, axes = new_fig(7.0, 2.35, ncols=3)
    titles = ["marginal (label-induced)", "within class: malignant",
              "within class: benign"]
    for ax, M, t in zip(axes, [marg, mats[0], mats[1]], titles):
        im = ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap
                       .from_list("seq", ["#ffffff"] + SEQ), vmin=0, vmax=1)
        ax.set_title(t)
        ax.set_xlabel("feature index")
        ax.grid(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("feature index")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
    cbar.set_label("Spearman dependence $\\hat D_{ij}$", fontsize=7)
    cbar.outline.set_visible(False)
    save_fig(fig, "fig_dependence_heatmaps")


def fig_weights_profile():
    from sklearn.datasets import load_breast_cancer
    from abdnb import ABDNB

    data = load_breast_cancer()
    m = ABDNB(gamma=1.0, class_specific=False).fit(data.data, data.target)
    w = m.weights_[0]
    order = np.argsort(w)
    names = np.array(data.feature_names)[order]
    fig, ax = new_fig(3.45, 3.4)
    ax.barh(np.arange(len(w)), w[order], color=BLUE, height=0.62)
    ax.axvline(1.0, color=MUTED, lw=0.8, ls="--")
    ax.text(1.02, len(w) - 2.0, "unweighted NB\n($w=1$)", fontsize=6,
            color=INK2, va="top")
    ax.set_yticks(np.arange(len(w)))
    ax.set_yticklabels(names, fontsize=5.5)
    ax.set_xlabel("adaptive weight $w_i$")
    ax.grid(axis="y", visible=False)
    d_eff = m.effective_dimension_[0]
    ax.set_title(f"Breast cancer: $d={len(w)}$, "
                 f"$d_{{\\mathrm{{eff}}}}={d_eff:.1f}$", fontsize=8)
    save_fig(fig, "fig_weights_profile")


# ---------------------------------------------------------------------------
# 4-5. Equicorrelation study
# ---------------------------------------------------------------------------

def _line_with_band(ax, x, mean, sd, color, label, marker="o"):
    ax.plot(x, mean, color=color, marker=marker, ms=3.5, label=label)
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.15, lw=0)


def fig_rho_study():
    df = load_csv("synthetic_rho")
    g = df.groupby(["model", "rho"]).agg(
        acc=("accuracy", "mean"), acc_sd=("accuracy", "std"),
        ll=("log_loss", "mean"), ll_sd=("log_loss", "std"),
        ece=("ece", "mean"), ece_sd=("ece", "std")).reset_index()
    fig, axes = new_fig(7.0, 2.2, ncols=3)
    for model in ["GNB", "ABD-NB", "LR"]:
        sub = g[g.model == model].sort_values("rho")
        c = MODEL_COLORS[model.replace("ABD-NB", "ABD-NB-C")] if model not in MODEL_COLORS else MODEL_COLORS[model]
        _line_with_band(axes[0], sub.rho, sub.acc, sub.acc_sd, c, model)
        _line_with_band(axes[1], sub.rho, sub.ll, sub.ll_sd, c, model)
        _line_with_band(axes[2], sub.rho, sub.ece, sub.ece_sd, c, model)
    axes[0].set_ylabel("accuracy")
    axes[1].set_ylabel("log-loss")
    axes[2].set_ylabel("ECE")
    for ax in axes:
        ax.set_xlabel(r"within-class correlation $\rho$")
    axes[0].legend(loc="lower left")
    save_fig(fig, "fig_rho_study")


def fig_conflict_study():
    df = load_csv("synthetic_conflict")
    g = df.groupby(["model", "r"]).agg(
        acc=("accuracy", "mean"), acc_sd=("accuracy", "std"),
        ll=("log_loss", "mean"), ll_sd=("log_loss", "std"),
        ece=("ece", "mean"), ece_sd=("ece", "std")).reset_index()
    fig, axes = new_fig(7.0, 2.2, ncols=3)
    for model in ["GNB", "ABD-NB", "LR"]:
        sub = g[g.model == model].sort_values("r")
        c = MODEL_COLORS.get(model, MODEL_COLORS["ABD-NB-C"])
        _line_with_band(axes[0], sub.r, sub.acc, sub.acc_sd, c, model)
        _line_with_band(axes[1], sub.r, sub.ll, sub.ll_sd, c, model)
        _line_with_band(axes[2], sub.r, sub.ece, sub.ece_sd, c, model)
    axes[0].set_ylabel("accuracy")
    axes[1].set_ylabel("log-loss")
    axes[2].set_ylabel("ECE")
    for ax in axes:
        ax.set_xlabel("redundant block size $r$")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xticklabels([1, 2, 4, 8, 16])
    axes[0].legend(loc="lower left")
    save_fig(fig, "fig_conflict_study")


# ---------------------------------------------------------------------------
# 6. Duplication stress on real data
# ---------------------------------------------------------------------------

def fig_redundancy():
    df = load_csv("redundancy")
    datasets = ["iris", "wine", "breast-cancer"]
    fig, axes = new_fig(7.0, 2.2, ncols=3, sharey=False)
    for ax, ds in zip(axes, datasets):
        sub = df[df.dataset == ds]
        g = sub.groupby(["model", "r"]).accuracy.agg(["mean", "std"]).reset_index()
        for model in ["GNB", "ABD-NB", "LR"]:
            s = g[g.model == model].sort_values("r")
            c = MODEL_COLORS.get(model, MODEL_COLORS["ABD-NB-C"])
            _line_with_band(ax, s.r, s["mean"], s["std"], c, model)
        ax.set_title(ds)
        ax.set_xlabel("number of duplicated copies $r$")
        ax.set_xscale("symlog", base=2, linthresh=1)
        ax.set_xticks([0, 1, 2, 4, 8, 16, 32])
        ax.set_xticklabels([0, 1, 2, 4, 8, 16, 32])
    axes[0].set_ylabel("accuracy (5-fold CV)")
    axes[0].legend(loc="lower left")
    save_fig(fig, "fig_redundancy")


# ---------------------------------------------------------------------------
# 7-9. Theory support
# ---------------------------------------------------------------------------

def fig_duplicates():
    df = load_csv("duplicates")
    fig, axes = new_fig(3.45, 2.0, ncols=2)
    axes[0].plot(df.r, df.w_theory, color=MUTED, ls="--", label="$1/r$ (theory)")
    axes[0].plot(df.r, df.w_copy_mean, color=BLUE, marker="o", ms=3.5,
                 lw=0, label="estimated")
    axes[0].set_xlabel("number of copies $r$")
    axes[0].set_ylabel("weight per copy")
    axes[0].legend()
    axes[1].plot(df.r, df.d_eff_theory, color=MUTED, ls="--",
                 label="$d^*=6$ (theory)")
    axes[1].plot(df.r, df.d_eff, color=BLUE, marker="o", ms=3.5, lw=0,
                 label="estimated")
    axes[1].set_xlabel("number of copies $r$")
    axes[1].set_ylabel("$d_{\\mathrm{eff}}$")
    axes[1].set_ylim(0, 8)
    axes[1].legend(loc="lower right")
    save_fig(fig, "fig_duplicates")


def fig_consistency():
    df = load_csv("consistency")
    g = df.groupby("n").err_l2.agg(["mean", "std"]).reset_index()
    bv = load_csv("bias_variance")
    fig, axes = new_fig(7.0, 2.3, ncols=2)
    ax = axes[0]
    ax.errorbar(g.n, g["mean"], yerr=g["std"], color=BLUE, marker="o",
                ms=3.5, capsize=2, lw=1.4, label=r"$\|\hat w - w^\ast\|_2$")
    ref = g["mean"].iloc[0] * np.sqrt(g.n.iloc[0]) / np.sqrt(g.n)
    ax.plot(g.n, ref, color=MUTED, ls="--", label=r"$C\,n^{-1/2}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample size $n$")
    ax.set_ylabel("weight estimation error")
    ax.legend()
    ax = axes[1]
    ax.plot(bv.n, np.abs(bv.bias), color=ORANGE, marker="o", ms=3.5,
            label="|bias|")
    ax.plot(bv.n, np.sqrt(bv.variance), color=AQUA, marker="s", ms=3.5,
            label="std. deviation")
    ax.plot(bv.n, np.sqrt(bv.mse), color=BLUE, marker="^", ms=3.5,
            label=r"$\sqrt{\mathrm{MSE}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample size $n$")
    ax.set_ylabel(r"error of raw weight $\tilde w_1$")
    ax.legend()
    save_fig(fig, "fig_consistency")


def fig_learning_curve():
    df = load_csv("learning_curve")
    g = df.groupby(["model", "n"]).accuracy.agg(["mean", "std"]).reset_index()
    fig, ax = new_fig(3.45, 2.3)
    for model in ["GNB", "ABD-NB"]:
        s = g[g.model == model].sort_values("n")
        c = MODEL_COLORS.get(model, MODEL_COLORS["ABD-NB-C"])
        _line_with_band(ax, s.n, s["mean"], s["std"], c, model)
    ax.set_xscale("log")
    ax.set_xlabel("training-set size $n$")
    ax.set_ylabel("test accuracy")
    ax.legend(loc="lower right")
    save_fig(fig, "fig_learning_curve")


# ---------------------------------------------------------------------------
# 10-12. Sensitivity / ablations
# ---------------------------------------------------------------------------

DS_COLORS = {
    "wine": AQUA, "breast-cancer": BLUE, "synth-conflict": RED,
    "synth-rho0.6": VIOLET, "synth-nonlin": ORANGE,
}


def fig_gamma_sensitivity():
    df = load_csv("sensitivity_gamma")
    fig, axes = new_fig(7.0, 2.3, ncols=2)
    for ds, sub in df.groupby("dataset"):
        sub = sub.sort_values("gamma")
        axes[0].plot(sub.gamma, sub.accuracy, marker="o", ms=3,
                     color=DS_COLORS[ds], label=ds)
        axes[1].plot(sub.gamma, sub.log_loss, marker="o", ms=3,
                     color=DS_COLORS[ds], label=ds)
    for ax in axes:
        ax.set_xscale("symlog", linthresh=0.25)
        ax.set_xticks([0, 0.25, 0.5, 1, 2, 4, 8])
        ax.set_xticklabels(["0", ".25", ".5", "1", "2", "4", "8"])
        ax.set_xlabel(r"correction strength $\gamma$")
    axes[0].set_ylabel("accuracy")
    axes[1].set_ylabel("log-loss")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=6, loc="lower left")
    save_fig(fig, "fig_gamma_sensitivity")


def _grouped_bars(ax, table, group_labels, series_labels, colors, width=None):
    n_g, n_s = len(group_labels), len(series_labels)
    width = width or 0.8 / n_s
    x = np.arange(n_g)
    for i, s in enumerate(series_labels):
        ax.bar(x + (i - (n_s - 1) / 2) * width, table[:, i], width * 0.92,
               color=colors[i], label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.grid(axis="x", visible=False)


def fig_measures():
    df = load_csv("sensitivity_measure")
    order = ["pearson", "spearman", "kendall", "mi", "cramersv", "dcor", "hsic"]
    datasets = ["wine", "breast-cancer", "synth-conflict", "synth-rho0.6",
                "synth-nonlin"]
    tab = df.pivot_table("accuracy", "dataset", "measure").loc[datasets, order]
    fig, ax = new_fig(7.0, 2.4)
    colors = [SERIES[i] for i in range(len(order))]
    _grouped_bars(ax, tab.values, datasets, order, colors)
    ax.set_ylabel("accuracy (5-fold CV)")
    ax.set_ylim(0.5, 1.0)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save_fig(fig, "fig_measures")


def fig_weightfn():
    df = load_csv("sensitivity_weightfn")
    order = ["linear", "exponential", "inverse", "harmonic"]
    datasets = ["wine", "breast-cancer", "synth-conflict", "synth-rho0.6",
                "synth-nonlin"]
    ta = df.pivot_table("accuracy", "dataset", "weight_fn").loc[datasets, order]
    tl = df.pivot_table("log_loss", "dataset", "weight_fn").loc[datasets, order]
    fig, axes = new_fig(7.0, 2.3, ncols=2)
    colors = [AQUA, YELLOW, VIOLET, BLUE]
    _grouped_bars(axes[0], ta.values, datasets, order, colors)
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0.5, 1.0)
    _grouped_bars(axes[1], tl.values, datasets, order, colors)
    axes[1].set_ylabel("log-loss")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=6)
    axes[0].legend(ncol=2, fontsize=6)
    save_fig(fig, "fig_weightfn")


# ---------------------------------------------------------------------------
# 13-16. Benchmark visualisations
# ---------------------------------------------------------------------------

NB_FAMILY = ["GNB", "MI-WNB", "ABD-NB-L", "ABD-NB-G", "ABD-NB-C", "ABD-NB"]
ALL_MODELS = NB_FAMILY + ["LR", "kNN", "CART", "RF", "SVM"]


def fig_benchmark_heatmap():
    df = load_csv("benchmark")
    tab = df.pivot_table("accuracy", "dataset", "model")[ALL_MODELS]
    # rank within each dataset (1 = best)
    ranks = tab.rank(axis=1, ascending=False)
    fig, ax = new_fig(7.0, 3.4)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "seqr", list(reversed(["#ffffff"] + SEQ)))
    im = ax.imshow(ranks.values, cmap=cmap, vmin=1, vmax=len(ALL_MODELS),
                   aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            r = ranks.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=5.6, color="white" if r <= 3 else INK)
    ax.set_xticks(range(len(ALL_MODELS)))
    ax.set_xticklabels(ALL_MODELS, rotation=35, ha="right", fontsize=6.5)
    ax.set_yticks(range(tab.shape[0]))
    ax.set_yticklabels(tab.index, fontsize=6.5)
    ax.grid(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cbar.set_label("rank (1 = best)", fontsize=7)
    cbar.outline.set_visible(False)
    save_fig(fig, "fig_benchmark_heatmap")


def fig_nb_family_bars():
    df = load_csv("benchmark")
    datasets = list(df.dataset.unique())
    tab = df.pivot_table("accuracy", "dataset", "model").loc[datasets, ["GNB", "MI-WNB", "ABD-NB"]]
    fig, ax = new_fig(7.0, 2.5)
    _grouped_bars(ax, tab.values, datasets,
                  ["GNB", "MI-WNB", "ABD-NB"],
                  [MODEL_COLORS["GNB"], MODEL_COLORS["MI-WNB"],
                   MODEL_COLORS["ABD-NB-C"]])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.tick_params(axis="x", rotation=35, labelsize=6)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    save_fig(fig, "fig_nb_family_bars")


def fig_calibration_benchmark():
    df = load_csv("benchmark")
    datasets = list(df.dataset.unique())
    te = df.pivot_table("ece", "dataset", "model").loc[datasets, ["GNB", "ABD-NB"]]
    tl = df.pivot_table("log_loss", "dataset", "model").loc[datasets, ["GNB", "ABD-NB"]]
    fig, axes = new_fig(7.0, 2.5, ncols=2)
    colors = [MODEL_COLORS["GNB"], MODEL_COLORS["ABD-NB-C"]]
    _grouped_bars(axes[0], te.values, datasets, ["GNB", "ABD-NB"], colors)
    axes[0].set_ylabel("ECE")
    _grouped_bars(axes[1], tl.values, datasets, ["GNB", "ABD-NB"], colors)
    axes[1].set_ylabel("log-loss")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45, labelsize=5.4)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    axes[0].legend(ncol=2)
    save_fig(fig, "fig_calibration_benchmark")


def fig_boxplots():
    df = load_csv("benchmark")
    datasets = ["breast-cancer", "wine-red", "synth-conflict", "synth-blocks"]
    models = ["GNB", "MI-WNB", "ABD-NB", "LR", "RF"]
    fig, axes = new_fig(7.0, 2.3, ncols=len(datasets), sharey=False)
    for ax, ds in zip(axes, datasets):
        data = [df[(df.dataset == ds) & (df.model == m)].accuracy.values
                for m in models]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops=dict(color=INK, lw=1.2),
                        flierprops=dict(marker="o", ms=2, mfc=MUTED, mec=MUTED),
                        whiskerprops=dict(color=MUTED, lw=0.8),
                        capprops=dict(color=MUTED, lw=0.8))
        for patch, m in zip(bp["boxes"], models):
            c = MODEL_COLORS.get(m, MODEL_COLORS["ABD-NB-C"])
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
            patch.set_edgecolor(c)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=6)
        ax.set_title(ds)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("accuracy per fold")
    save_fig(fig, "fig_boxplots")


def fig_runtime():
    df = load_csv("benchmark")
    g = df.groupby("model")[["fit_time", "predict_time"]].mean().loc[ALL_MODELS]
    fig, ax = new_fig(3.45, 2.4)
    x = np.arange(len(ALL_MODELS))
    ax.bar(x - 0.2, g.fit_time, 0.38, color=BLUE, label="fit")
    ax.bar(x + 0.2, g.predict_time, 0.38, color=YELLOW, label="predict")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_MODELS, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("time per fold (s)")
    ax.grid(axis="x", visible=False)
    ax.legend()
    save_fig(fig, "fig_runtime")


# ---------------------------------------------------------------------------
# 17. Critical-difference diagram (Demsar)
# ---------------------------------------------------------------------------

def fig_cd_diagram():
    ranks = load_csv("stats_ranks").set_index("model")["avg_rank"]
    fr = load_csv("stats_friedman")
    cd = float(fr["cd"].iloc[0])
    ranks = ranks.sort_values()
    k = len(ranks)
    lo, hi = 1, k
    set_style()
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(hi + 0.3, lo - 0.3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y_axis = 0.88
    ax.hlines(y_axis, lo, hi, color=INK, lw=1.2)
    for t in range(lo, hi + 1):
        ax.vlines(t, y_axis, y_axis + 0.035, color=INK, lw=1.0)
        ax.text(t, y_axis + 0.075, str(t), ha="center", fontsize=7, color=INK2)
    # CD ruler
    ax.hlines(y_axis + 0.16, lo, lo + cd, color=INK2, lw=1.6)
    ax.text(lo + cd / 2, y_axis + 0.20, f"CD = {cd:.2f}", ha="center",
            fontsize=7, color=INK2)
    # model stems: left column = best half, right column = worst half
    names = list(ranks.index)
    half = int(np.ceil(k / 2))
    slot_h = 0.60 / half
    for i, name in enumerate(names):
        r = ranks[name]
        left = i < half
        yy = y_axis - 0.14 - (i if left else k - 1 - i) * slot_h
        xx = lo - 0.25 if left else hi + 0.25
        ax.plot([r, r, xx], [y_axis, yy, yy], color=MUTED, lw=0.9)
        c = MODEL_COLORS.get(name, MODEL_COLORS["ABD-NB-C"])
        weight = "bold" if name == "ABD-NB" else "normal"
        ax.text(xx + (0.06 if left else -0.06), yy,
                f"{name}  ({r:.2f})", fontsize=7, color=c,
                ha="left" if left else "right", va="center", fontweight=weight)
    # cliques: groups not significantly different
    ys = y_axis - 0.075
    rank_vals = ranks.values
    drawn = []
    for i in range(k):
        j = i
        while j + 1 < k and rank_vals[j + 1] - rank_vals[i] <= cd:
            j += 1
        if j > i and not any(i >= a and j <= b for a, b in drawn):
            ax.hlines(ys, rank_vals[i] - 0.05, rank_vals[j] + 0.05,
                      color=INK, lw=2.4, alpha=0.65)
            drawn.append((i, j))
            ys -= 0.05
    save_fig(fig, "fig_cd_diagram")


# ---------------------------------------------------------------------------
# 18. Reliability diagrams
# ---------------------------------------------------------------------------

def fig_reliability():
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from abdnb import ABDNB
    from experiments.datasets import conflict_block, equicorrelated_gaussian

    configs = [("conflict ($r=12$)", conflict_block(n=4000, seed=11)),
               (r"equicorrelated ($\rho=0.6$)",
                equicorrelated_gaussian(0.6, n=4000, d=12, delta=2.0, seed=11))]
    fig, axes = new_fig(7.0, 2.5, ncols=2)
    for ax, (title, (X, y)) in zip(axes, configs):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5,
                                              random_state=0, stratify=y)
        ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=0.9,
                label="perfect calibration")
        for name, mk in [("GNB", GaussianNB), ("ABD-NB", ABDNB)]:
            m = mk().fit(Xtr, ytr)
            p = m.predict_proba(Xte)
            curve = reliability_curve(yte, p, n_bins=12)
            c = MODEL_COLORS.get(name, MODEL_COLORS["ABD-NB-C"])
            ax.plot(curve[:, 0], curve[:, 1], color=c, marker="o", ms=3.5,
                    label=name)
        ax.set_xlabel("mean predicted confidence")
        ax.set_title(title)
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(0.35, 1.02)
    axes[0].set_ylabel("empirical accuracy")
    axes[0].legend(loc="upper left", fontsize=6.5)
    save_fig(fig, "fig_reliability")


# ---------------------------------------------------------------------------
# 19. Concept drift
# ---------------------------------------------------------------------------

def fig_drift():
    df = load_csv("drift")
    fig, axes = new_fig(7.0, 2.4, ncols=2)
    ax = axes[0]
    abd = df[df.model == "Online ABD-NB"]
    g = abd.groupby("t")[["w_strong", "w_blockA", "w_blockB"]].mean()
    ax.axvline(4000, color=MUTED, ls=":", lw=1)
    ax.text(4050, 0.15, "dependence switch", fontsize=6, color=INK2, rotation=90)
    ax.plot(g.index, g.w_strong, color=GREEN, label="strong indep. feature")
    ax.plot(g.index, g.w_blockA, color=BLUE, label="block A (redundant $\\to$ indep.)")
    ax.plot(g.index, g.w_blockB, color=RED, label="block B (indep. $\\to$ redundant)")
    ax.set_xlabel("stream position $t$")
    ax.set_ylabel("online weight $w_i(t)$")
    ax.legend(fontsize=6, loc="center right")
    ax = axes[1]
    for model, c in [("Online GNB", MODEL_COLORS["GNB"]),
                     ("Online ABD-NB", MODEL_COLORS["ABD-NB-C"])]:
        sub = df[df.model == model].groupby("t").acc_window.agg(["mean", "std"])
        _line_with_band(ax, sub.index, sub["mean"], sub["std"], c, model,
                        marker="")
    ax.axvline(4000, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("stream position $t$")
    ax.set_ylabel("prequential accuracy (window 500)")
    ax.legend(fontsize=6.5, loc="lower right")
    save_fig(fig, "fig_drift")


# ---------------------------------------------------------------------------

FIGS = [
    fig_weight_functions,
    fig_dependence_heatmaps,
    fig_weights_profile,
    fig_rho_study,
    fig_conflict_study,
    fig_redundancy,
    fig_duplicates,
    fig_consistency,
    fig_learning_curve,
    fig_gamma_sensitivity,
    fig_measures,
    fig_weightfn,
    fig_benchmark_heatmap,
    fig_nb_family_bars,
    fig_calibration_benchmark,
    fig_boxplots,
    fig_runtime,
    fig_cd_diagram,
    fig_reliability,
    fig_drift,
]

if __name__ == "__main__":
    only = sys.argv[1:] or None
    for f in FIGS:
        if only and not any(o in f.__name__ for o in only):
            continue
        try:
            f()
        except FileNotFoundError as e:
            print(f"  [skip] {f.__name__}: missing input ({e})")
        except Exception as e:
            print(f"  [FAIL] {f.__name__}: {type(e).__name__}: {e}")
