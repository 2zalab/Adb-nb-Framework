"""Emit every LaTeX table body of the paper from the result CSVs.

Each function writes a fragment to ``results/tables/<name>.tex`` that is
included verbatim in the manuscript, so no number in the paper is typed
by hand.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.common import RESULTS, load_csv
from experiments.datasets import dataset_stratum

TABLES = os.path.join(RESULTS, "tables")
os.makedirs(TABLES, exist_ok=True)


def _write(name: str, body: str) -> None:
    path = os.path.join(TABLES, f"{name}.tex")
    with open(path, "w") as fh:
        # The fragment carries its own \bottomrule: a file that ends on a
        # row terminator makes \input cross the alignment boundary in the
        # middle of the \\ lookahead, which LaTeX reports as a misplaced
        # \noalign.  Closing the rule here keeps every fragment valid.
        fh.write(body.rstrip() + "\n\t\t\t\\bottomrule\n")
    print(f"  [tex] {name}.tex")


def _esc(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%")


# ---------------------------------------------------------------------------

def table_nb_family():
    """Accuracy of the naive Bayes family + the semi-naive references."""
    df = load_csv("benchmark")
    cols = ["GNB", "MI-WNB", "CW-NB", "WANBIA", "ABD-NB-G", "ABD-NB",
            "TAN", "AODE"]
    acc = df.pivot_table("accuracy", "dataset", "model")
    sd = df.groupby(["dataset", "model"])["accuracy"].std().unstack()
    order = sorted(acc.index, key=lambda d: ({"real": 0, "augmented": 1,
                                              "synthetic": 2}[dataset_stratum(d)], d))
    lines, prev = [], None
    for d in order:
        st = dataset_stratum(d)
        if st != prev:
            label = {"real": "\\emph{Real-world benchmarks}",
                     "augmented": "\\emph{Redundancy-augmented real data}",
                     "synthetic": "\\emph{Controlled synthetic families}"}[st]
            # a \multicolumn may not be the first token of an \input-ed
            # fragment (TeX is not at a cell boundary there), so the
            # stratum header is written as an ordinary row
            lines.append(f"\t\t\t{label}" + " &" * len(cols) + " \\\\")
            prev = st
        best = max(acc.loc[d, c] for c in cols)
        cells = []
        for c in cols:
            v = 100 * acc.loc[d, c]
            txt = f"{v:.1f}"
            if acc.loc[d, c] >= best - 1e-12:
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(f"\t\t\t{_esc(d)} & " + " & ".join(cells) + " \\\\")
    ranks = acc.rank(axis=1, ascending=False).mean()
    lines.append("\t\t\t\\midrule")
    lines.append("\t\t\tAvg.\\ rank (20 models) & " +
                 " & ".join(f"{ranks[c]:.2f}" for c in cols) + " \\\\")
    real = [d for d in order if dataset_stratum(d) == "real"]
    ranks_r = acc.loc[real].rank(axis=1, ascending=False).mean()
    lines.append("\t\t\tAvg.\\ rank (real only) & " +
                 " & ".join(f"{ranks_r[c]:.2f}" for c in cols) + " \\\\")
    _write("nb_family", "\n".join(lines))


def table_ranks():
    """Mean ranks of all models on the three metrics."""
    df = load_csv("benchmark")
    out = {}
    for metric, asc in (("accuracy", False), ("log_loss", True), ("ece", True)):
        tab = df.pivot_table(metric, "dataset", "model").dropna(axis=1)
        out[metric] = tab.rank(axis=1, ascending=asc).mean()
        real = [d for d in tab.index if dataset_stratum(d) == "real"]
        out[metric + "_real"] = tab.loc[real].rank(axis=1, ascending=asc).mean()
    r = pd.DataFrame(out).sort_values("accuracy")
    lines = []
    for m, row in r.iterrows():
        lines.append(f"\t\t\t{_esc(m)} & {row['accuracy']:.2f} & {row['accuracy_real']:.2f} & "
                     f"{row['log_loss']:.2f} & {row['log_loss_real']:.2f} & "
                     f"{row['ece']:.2f} & {row['ece_real']:.2f} \\\\")
    _write("ranks", "\n".join(lines))


def table_wilcoxon():
    w = load_csv("stats_wilcoxon")
    wl = load_csv("stats_wilcoxon_logloss").set_index("opponent")
    # Grouped by family, then as Section 6.2 lists them.  Any opponent the
    # statistics file carries but this order forgets is appended rather than
    # silently dropped: an earlier version of this list predated HNB and CFW
    # and quietly omitted both from the table while the text counted them.
    order = ["GNB", "MI-WNB", "CW-NB", "CFW-NB", "WANBIA", "ABD-NB-L",
             "ABD-NB-G", "ABD-NB-C", "TAN", "KDB", "AODE", "HNB",
             "GNB-Platt", "GNB-Iso", "LR", "kNN", "CART", "RF", "XGB",
             "LGBM", "SVM"]
    w = w.set_index("opponent")
    missing = [m for m in w.index if m not in order]
    if missing:
        print(f"  [warn] opponents absent from the table order, appended: {missing}")
        order = order + missing
    lines = []
    for m in order:
        if m not in w.index:
            continue
        a, b = w.loc[m], wl.loc[m]
        lines.append(
            f"\t\t\t{_esc(m)} & {a.wins:.0f}/{a.ties:.0f}/{a.losses:.0f} & "
            f"{a.p_raw:.3f} & {a.p_holm:.3f} & "
            f"{b.wins:.0f}/{b.ties:.0f}/{b.losses:.0f} & {b.p_raw:.3f} & {b.p_holm:.3f} \\\\")
    _write("wilcoxon", "\n".join(lines))


def table_composition():
    d = load_csv("composition")
    order = ["GNB", "GNB+Platt", "GNB+Iso", "ABD-NB", "ABD-NB+Platt", "ABD-NB+Iso"]
    acc = d.pivot_table("accuracy", "dataset", "model")[order]
    ll = d.pivot_table("log_loss", "dataset", "model")[order]
    ece = d.pivot_table("ece", "dataset", "model")[order]
    ra = acc.rank(axis=1, ascending=False).mean()
    rl = ll.rank(axis=1).mean()
    lines = []
    for m in order:
        lines.append(f"\t\t\t{_esc(m)} & {100*acc[m].mean():.2f} & {ra[m]:.2f} & "
                     f"{ll[m].median():.3f} & {rl[m]:.2f} & {ece[m].median():.3f} \\\\")
    _write("composition", "\n".join(lines))


def table_conditions():
    s = load_csv("conditions_summary")
    lines, prev = [], None
    for _, r in s.iterrows():
        if r["factor"] != prev:
            lines.append("\t\t\t\\midrule" if prev is not None else "")
            prev = r["factor"]
            label = r["factor"]
        else:
            label = ""
        verdict = {"help": "\\textbf{helps}", "hurt": "\\textbf{hurts}",
                   "parity": "parity"}[r["verdict"]]
        lines.append(
            f"\t\t\t{label} & {r['value']} & {r['d_acc']:+.2f} & "
            f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] & {r['d_logloss']:+.3f} & "
            f"{r['d_ece']:+.3f} & {r['gamma_med']:.2f} & {r['deff_ratio']:.2f} & {verdict} \\\\")
    _write("conditions", "\n".join(x for x in lines if x))


def table_redundancy_bands():
    df = load_csv("redundancy_profile")
    st = load_csv("redundancy_profile_stats")
    q = df["redundancy"].quantile([1 / 3, 2 / 3]).to_numpy()
    df["band"] = np.where(df["redundancy"] <= q[0], "low",
                          np.where(df["redundancy"] <= q[1], "medium", "high"))
    lines = []
    for band in ("low", "medium", "high"):
        g = df[df["band"] == band]
        lines.append(f"\t\t\t{band} ($R \\in [{g['redundancy'].min():.2f}, "
                     f"{g['redundancy'].max():.2f}]$) & {len(g)} & "
                     f"{g['redundancy'].mean():.2f} & {g['d_acc'].mean():+.2f} & "
                     f"{g['d_acc'].max():+.2f} & {g['d_ll_rel'].mean():+.1f} & "
                     f"{g['d_ll_rel'].max():+.1f} \\\\")
    _write("redundancy_bands", "\n".join(lines))
    lines = []
    for _, r in st.iterrows():
        lines.append(f"\t\t\t{_esc(r['target'])} & {r['spearman_rho']:.3f} & "
                     f"{r['spearman_p']:.1e} & {r['pearson_r']:.3f} & "
                     f"{r['pearson_p']:.1e} \\\\")
    _write("redundancy_corr", "\n".join(lines))


def table_categorical():
    d = load_csv("categorical")
    order = ["Cat-NB", "Cat-ABD-G", "Cat-ABD", "TAN", "AODE"]
    acc = d.pivot_table("accuracy", "dataset", "model")
    ll = d.pivot_table("log_loss", "dataset", "model")
    gam = d[d.model == "Cat-ABD"].groupby("dataset")["gamma"].median()
    cols = [c for c in order if c in acc.columns]
    lines = []
    for ds in acc.index:
        best_a = acc.loc[ds, cols].max()
        best_l = ll.loc[ds, cols].min()
        cells = []
        for c in cols:
            a = f"{100*acc.loc[ds, c]:.1f}"
            l = f"{ll.loc[ds, c]:.3f}"
            if acc.loc[ds, c] >= best_a - 1e-12:
                a = f"\\textbf{{{a}}}"
            if ll.loc[ds, c] <= best_l + 1e-12:
                l = f"\\textbf{{{l}}}"
            cells += [a, l]
        lines.append(f"\t\t\t{_esc(ds)} & " + " & ".join(cells) +
                     f" & {gam.get(ds, float('nan')):.2f} \\\\")
    ra = acc[cols].rank(axis=1, ascending=False).mean()
    rl = ll[cols].rank(axis=1).mean()
    lines.append("\t\t\t\\midrule")
    lines.append("\t\t\tAvg.\\ rank & " +
                 " & ".join(f"{ra[c]:.2f} & {rl[c]:.2f}" for c in cols) + " & \\\\")
    _write("categorical", "\n".join(lines))


def table_finite_sample():
    dev = load_csv("finite_sample_deviation")
    lines = []
    for n, g in dev.groupby("n"):
        r0 = g.iloc[0]
        p02 = g[g["eps"] == 0.02].iloc[0]
        p05 = g[g["eps"] == 0.05].iloc[0]
        p10 = g[g["eps"] == 0.10].iloc[0]
        lines.append(
            f"\t\t\t{int(n)} & {r0['mean_dev']:.4f} & {r0['q95_dev']:.4f} & "
            f"{r0['mean_dev']*np.sqrt(n):.3f} & {p02['empirical']:.3f} & "
            f"{p05['empirical']:.3f} & {p10['empirical']:.3f} & "
            f"{r0['L_implied']:.3f} \\\\")
    _write("finite_sample", "\n".join(lines))

    reg = load_csv("finite_sample_regret_summary")
    lines = []
    for _, r in reg.iterrows():
        lines.append(
            f"\t\t\t{int(r['n'])} & {100*r['selected']:.2f} & {100*r['vs_nb']:.2f} & "
            f"{100*r['regret_mean']:+.2f} & {100*r['regret_q95']:+.2f} & "
            f"{100*r['bound']:.1f} \\\\")
    _write("selection_regret", "\n".join(lines))


def table_gamma_selection():
    """What the selector actually chooses, and how it tracks redundancy."""
    marg = load_csv("gamma_selection_marginal")
    lines = []
    for _, r in marg.iterrows():
        g = r["gamma"]
        label = f"${g:g}$" + ("\\ (= plain naive Bayes)" if g == 0 else "")
        lines.append(f"\t\t\t{label} & {int(r['n_fits'])} & {r['all']:.1f} & "
                     f"{r['real']:.1f} & {r['augmented']:.1f} & {r['synthetic']:.1f} \\\\")
    _write("gamma_marginal", "\n".join(lines))

    resc = load_csv("gamma_selection_rescale")
    _write("gamma_rescale", "\n".join(
        f"\t\t\t{r['rule']} & {int(r['n_fits'])} & {r['all']:.1f} & {r['real']:.1f} \\\\"
        for _, r in resc.iterrows()))

    bands = load_csv("gamma_selection_bands")
    has_meff = "meff_pct" in bands.columns
    lines = []
    for _, r in bands.iterrows():
        extra = f" & {r['meff_pct']:.1f}" if has_meff else ""
        lines.append(f"\t\t\t{r['band']} & {int(r['n_datasets'])} & "
                     f"{r['zero_pct']:.1f} & {r['ge1_pct']:.1f} & "
                     f"{r['cls_pct']:.1f}{extra} & {r['deff_ratio']:.2f} \\\\")
    _write("gamma_bands", "\n".join(lines))


def table_ablation_components():
    s = load_csv("ablation_components_summary")
    lines = []
    for _, r in s.iterrows():
        lines.append(f"\t\t\t{r['config']} & {r['acc_rank']:.2f} & {r['ll_rank']:.2f} & "
                     f"{r['d_acc_mean']:+.2f} & {r['d_acc_worst']:+.2f} & "
                     f"{r['d_ll_median']:+.1f} & {int(r['wins'])}/{int(r['losses'])} \\\\")
    _write("ablation_components", "\n".join(lines))


def table_redundancy_measures():
    s = load_csv("redundancy_measures_stats")
    lines = []
    for _, r in s.iterrows():
        lines.append(f"\t\t\t{_esc(r['measure'])} & {r['mean_R']:.3f} & {r['rho_acc']:.3f} & "
                     f"{r['p_acc']:.3f} & {r['rho_ll']:.3f} & {r['p_ll']:.1e} & "
                     f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] \\\\")
    _write("redundancy_measures", "\n".join(lines))

    st = load_csv("redundancy_stability")
    cols = [c for c in st.columns if c.startswith("R_n")]
    lines = [f"\t\t\tbootstrap SD of $R$ & {st['boot_sd'].median():.4f} \\\\",
             f"\t\t\tbootstrap CV of $R$ & {st['boot_cv'].median():.3f} \\\\",
             f"\t\t\tsplit-half $|\\Delta R|$ & {st['splithalf_absdiff'].median():.4f} \\\\"]
    for c in sorted(cols, key=lambda x: int(x[3:])):
        nn = int(c[3:])
        b = st[f"bias_n{nn}"].dropna()
        if len(b):
            lines.append(f"\t\t\tbias of $R$ at $n={nn}$ & {b.median():+.4f} \\\\")
    _write("redundancy_stability", "\n".join(lines))


def table_higher_order():
    s = load_csv("higher_order_summary")
    lines = []
    for _, r in s.iterrows():
        lines.append(f"\t\t\t{_esc(r['design'])} & {r['max_rho']:.3f} & {r['max_mi']:.3f} & "
                     f"{r['gamma']:.2f} & {r['deff_ratio']:.2f} & "
                     f"{100*r['nb']:.1f} & {100*r['abd']:.1f} & {100*r['aode']:.1f} & "
                     f"{100*r['rf']:.1f} & {r['d_acc']:+.2f} \\\\")
    _write("higher_order", "\n".join(lines))


def table_scalability():
    f = load_csv("scalability_exponents")
    lines = []
    for _, r in f.iterrows():
        lines.append(f"\t\t\t${r['axis']}$ & {_esc(r['model'])} & {r['exponent']:.2f} & "
                     f"{r['exponent_asym']:.2f} & {r['t_min_ms']:.0f} & {r['t_max_ms']:.0f} \\\\")
    _write("scalability_cost", "\n".join(lines))

    full = load_csv("scalability_full")
    acc = full.pivot_table("accuracy", "dataset", "model")
    ll = full.pivot_table("log_loss", "dataset", "model")
    ft = full.pivot_table("fit_time", "dataset", "model")
    meta = full.groupby("dataset")[["n", "d"]].first()
    lines = []
    for ds in acc.index:
        best = acc.loc[ds].max()
        cells = []
        for m in ("GNB", "ABD-NB", "AODE", "LGBM"):
            v = f"{100*acc.loc[ds, m]:.1f}"
            if acc.loc[ds, m] >= best - 1e-12:
                v = f"\\textbf{{{v}}}"
            cells.append(v)
        dll = 100 * (ll.loc[ds, "GNB"] - ll.loc[ds, "ABD-NB"]) / ll.loc[ds, "GNB"]
        lines.append(f"\t\t\t{_esc(ds)} & {int(meta.loc[ds,'n'])} & {int(meta.loc[ds,'d'])} & "
                     + " & ".join(cells) + f" & {dll:+.1f} & {ft.loc[ds,'ABD-NB']:.2f} \\\\")
    _write("scalability_full", "\n".join(lines))


def table_degeneracy():
    """Why the Gaussian instantiation fails where it fails."""
    dg = load_csv("degeneracy").set_index("dataset")
    full = load_csv("scalability_full")
    acc = full.pivot_table("accuracy", "dataset", "model") * 100
    order = ["kr-vs-kp", "ann-thyroid", "nursery", "coil2000",
             "pendigits", "letter", "texture-full"]
    alias = {"nursery": "nursery-full"}
    lines = []
    for ds in order:
        if ds not in dg.index:
            continue
        r = dg.loc[ds]
        key = alias.get(ds, ds)
        if key in acc.index:
            cells = [f"{acc.loc[key, m]:.1f}" for m in ("GNB", "ABD-NB", "AODE")]
        else:
            cells = ["---"] * 3
        lines.append(
            f"\t\t\t{_esc(ds)} & {int(r['d'])} & {int(r['binary'])} & "
            f"{r['median_levels']:.0f} & {int(r['features_constant_in_some_class'])} & "
            + " & ".join(cells) + " \\\\")
    _write("degeneracy", "\n".join(lines))


def table_stages():
    """Per-stage cost, which is what tests the complexity analysis directly."""
    st = load_csv("scalability_stages")
    lines = []
    for axis, var, other in (("n", "n", "d"), ("d", "d", "n")):
        sub = st[st["axis"] == axis].sort_values(var)
        for _, r in sub.iterrows():
            lines.append(
                f"\t\t\t${axis}={int(r[var])}$ & {1000*r['dependence']:.1f} & "
                f"{1000*r['eigendecomposition']:.2f} & "
                f"{1000*r['gaussian_statistics']:.1f} & "
                f"{100*r['eigendecomposition']/r['dependence']:.2f} \\\\")
        e = sub.iloc[0]
        lines.append(
            f"\t\t\t\\emph{{exponent in }}${axis}$ & "
            f"{e['dependence_exponent']:.2f} & {e['eigendecomposition_exponent']:.2f} & "
            f"{e['gaussian_statistics_exponent']:.2f} & --- \\\\")
        if axis == "n":
            lines.append("\t\t\t\\midrule")
    _write("scalability_stages", "\n".join(lines))


def table_tuned():
    s = load_csv("tuned_baselines_summary")
    piv = s.pivot_table(index="model", values=["acc_mean", "acc_rank", "ll_rank",
                                               "fit_time"], columns="protocol")
    order = piv[("acc_rank", "tuned")].sort_values().index
    lines = []
    for m in order:
        lines.append(
            f"\t\t\t{_esc(m)} & {piv.loc[m,('acc_mean','default')]:.2f} & "
            f"{piv.loc[m,('acc_mean','tuned')]:.2f} & "
            f"{piv.loc[m,('acc_rank','default')]:.2f} & {piv.loc[m,('acc_rank','tuned')]:.2f} & "
            f"{piv.loc[m,('ll_rank','default')]:.2f} & {piv.loc[m,('ll_rank','tuned')]:.2f} & "
            f"{piv.loc[m,('fit_time','tuned')]:.2f} \\\\")
    _write("tuned_baselines", "\n".join(lines))


ALL = [table_stages, table_degeneracy, table_nb_family, table_ranks, table_wilcoxon, table_composition,
       table_conditions, table_redundancy_bands, table_categorical,
       table_finite_sample, table_gamma_selection,
       table_ablation_components, table_redundancy_measures,
       table_higher_order, table_scalability, table_tuned]

if __name__ == "__main__":
    for f in ALL:
        try:
            f()
        except FileNotFoundError as e:
            print(f"  [skip] {f.__name__}: {e}")
        except Exception as e:
            print(f"  [FAIL] {f.__name__}: {type(e).__name__}: {e}")
