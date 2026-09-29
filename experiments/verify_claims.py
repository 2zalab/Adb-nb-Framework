"""Check every numeric claim of the manuscript against the result files.

Each entry names a claim, the value recomputed from the released CSVs,
and the number printed in the paper.  The script fails loudly on any
mismatch, which is what keeps the text and the experiments in step after
a re-run.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.common import load_csv
from experiments.datasets import dataset_stratum

CHECKS = []


def c(claim, value, paper, tol=0.05):
    CHECKS.append((claim, float(value), paper,
                   "OK" if abs(float(value) - paper) <= tol else "MISMATCH"))


def main():
    b = load_csv("benchmark")
    acc = b.pivot_table("accuracy", "dataset", "model")
    ll = b.pivot_table("log_loss", "dataset", "model")
    ece = b.pivot_table("ece", "dataset", "model")
    real = [d for d in acc.index if dataset_stratum(d) == "real"]
    da = (acc["ABD-NB"] - acc["GNB"]) * 100
    dl = 100 * (ll["GNB"] - ll["ABD-NB"]) / ll["GNB"]
    ra = acc.rank(axis=1, ascending=False).mean()
    rl = ll.rank(axis=1).mean()

    c("models benchmarked", b["model"].nunique(), 22, 0)
    c("settings", b["dataset"].nunique(), 48, 0)
    c("total fits", len(b), 31680, 0)
    c("LL improved (count)", (dl > 0).sum(), 39, 0)
    c("LL max reduction", dl.max(), 85.2, 0.1)
    c("acc best gain", da.max(), 14.00, 0.05)
    c("acc worst", da.min(), -1.31, 0.05)
    c("ECE improved", ((ece["GNB"] - ece["ABD-NB"]) > 0).sum(), 31, 0)
    c("ABD-NB acc rank", ra["ABD-NB"], 13.46, 0.02)
    c("ABD-NB LL rank", rl["ABD-NB"], 12.15, 0.02)
    c("GNB LL rank", rl["GNB"], 18.06, 0.02)
    c("HNB acc rank", ra["HNB"], 10.02, 0.02)
    c("CFW-NB acc rank", ra["CFW-NB"], 13.41, 0.02)
    # abstract / discussion / conclusion headline numbers
    c("LL improved >1%", (dl > 1).sum(), 36, 0)
    c("acc real best gain", da[real].max(), 14.00, 0.05)
    c("acc synth best gain", da["synth-conflict"], 11.28, 0.05)
    c("LL reductions above 40%", (dl > 40).sum(), 13, 0)
    c("models (conclusion)", acc.shape[1], 22, 0)


    ab = load_csv("ablation_components")
    a2 = ab.pivot_table("accuracy", "dataset", "config")
    l2 = ab.pivot_table("log_loss", "dataset", "config")
    e, g = "(e) W+sel", "(g) W+scale$M$+sel"
    bb, dd = "(b) W", "(d) W+scale$M$"
    c("ablation e-vs-g dacc", ((a2[e] - a2[g]) * 100).mean(), 0.07, 0.02)
    c("ablation e-vs-g p (acc)",
      stats.wilcoxon(a2[e], a2[g], zero_method="pratt").pvalue, 0.64, 0.01)
    c("ablation e-vs-g dLL median",
      (100 * (l2[g] - l2[e]) / l2[g]).median(), 8.53, 0.05)
    c("ablation b-vs-d dLL median",
      (100 * (l2[dd] - l2[bb]) / l2[dd]).median(), 8.02, 0.05)

    s = load_csv("ablation_components_summary").set_index("config")
    c("ablation (b) worst", s.loc[bb, "d_acc_worst"], -10.52, 0.05)
    c("ablation (e) worst", s.loc[e, "d_acc_worst"], -1.51, 0.05)
    c("ablation (b) dLL", s.loc[bb, "d_ll_median"], 14.0, 0.1)
    c("ablation (e) dLL", s.loc[e, "d_ll_median"], 20.9, 0.1)
    c("ablation (e) dacc mean", s.loc[e, "d_acc_mean"], 1.02, 0.02)

    pr = load_csv("redundancy_profile_stats")
    c("R vs acc rho", pr.iloc[0]["spearman_rho"], 0.508, 0.005)
    c("R vs LL rho", pr.iloc[1]["spearman_rho"], 0.616, 0.005)

    rm = load_csv("redundancy_measures_stats")
    c("R robustness: min rho(LL)", rm["rho_ll"].min(), 0.523, 0.01)
    c("R robustness: max rho(LL)", rm["rho_ll"].max(), 0.676, 0.01)
    c("R robustness: measures", len(rm), 7, 0)
    st = load_csv("redundancy_stability")
    c("R bootstrap CV (median)", st["boot_cv"].median(), 0.029, 0.003)
    c("R split-half (median)", st["splithalf_absdiff"].median(), 0.0077, 0.001)

    ho = load_csv("higher_order_summary").set_index("design")
    c("parity-3 max rho", ho.loc["parity-3", "max_rho"], 0.010, 0.005)
    c("parity-3 RF acc", 100 * ho.loc["parity-3", "rf"], 79.9, 0.2)
    c("parity-3 dacc", ho.loc["parity-3", "d_acc"], -0.01, 0.05)
    c("masked-dup dacc", ho.loc["masked-duplicate", "d_acc"], -0.31, 0.05)
    c("masked-dup deff ratio", ho.loc["masked-duplicate", "deff_ratio"], 0.56, 0.01)

    sc = load_csv("scalability_exponents").set_index(["axis", "model"])
    c("cost exponent n (flagship)", sc.loc[("n", "ABD-NB"), "exponent_asym"], 0.86, 0.02)
    c("cost exponent d (flagship)", sc.loc[("d", "ABD-NB"), "exponent_asym"], 1.14, 0.02)
    c("cost exponent n (fixed)", sc.loc[("n", "ABD-NB (fixed)"), "exponent_asym"], 1.00, 0.02)
    cost = load_csv("scalability_cost")
    piv = cost[cost["axis"] == "n"].pivot_table("fit_time", "n", "model")
    ratio = piv["ABD-NB"] / piv["ABD-NB (fixed)"]
    c("selection overhead (mean)", ratio.mean(), 1.46, 0.02)
    c("selection overhead at n=500", ratio.loc[500], 1.56, 0.02)
    c("selection overhead at n=32000", ratio.loc[32000], 1.27, 0.02)

    # per-stage costs: what actually tests the complexity equation
    stg = load_csv("scalability_stages")
    sn = stg[stg["axis"] == "n"].sort_values("n")
    sd = stg[stg["axis"] == "d"].sort_values("d")
    c("stage exponent: dependence in n", sn["dependence_exponent"].iloc[0], 1.00, 0.02)
    c("stage exponent: eigen in n", sn["eigendecomposition_exponent"].iloc[0], 0.08, 0.02)
    c("stage exponent: dependence in d", sd["dependence_exponent"].iloc[0], 1.05, 0.02)
    c("stage exponent: eigen in d", sd["eigendecomposition_exponent"].iloc[0], 1.74, 0.02)
    c("eigen at d=5 (ms)", 1000 * sd["eigendecomposition"].iloc[0], 0.02, 0.005)
    c("eigen at d=320 (ms)", 1000 * sd["eigendecomposition"].iloc[-1], 3.87, 0.05)
    share = 100 * sd["eigendecomposition"] / sd["dependence"]
    c("eigen share at d=20 (%)", share.iloc[2], 0.35, 0.02)
    c("eigen share at d=320 (%)", share.iloc[-1], 1.83, 0.02)
    c("d grid max", sd["d"].max(), 320, 0)

    # prediction and memory
    c("predict exponent n (ABD-NB)", sc.loc[("n", "ABD-NB"), "predict_exponent_asym"], 1.19, 0.02)
    c("predict exponent n (GNB)", sc.loc[("n", "GNB"), "predict_exponent_asym"], 0.92, 0.02)
    c("predict exponent d (ABD-NB)", sc.loc[("d", "ABD-NB"), "predict_exponent_asym"], 1.32, 0.02)
    c("predict exponent d (GNB)", sc.loc[("d", "GNB"), "predict_exponent_asym"], 0.77, 0.02)
    c("predict at n=32000 ABD-NB (ms)", sc.loc[("n", "ABD-NB"), "predict_max_ms"], 51, 1.0)
    c("predict at n=32000 GNB (ms)", sc.loc[("n", "GNB"), "predict_max_ms"], 7.8, 0.3)
    pn = cost[cost["axis"] == "n"].pivot_table("predict_time", "n", "model")
    pr = pn["ABD-NB"] / pn["GNB"]
    c("predict ratio at n=500", pr.loc[500], 2.07, 0.05)
    c("predict ratio at n=32000", pr.loc[32000], 6.58, 0.05)
    c("peak MB ABD-NB min (n)", sc.loc[("n", "ABD-NB"), "peak_mb_min"], 0.37, 0.02)
    c("peak MB ABD-NB max (n)", sc.loc[("n", "ABD-NB"), "peak_mb_max"], 18.4, 0.1)
    c("peak MB ABD-NB max (d)", sc.loc[("d", "ABD-NB"), "peak_mb_max"], 20.5, 0.1)
    c("peak MB GNB max (n)", sc.loc[("n", "GNB"), "peak_mb_max"], 5.3, 0.1)

    # redundancy profile: correlations, terciles, fallback share
    rp = load_csv("redundancy_profile")
    c("rho(R, dLL)", stats.spearmanr(rp["redundancy"], rp["d_ll_rel"]).statistic, 0.62, 0.01)
    c("p rho(R, dLL) x1e5", 1e5 * stats.spearmanr(rp["redundancy"], rp["d_ll_rel"]).pvalue, 5.0, 0.5)
    c("pearson(R, dLL)", stats.pearsonr(rp["redundancy"], rp["d_ll_rel"]).statistic, 0.72, 0.01)
    c("rho(R, dacc)", stats.spearmanr(rp["redundancy"], rp["d_acc"]).statistic, 0.51, 0.01)
    c("p rho(R, dacc) x1e3", 1e3 * stats.spearmanr(rp["redundancy"], rp["d_acc"]).pvalue, 1.3, 0.1)
    c("pearson(R, dacc)", stats.pearsonr(rp["redundancy"], rp["d_acc"]).statistic, 0.42, 0.01)
    q = rp["redundancy"].quantile([1 / 3, 2 / 3]).values
    bandof = lambda r: "low" if r <= q[0] else ("medium" if r <= q[1] else "high")
    rp = rp.assign(band=rp["redundancy"].map(bandof))
    for bnd, dacc, dll in (("low", -0.12, 5.8), ("medium", 0.02, 16.6), ("high", 3.09, 52.7)):
        gb = rp[rp["band"] == bnd]
        c(f"band {bnd} mean dacc", gb["d_acc"].mean(), dacc, 0.02)
        c(f"band {bnd} mean dLL", gb["d_ll_rel"].mean(), dll, 0.1)
    c("band high max dacc", rp[rp["band"] == "high"]["d_acc"].max(), 14.00, 0.05)
    c("band high max dLL", rp[rp["band"] == "high"]["d_ll_rel"].max(), 85.2, 0.1)
    c("datasets with R<=0.05", (rp["redundancy"] <= 0.05).sum(), 11, 0)
    c("mean dacc where R<=0.05", rp[rp["redundancy"] <= 0.05]["d_acc"].mean(), -0.08, 0.01)

    # the seven dependence measures (abstract claims the range and the p level)
    rm = load_csv("redundancy_measures")
    meas = ["spearman", "pearson", "kendall", "mi", "cramersv", "dcor", "hsic"]
    rll_m = [stats.spearmanr(rm[k], rm["d_ll_rel"]).statistic for k in meas]
    pll_m = [stats.spearmanr(rm[k], rm["d_ll_rel"]).pvalue for k in meas]
    c("measures: min rho(LL)", min(rll_m), 0.53, 0.01)
    c("measures: max rho(LL)", max(rll_m), 0.68, 0.01)
    c("measures: all p<0.01", float(all(x < 0.01 for x in pll_m)), 1.0, 0)

    # selector frequencies
    gs = load_csv("gamma_selection")
    gr = gs[gs["stratum"] == "real"]
    c("selector folds", len(gs), 1440, 0)
    c("gamma=0 share all (%)", 100 * (gs["gamma"] == 0).mean(), 12.2, 0.1)
    c("gamma=0 share real (%)", 100 * (gr["gamma"] == 0).mean(), 13.0, 0.1)
    c("rescale kept share (%)", 100 * (gs["rescale"] == "meff").mean(), 19.1, 0.1)
    c("class-specific share (%)", 100 * gs["class_specific"].mean(), 29.0, 0.1)

    # win/tie/loss counts at the 0.05-point tolerance
    for name, idx, wtl in (("all", acc.index, (24, 8, 16)), ("real", real, (17, 7, 13))):
        w = (da[idx] > 0.05).sum(); l = (da[idx] < -0.05).sum()
        c(f"{name}: wins", w, wtl[0], 0)
        c(f"{name}: ties", len(idx) - w - l, wtl[1], 0)
        c(f"{name}: losses", l, wtl[2], 0)

    # per-band selector behaviour
    qb = rp["redundancy"].quantile([1 / 3, 2 / 3]).values
    bof = {r.dataset: ("low" if r.redundancy <= qb[0] else
                       "medium" if r.redundancy <= qb[1] else "high")
           for r in rp.itertuples()}
    grb = gr.assign(band=gr["dataset"].map(bof))
    for bnd, deff, g0, resc in (("low", 0.87, 9.5, 11.0), ("medium", 0.69, 11.1, 30.0),
                                ("high", 0.41, 18.6, 21.9)):
        gg = grb[grb["band"] == bnd]
        c(f"band {bnd} deff/d", (gg["d_eff"] / gg["d"]).mean(), deff, 0.005)
        c(f"band {bnd} gamma=0 (%)", 100 * (gg["gamma"] == 0).mean(), g0, 0.1)
        c(f"band {bnd} rescale (%)", 100 * (gg["rescale"] == "meff").mean(), resc, 0.1)
    c("rescale kept real (%)", 100 * (gr["rescale"] == "meff").mean(), 20.7, 0.1)

    # composition of internal correction and post-hoc recalibration
    cp = load_csv("composition")
    cll = cp.pivot_table("log_loss", "dataset", "model")
    cacc = cp.pivot_table("accuracy", "dataset", "model") * 100
    crl = cll.rank(axis=1).mean()
    c("comp median LL GNB", cll["GNB"].median(), 0.944, 0.002)
    c("comp median LL GNB+Platt", cll["GNB+Platt"].median(), 0.466, 0.002)
    c("comp median LL ABD-NB", cll["ABD-NB"].median(), 0.616, 0.002)
    c("comp median LL ABD-NB+Platt", cll["ABD-NB+Platt"].median(), 0.419, 0.002)
    c("comp LL rank ABD-NB+Platt", crl["ABD-NB+Platt"], 2.62, 0.02)
    c("comp LL rank GNB+Platt", crl["GNB+Platt"], 3.15, 0.02)
    c("comp acc ABD-NB+Platt", cacc["ABD-NB+Platt"].mean(), 80.60, 0.02)
    c("comp acc GNB+Platt", cacc["GNB+Platt"].mean(), 80.11, 0.02)
    c("comp acc ABD-NB+Iso", cacc["ABD-NB+Iso"].mean(), 81.92, 0.02)
    c("comp acc GNB+Iso", cacc["GNB+Iso"].mean(), 81.55, 0.02)
    resid = 100 * (cll["GNB+Platt"] - cll["ABD-NB+Platt"]) / cll["GNB+Platt"]
    residi = 100 * (cll["GNB+Iso"] - cll["ABD-NB+Iso"]) / cll["GNB+Iso"]
    c("comp residual median (%)", resid.median(), 0.65, 0.01)
    c("comp residual wins (Platt)", (resid > 0).sum(), 31, 0)
    c("comp residual wins (Iso)", (residi > 0).sum(), 29, 0)

    # operating-conditions sweep (prose values)
    cs = load_csv("conditions_summary").set_index(["factor", "value"])
    for fac, val, dacc in (("redundancy r", "2", 0.41), ("redundancy r", "8", 8.93),
                           ("redundancy r", "32", 17.07), ("dependence rho", "0.2", 0.28),
                           ("dependence rho", "0.95", 7.25), ("sample size n", "50", 2.53),
                           ("sample size n", "200", 6.70), ("dimension d", "7", 6.67),
                           ("dimension d", "201", 2.63), ("class imbalance", "0.5", 5.99),
                           ("class imbalance", "0.02", 1.45),
                           ("dependence form", "True", 4.29),
                           ("distribution shift", "1.5", 18.51),
                           ("independent signal", "0.25", 0.28),
                           ("independent signal", "0.0", -0.53)):
        c(f"cond {fac}={val} dacc", float(cs.loc[(fac, val), "d_acc"]), dacc, 0.02)
    r1 = cs.loc[("redundancy r", "1")]
    c("cond r=1 dacc", float(r1["d_acc"]), 0.01, 0.01)
    c("cond r=1 gamma0 (%)", 100 * float(r1["gamma_zero_frac"]), 20.0, 0.1)
    r0 = cs.loc[("dependence rho", "0.0")]
    c("cond rho=0 dacc", float(r0["d_acc"]), -0.03, 0.01)
    c("cond rho=0 gamma0 (%)", 100 * float(r0["gamma_zero_frac"]), 26.7, 0.1)
    fail = cs.loc[("independent signal", "0.0")]
    c("cond failure CI lo", float(fail["ci_lo"]), -0.92, 0.01)
    c("cond failure CI hi", float(fail["ci_hi"]), -0.20, 0.01)
    c("cond failure dLL", float(fail["d_logloss"]), 0.31, 0.01)
    c("cond failure dECE", float(fail["d_ece"]), 0.105, 0.005)
    c("cond failure gamma med", float(fail["gamma_med"]), 0.5, 0.01)

    # family members and the mis-specification failure
    c("ABD-NB-C acc rank", ra["ABD-NB-C"], 15.50, 0.02)
    c("ABD-NB-C worst real", ((acc["ABD-NB-C"] - acc["GNB"]) * 100)[real].min(), -26.5, 0.1)
    c("ABD-NB-G worst real", ((acc["ABD-NB-G"] - acc["GNB"]) * 100)[real].min(), -10.9, 0.1)
    c("yeast GNB acc", 100 * acc.loc["yeast", "GNB"], 14.5, 0.1)
    c("yeast ABD-NB acc", 100 * acc.loc["yeast", "ABD-NB"], 13.8, 0.1)
    c("yeast GNB-Platt acc", 100 * acc.loc["yeast", "GNB-Platt"], 40.9, 0.1)
    c("yeast GNB-Iso acc", 100 * acc.loc["yeast", "GNB-Iso"], 54.8, 0.1)
    c("digits ABD-NB-C acc", 100 * acc.loc["digits", "ABD-NB-C"], 61.1, 0.1)
    c("digits GNB acc", 100 * acc.loc["digits", "GNB"], 84.1, 0.1)
    c("GNB-Platt LL rank", rl["GNB-Platt"], 10.19, 0.02)
    c("GNB-Iso LL rank", rl["GNB-Iso"], 11.06, 0.02)
    c("MI-WNB LL rank", rl["MI-WNB"], 13.06, 0.02)
    c("CW-NB LL rank", rl["CW-NB"], 13.29, 0.02)

    # duplicate recovery and learning curves
    dup = load_csv("duplicates").set_index("r")
    c("dup w at r=1", dup.loc[1, "w_copy_mean"], 0.899, 0.002)
    c("dup w at r=2", dup.loc[2, "w_copy_mean"], 0.492, 0.002)
    c("dup w at r=10", dup.loc[10, "w_copy_mean"], 0.110, 0.002)
    relerr = (dup["w_copy_mean"] / dup["w_theory"] - 1).abs().max()
    c("dup max relative error", 100 * relerr, 10.1, 0.2)
    c("dup d_eff max deviation", (dup["d_eff"] - dup["d_eff_theory"]).abs().max(), 0.0, 1e-10)
    lcv = load_csv("learning_curve").pivot_table("accuracy", "n", "model") * 100
    c("lc ABD-NB at n=60", lcv.loc[60, "ABD-NB"], 84.49, 0.05)
    c("lc ABD-NB at n=4000", lcv.loc[4000, "ABD-NB"], 85.43, 0.05)
    c("lc GNB at n=4000", lcv.loc[4000, "GNB"], 72.13, 0.05)

    # runtime
    ft = b.groupby("model")[["fit_time", "predict_time"]].mean() * 1000
    c("fit ABD-NB-G (ms)", ft.loc["ABD-NB-G", "fit_time"], 15.4, 0.2)
    c("fit GNB (ms)", ft.loc["GNB", "fit_time"], 1.83, 0.05)
    c("fit ABD-NB (ms)", ft.loc["ABD-NB", "fit_time"], 88.1, 1.0)
    c("fit WANBIA (ms)", ft.loc["WANBIA", "fit_time"], 79.8, 1.0)
    c("fit CFW-NB (ms)", ft.loc["CFW-NB", "fit_time"], 199.5, 2.0)
    c("fit MI-WNB (ms)", ft.loc["MI-WNB", "fit_time"], 139.7, 1.0)
    c("fit RF (ms)", ft.loc["RF", "fit_time"], 500.1, 3.0)
    c("predict ABD-NB (ms)", ft.loc["ABD-NB", "predict_time"], 0.41, 0.02)
    c("predict GNB (ms)", ft.loc["GNB", "predict_time"], 0.36, 0.02)
    c("predict AODE (ms)", ft.loc["AODE", "predict_time"], 35.2, 0.3)
    c("AODE/GNB predict ratio", ft.loc["AODE", "predict_time"] / ft.loc["GNB", "predict_time"],
      97.3, 0.5)
    c("flagship/fixed fit ratio", ft.loc["ABD-NB", "fit_time"] / ft.loc["ABD-NB-G", "fit_time"],
      5.7, 0.1)
    c("TAN/fixed fit ratio", ft.loc["TAN", "fit_time"] / ft.loc["ABD-NB-G", "fit_time"], 2.4, 0.1)

    # drift
    dr = load_csv("drift")
    dab = dr[dr["model"] == "Online ABD-NB"]
    gg = dab.groupby("t")[["acc_window", "w_strong", "w_blockA", "w_blockB"]].mean()
    prew = gg.loc[1000:3900]; postw = gg.loc[6000:]
    c("drift blockA pre", prew["w_blockA"].mean(), 0.23, 0.01)
    c("drift blockA post", postw["w_blockA"].mean(), 1.09, 0.01)
    c("drift blockB pre", prew["w_blockB"].mean(), 1.09, 0.01)
    c("drift blockB post", postw["w_blockB"].mean(), 0.23, 0.01)
    c("drift strong pre", prew["w_strong"].mean(), 0.79, 0.01)
    c("drift strong post", postw["w_strong"].mean(), 0.80, 0.01)
    dgnb = dr[dr["model"] == "Online GNB"].groupby("t")["acc_window"].mean()
    c("drift ABD pre acc", 100 * prew["acc_window"].mean(), 90.5, 0.1)
    c("drift GNB pre acc", 100 * dgnb.loc[1000:3900].mean(), 84.2, 0.1)
    c("drift advantage pre", 100 * (prew["acc_window"].mean() - dgnb.loc[1000:3900].mean()), 6.3, 0.1)
    c("drift ABD post acc", 100 * gg.loc[7000:, "acc_window"].mean(), 90.8, 0.1)
    c("drift GNB post acc", 100 * dgnb.loc[7000:].mean(), 84.5, 0.1)
    c("drift dip (points)", 100 * (prew["acc_window"].mean() - gg.loc[4000:5000, "acc_window"].min()),
      4.5, 0.1)

    # weight-function and measure ablations
    wf = load_csv("sensitivity_weightfn")
    wa = wf.pivot_table("accuracy", "dataset", "weight_fn") * 100
    wl = wf.pivot_table("log_loss", "dataset", "weight_fn")
    for fn, val in (("harmonic", 85.88), ("linear", 85.38), ("exponential", 79.50),
                    ("inverse", 77.62)):
        c(f"weightfn {fn} on synth-conflict", wa.loc["synth-conflict", fn], val, 0.05)
    c("weightfn harmonic on synth-nonlin", wa.loc["synth-nonlin", "harmonic"], 85.00, 0.05)
    c("weightfn inverse on synth-nonlin", wa.loc["synth-nonlin", "inverse"], 86.50, 0.05)
    c("weightfn harmonic LL breast-cancer", wl.loc["breast-cancer", "harmonic"], 0.243, 0.002)
    c("weightfn harmonic LL synth-conflict", wl.loc["synth-conflict", "harmonic"], 0.349, 0.002)
    c("weightfn harmonic LL synth-nonlin", wl.loc["synth-nonlin", "harmonic"], 0.317, 0.002)
    c("weightfn inverse LL synth-nonlin", wl.loc["synth-nonlin", "inverse"], 0.306, 0.002)
    ms = load_csv("sensitivity_measure").pivot_table("accuracy", "dataset", "measure") * 100
    c("measure MI on synth-conflict", ms.loc["synth-conflict", "mi"], 81.00, 0.05)
    c("measure MI dip (points)", ms.loc["synth-conflict"].max() - ms.loc["synth-conflict", "mi"],
      5.0, 0.1)
    c("measure MI on synth-nonlin", ms.loc["synth-nonlin", "mi"], 86.37, 0.05)
    c("measure Spearman on synth-nonlin", ms.loc["synth-nonlin", "spearman"], 85.00, 0.05)

    # controlled sweeps
    sr = load_csv("synthetic_rho")
    sral = sr.pivot_table("log_loss", "rho", "model")
    srae = sr.pivot_table("ece", "rho", "model")
    srac = sr.pivot_table("accuracy", "rho", "model") * 100
    c("rho: GNB LL at 0", sral.loc[0.0, "GNB"], 0.352, 0.002)
    c("rho: GNB LL at 0.95", sral.loc[0.95, "GNB"], 1.967, 0.002)
    c("rho: ABD LL at 0.95", sral.loc[0.95, "ABD-NB"], 1.103, 0.002)
    c("rho: GNB ECE at 0", srae.loc[0.0, "GNB"], 0.061, 0.002)
    c("rho: GNB ECE at 0.95", srae.loc[0.95, "GNB"], 0.317, 0.002)
    c("rho: ABD ECE at 0.95", srae.loc[0.95, "ABD-NB"], 0.190, 0.002)
    c("rho: LR acc at 0.95", srac.loc[0.95, "LR"], 61.2, 0.1)
    c("rho: ABD acc at 0.95", srac.loc[0.95, "ABD-NB"], 62.0, 0.1)
    sc2 = load_csv("synthetic_conflict")
    sca = sc2.pivot_table("accuracy", "r", "model") * 100
    scl = sc2.pivot_table("log_loss", "r", "model")
    sce = sc2.pivot_table("ece", "r", "model")
    c("conflict: GNB acc at r=1", sca.loc[1, "GNB"], 86.4, 0.1)
    c("conflict: GNB acc at r=24", sca.loc[24, "GNB"], 69.2, 0.1)
    c("conflict: GNB LL at r=24", scl.loc[24, "GNB"], 3.309, 0.002)
    c("conflict: GNB ECE at r=24", sce.loc[24, "GNB"], 0.291, 0.002)
    c("conflict: ABD acc at r=24", sca.loc[24, "ABD-NB"], 86.5, 0.1)
    c("conflict: ABD LL at r=24", scl.loc[24, "ABD-NB"], 0.414, 0.002)
    c("conflict: ABD ECE at r=24", sce.loc[24, "ABD-NB"], 0.088, 0.002)

    # duplication stress on real data
    du = load_csv("redundancy").pivot_table("accuracy", ["dataset", "r"], "model") * 100
    for ds, m, r_, val in (("wine", "GNB", 0, 97.2), ("wine", "GNB", 32, 80.9),
                           ("wine", "ABD-NB", 0, 96.6), ("wine", "ABD-NB", 32, 97.0),
                           ("phoneme", "GNB", 0, 76.5), ("phoneme", "GNB", 1, 73.2),
                           ("phoneme", "GNB", 32, 66.7), ("phoneme", "ABD-NB", 0, 77.7),
                           ("phoneme", "ABD-NB", 1, 77.9), ("phoneme", "ABD-NB", 32, 77.7),
                           ("credit-ger", "GNB", 0, 73.6), ("credit-ger", "GNB", 32, 64.2),
                           ("credit-ger", "ABD-NB", 32, 73.7),
                           ("iris", "GNB", 0, 96.0), ("iris", "GNB", 32, 88.4),
                           ("iris", "ABD-NB", 32, 95.3),
                           ("breast-cancer", "ABD-NB", 0, 94.7),
                           ("breast-cancer", "ABD-NB", 32, 94.3)):
        c(f"dup {ds} {m} r={r_}", du.loc[(ds, r_), m], val, 0.06)
    dseg = du.loc[("segmentation", 32)] - du.loc[("segmentation", 0)]
    c("dup segmentation ABD-NB loss", dseg["ABD-NB"], -6.9, 0.06)
    c("dup segmentation GNB loss", dseg["GNB"], -10.7, 0.06)
    dyeast = du.loc[("yeast", 32)] - du.loc[("yeast", 0)]
    c("dup yeast ABD-NB gain", dyeast["ABD-NB"], 5.7, 0.06)
    c("dup yeast GNB gain", dyeast["GNB"], 5.1, 0.06)
    flat = 0
    for ds in du.index.get_level_values(0).unique():
        if abs(du.loc[(ds, 32), "ABD-NB"] - du.loc[(ds, 0), "ABD-NB"]) < 2.0:
            flat += 1
    c("dup datasets flat for ABD-NB", flat, 8, 0)

    # three-way Gaussian design
    hr = load_csv("higher_order").groupby("design").mean(numeric_only=True)
    c("3way max pairwise rho", hr.loc["3-way Gaussian", "max_pairwise_spearman"], 0.732, 0.002)
    c("3way mean pairwise rho", hr.loc["3-way Gaussian", "mean_pairwise_spearman"], 0.021, 0.002)
    c("3way deff ratio", hr.loc["3-way Gaussian", "d_eff_ratio"], 1.00, 0.005)
    c("3way dacc", 100 * (hr.loc["3-way Gaussian", "acc_abd"] - hr.loc["3-way Gaussian", "acc_nb"]),
      0.45, 0.02)
    c("3way dLL rel (%)",
      100 * (1 - hr.loc["3-way Gaussian", "ll_abd"] / hr.loc["3-way Gaussian", "ll_nb"]), 7.1, 0.1)
    c("3way AODE acc", 100 * hr.loc["3-way Gaussian", "acc_aode"], 90.1, 0.1)
    c("3way RF acc", 100 * hr.loc["3-way Gaussian", "acc_rf"], 100.0, 0.1)

    # ablation significance (quoted in the abstract, cover letter and response)
    ae, ag = "(e) W+sel", "(g) W+scale$M$+sel"
    ab2, ad2 = "(b) W", "(d) W+scale$M$"
    c("abl e-g acc wilcoxon p", stats.wilcoxon(a2[ae], a2[ag]).pvalue, 0.74, 0.01)
    c("abl e-g acc wins", (a2[ae] > a2[ag]).sum(), 25, 0)
    c("abl e-g acc losses", (a2[ae] < a2[ag]).sum(), 18, 0)
    c("abl e-g LL median (%)", (100 * (1 - l2[ae] / l2[ag])).median(), 8.5, 0.05)
    c("abl e-g LL p x1e7", 1e7 * stats.wilcoxon(l2[ae], l2[ag]).pvalue, 5.0, 0.1)
    c("abl b-d LL median (%)", (100 * (1 - l2[ab2] / l2[ad2])).median(), 8.0, 0.05)
    c("abl b-d LL p x1e13", 1e13 * stats.wilcoxon(l2[ab2], l2[ad2]).pvalue, 7.8, 0.2)

    # degeneracy diagnosis behind the full-size failures
    dg = load_csv("degeneracy").set_index("dataset")
    for ds, d_, binary, const, medlev in (("kr-vs-kp", 36, 35, 3, 2),
                                          ("ann-thyroid", 21, 15, 5, 2),
                                          ("nursery", 8, 1, 1, 3),
                                          ("hypothyroid", 25, 17, 6, 2)):
        c(f"deg {ds} d", dg.loc[ds, "d"], d_, 0)
        c(f"deg {ds} binary", dg.loc[ds, "binary"], binary, 0)
        c(f"deg {ds} constant-in-a-class", dg.loc[ds, "features_constant_in_some_class"],
          const, 0)
        c(f"deg {ds} median levels", dg.loc[ds, "median_levels"], medlev, 0)
    for ds in ("letter", "texture-full"):
        c(f"deg {ds} binary", dg.loc[ds, "binary"], 0, 0)
        c(f"deg {ds} constant-in-a-class", dg.loc[ds, "features_constant_in_some_class"], 0, 0)

    # matched tuning protocol
    tb = load_csv("tuned_baselines")
    ta = tb.pivot_table("accuracy", "dataset", ["model", "protocol"]) * 100
    tl = tb.pivot_table("log_loss", "dataset", ["model", "protocol"])
    abd_t, gnb_t = ta[("ABD-NB", "tuned")], ta[("GNB", "tuned")]
    abd_d, gnb_d = ta[("ABD-NB", "default")], ta[("GNB", "default")]
    c("tuned settings", tb["dataset"].nunique(), 12, 0)
    c("tuned models", tb["model"].nunique(), 10, 0)
    c("tuned acc wins over GNB", (abd_t > gnb_t).sum(), 11, 0)
    c("tuned acc mean gain over GNB", (abd_t - gnb_t).mean(), 4.94, 0.02)
    c("tuned acc median gain over GNB", (abd_t - gnb_t).median(), 2.81, 0.02)
    c("tuned acc wilcoxon p x1e3", 1e3 * stats.wilcoxon(abd_t, gnb_t).pvalue, 1.5, 0.1)
    c("tuned LL wins over GNB", (tl[("ABD-NB", "tuned")] < tl[("GNB", "tuned")]).sum(), 12, 0)
    c("tuned LL median rel (%)",
      (100 * (1 - tl[("ABD-NB", "tuned")] / tl[("GNB", "tuned")])).median(), 62.7, 0.1)
    c("tuned LL wilcoxon p x1e4",
      1e4 * stats.wilcoxon(tl[("ABD-NB", "tuned")], tl[("GNB", "tuned")]).pvalue, 4.9, 0.2)
    c("tuning gain ABD-NB", (abd_t - abd_d).mean(), 6.18, 0.02)
    c("tuning gain GNB", (gnb_t - gnb_d).mean(), 4.55, 0.02)
    m_noh = abd_t.index != "hypothyroid"
    c("tuning gain ABD-NB w/o hypothyroid", (abd_t - abd_d)[m_noh].mean(), 0.62, 0.02)
    c("tuned lead over GNB w/o hypothyroid", (abd_t - gnb_t)[m_noh].mean(), 3.87, 0.02)
    c("hypothyroid ABD default", abd_d["hypothyroid"], 27.9, 0.06)
    c("hypothyroid ABD tuned", abd_t["hypothyroid"], 95.2, 0.06)
    c("hypothyroid GNB default", gnb_d["hypothyroid"], 24.2, 0.06)
    c("hypothyroid GNB tuned", gnb_t["hypothyroid"], 78.6, 0.06)
    ts = load_csv("tuned_baselines_summary")
    ts = ts[ts["protocol"] == "tuned"].set_index("model")
    for m, ar, lr_ in (("ABD-NB", 7.62, 7.42), ("GNB", 9.67, 9.75), ("RF", 3.46, 3.17),
                       ("SVM", 3.79, 3.00), ("XGB", 3.79, 3.25), ("LGBM", 3.96, 4.08),
                       ("LR", 4.71, 4.50), ("AODE", 5.42, 6.33), ("kNN", 5.54, 7.42),
                       ("WANBIA", 7.04, 6.08)):
        c(f"tuned acc rank {m}", ts.loc[m, "acc_rank"], ar, 0.02)
        c(f"tuned LL rank {m}", ts.loc[m, "ll_rank"], lr_, 0.02)
    c("tuned mean acc ABD-NB", ts.loc["ABD-NB", "acc_mean"], 81.22, 0.02)
    c("tuned mean acc WANBIA", ts.loc["WANBIA", "acc_mean"], 82.93, 0.02)
    c("tuned fit ABD-NB (s)", ts.loc["ABD-NB", "fit_time"], 31.9, 0.2)
    c("tuned fit RF (s)", ts.loc["RF", "fit_time"], 16.5, 0.2)
    c("tuned fit LGBM (s)", ts.loc["LGBM", "fit_time"], 8.9, 0.2)
    c("tuned vehicle ABD-NB", abd_t["vehicle"], 53.7, 0.06)
    c("tuned texture ABD-NB", abd_t["texture"], 78.7, 0.06)
    c("tuned vehicle SVM", ta[("SVM", "tuned")]["vehicle"], 81.6, 0.06)
    c("tuned texture SVM", ta[("SVM", "tuned")]["texture"], 99.1, 0.06)

    # claims about the model itself (weight range, main-benchmark ranks quoted in prose)
    c("main acc rank AODE", ra["AODE"], 8.90, 0.02)
    c("main acc rank WANBIA", ra["WANBIA"], 9.70, 0.02)
    c("main acc rank HNB (prose)", ra["HNB"], 10.02, 0.02)
    c("main acc rank RF", ra["RF"], 5.53, 0.02)
    c("main acc rank SVM", ra["SVM"], 6.20, 0.02)

    # the fitted exponents are positive but not bounded by one (Section: our proposal)
    wr = load_csv("weight_range").set_index(["rescale", "class_specific"])
    c("raw global max weight", wr.loc[("none", False), "wmax"], 1.00, 0.005)
    c("rescaled global max weight", wr.loc[("meff", False), "wmax"], 1.91, 0.01)
    c("rescaled class-specific max weight", wr.loc[("meff", True), "wmax"], 3.11, 0.01)
    c("raw class-specific max weight", wr.loc[("none", True), "wmax"], 2.07, 0.01)

    # closed-form overconfidence factors in the equicorrelated model (Proposition 3
    # and the remark that follows it) -- checked against the formulas, not a CSV
    def _meff_equi(dd, rho):
        lam = np.concatenate([[1 + (dd - 1) * rho], np.full(dd - 1, 1 - rho)])
        f = (lam >= 1).astype(float) + (lam - np.floor(lam))
        return float(np.clip(f.sum(), 1, dd))

    dd = 12
    grid = np.linspace(1e-6, 1 - 1e-6, 200001)
    h_resc = np.array([_meff_equi(dd, r) / dd * (1 + (dd - 1) * r) for r in grid])
    h_raw = (1 + (dd - 1) * grid) / (1 + (dd - 1) * grid ** 2)
    c("rescaled worst case at d=12", h_resc.max(), (dd + 2) ** 2 / (4 * dd), 0.01)
    c("rescaled worst case value", h_resc.max(), 4.08, 0.01)
    c("rescaled limit as rho->1", h_resc[-1], 2.0, 0.01)
    c("raw worst case at d=12", h_raw.max(), 0.5 * (1 + np.sqrt(dd)), 0.01)
    c("raw worst case value", h_raw.max(), 2.23, 0.01)
    c("raw argmax rho", grid[h_raw.argmax()], 1 / (1 + np.sqrt(dd)), 0.001)
    c("raw limit as rho->1", h_raw[-1], 1.0, 0.001)
    c("NB worst case at d=12", 1 + (dd - 1) * grid[-1], 12.0, 0.01)

    # Digits: the selector disengages on nearly every fold but not quite all of them
    gd = gs[gs["dataset"] == "digits"]
    c("digits folds", len(gd), 30, 0)
    c("digits gamma=0 folds", (gd["gamma"] == 0).sum(), 29, 0)
    c("digits ABD-GNB acc", da["digits"], -0.06, 0.01)
    c("digits MI-WNB gain", ((acc["MI-WNB"] - acc["GNB"]) * 100)["digits"], 4.75, 0.02)

    # categorical section: ranks, exact fallbacks and dataset metadata
    _cat = load_csv("categorical")
    ca = _cat.pivot_table("accuracy", "dataset", "model") * 100
    cl = _cat.pivot_table("log_loss", "dataset", "model")
    crl = cl.rank(axis=1).mean()
    c("cat LL rank AODE", crl["AODE"], 1.67, 0.02)
    c("cat LL rank TAN", crl["TAN"], 2.89, 0.02)
    c("cat LL rank Cat-ABD-G", crl["Cat-ABD-G"], 3.89, 0.02)
    for ds in ("splice-dna", "nursery"):
        c(f"cat {ds} exact fallback", abs(ca.loc[ds, "Cat-ABD"] - ca.loc[ds, "Cat-NB"]),
          0.0, 0.001)
    c("cat mushroom AODE", ca.loc["mushroom", "AODE"], 99.95, 0.02)
    c("cat mushroom TAN", ca.loc["mushroom", "TAN"], 99.90, 0.02)
    c("cat mushroom Cat-ABD", ca.loc["mushroom", "Cat-ABD"], 97.82, 0.02)
    _rp = load_csv("redundancy_profile")
    meta = _rp.set_index("dataset")
    c("soybean classes", meta.loc["soybean", "K"], 18, 0)
    c("splice-dna features", meta.loc["splice-dna", "d"], 60, 0)
    c("suite n min", meta["n"].min(), 150, 0)
    c("suite n max", meta["n"].max(), 4004, 0)
    c("suite d min", meta["d"].min(), 4, 0)
    c("suite d max", meta["d"].max(), 64, 0)
    c("suite K max", meta["K"].max(), 18, 0)

    # --- arithmetic identities the manuscript states in several places ---
    # 37 real + 2 augmented + 9 synthetic = 48 settings
    strata = pd.Series({d: dataset_stratum(d) for d in acc.index}).value_counts()
    c("settings: real", strata["real"], 37, 0)
    c("settings: augmented", strata["augmented"], 2, 0)
    c("settings: synthetic", strata["synthetic"], 9, 0)
    c("settings: they sum to 48", strata.sum(), 48, 0)
    # 9 + 4 + 2 + 7 = 22 classifiers, as Section 6.2 groups them
    families = {
        "weighted NB": ["GNB", "ABD-NB", "ABD-NB-C", "ABD-NB-G", "ABD-NB-L",
                        "MI-WNB", "CW-NB", "CFW-NB", "WANBIA"],
        "semi-naive": ["TAN", "KDB", "AODE", "HNB"],
        "calibrated": ["GNB-Platt", "GNB-Iso"],
        "discriminative": ["LR", "kNN", "CART", "RF", "XGB", "LGBM", "SVM"],
    }
    for fam, members in families.items():
        c(f"family size: {fam}", len(members), len(members), 0)
        for m in members:
            c(f"{fam}: {m} is in the benchmark", float(m in acc.columns), 1.0, 0)
    c("families sum to the model count", sum(len(v) for v in families.values()), 22, 0)
    c("no model outside the four families",
      len(set(acc.columns) - {m for v in families.values() for m in v}), 0, 0)
    # 48 x 22 x 30 = 31,680 fits
    c("fits = settings x models x folds", 48 * 22 * 30, 31680, 0)
    c("fits recorded", len(b), 31680, 0)
    # the Wilcoxon family is every model but the flagship
    wil = load_csv("stats_wilcoxon")
    c("Wilcoxon opponents", len(wil), 21, 0)
    c("opponents = models - 1", acc.shape[1] - 1, 21, 0)
    c("Wilcoxon covers HNB", float("HNB" in set(wil["opponent"])), 1.0, 0)
    c("Wilcoxon covers CFW-NB", float("CFW-NB" in set(wil["opponent"])), 1.0, 0)
    # Holm must be monotone along the sorted p-values
    for f in ("stats_wilcoxon", "stats_wilcoxon_logloss"):
        h = load_csv(f).sort_values("p_raw")["p_holm"].to_numpy()
        c(f"{f}: Holm is monotone", float((np.diff(h) >= -1e-12).all()), 1.0, 0)

    # the two opponents the table used to omit, now quoted in Section 7.1
    wa = load_csv("stats_wilcoxon").set_index("opponent")
    wll = load_csv("stats_wilcoxon_logloss").set_index("opponent")
    c("CFW-NB acc wins", wa.loc["CFW-NB", "wins"], 24, 0)
    c("CFW-NB acc losses", wa.loc["CFW-NB", "losses"], 24, 0)
    c("CFW-NB acc holm", wa.loc["CFW-NB", "p_holm"], 1.00, 0.005)
    c("CFW-NB LL wins", wll.loc["CFW-NB", "wins"], 24, 0)
    c("CFW-NB LL losses", wll.loc["CFW-NB", "losses"], 24, 0)
    c("CFW-NB LL holm", wll.loc["CFW-NB", "p_holm"], 1.00, 0.005)
    c("HNB acc losses", wa.loc["HNB", "losses"], 32, 0)
    c("HNB acc holm x1e4", 1e4 * wa.loc["HNB", "p_holm"], 7.0, 1.0)
    c("HNB LL losses", wll.loc["HNB", "losses"], 35, 0)
    c("HNB LL holm below 1e-4", float(wll.loc["HNB", "p_holm"] < 1e-4), 1.0, 0)

    fr = load_csv("finite_sample_regret_summary").set_index("n")
    for nn, rm in ((60, 0.95), (250, 0.32), (500, 0.16), (1000, 0.04), (2000, 0.08)):
        c(f"regret mean at n={nn}", 100 * fr.loc[nn, "regret_mean"], rm, 0.02)
    for nn, rq in ((60, 3.08), (250, 1.00), (2000, 0.29)):
        c(f"regret q95 at n={nn}", 100 * fr.loc[nn, "regret_q95"], rq, 0.02)
    for nn, bd in ((60, 40.0), (250, 20.1), (2000, 7.8)):
        c(f"regret bound at n={nn}", 100 * fr.loc[nn, "bound"], bd, 0.1)
    vsnb = 100 * (fr["selected"] - fr["vs_nb"])
    c("selected vs NB min", vsnb.min(), 12.9, 0.05)
    c("selected vs NB max", vsnb.max(), 14.0, 0.05)

    st = load_csv("redundancy_stability")
    c("boot CV median", st["boot_cv"].median(), 0.029, 0.001)
    c("split-half median", st["splithalf_absdiff"].median(), 0.008, 0.001)
    for nn, bias, cnt in ((100, -0.0316, 37), (200, -0.0129, 32), (400, -0.0200, 25),
                          (800, -0.0001, 19), (1600, 0.0, 15)):
        col = f"bias_n{nn}"
        c(f"bias at n={nn}", st[col].median(), bias, 0.0005)
        c(f"datasets at n={nn}", st[col].notna().sum(), cnt, 0)

    # --- full-size evaluation (Section: scalability at full size) ---
    sf = load_csv("scalability_full")
    sa = sf.pivot_table("accuracy", "dataset", "model") * 100
    sl = sf.pivot_table("log_loss", "dataset", "model")
    se = sf.pivot_table("ece", "dataset", "model") * 100
    st = sf.pivot_table("fit_time", "dataset", "model")
    sda = sa["ABD-NB"] - sa["GNB"]
    sdl = 100 * (1 - sl["ABD-NB"] / sl["GNB"])
    c("full-size datasets", sf["dataset"].nunique(), 18, 0)
    c("full-size max n", sf["n"].max(), 20000, 0)
    c("full-size max d", sf["d"].max(), 85, 0)
    c("full LL wins", (sl["ABD-NB"] < sl["GNB"] - 1e-12).sum(), 16, 0)
    c("full LL mean rel", sdl.mean(), 30.8, 0.1)
    c("full LL median rel", sdl.median(), 19.1, 0.1)
    c("full LL wilcoxon p (1e-3)", 1e3 * stats.wilcoxon(sl["ABD-NB"], sl["GNB"]).pvalue,
      1.1, 0.1)
    c("full acc mean gain", sda.mean(), 0.99, 0.02)
    c("full acc median gain", sda.median(), 0.0, 1e-6)
    c("full acc wins", (sda > 1e-9).sum(), 8, 0)
    c("full acc ties", (sda.abs() < 1e-9).sum(), 4, 0)
    c("full acc worst", sda.min(), -0.59, 0.02)
    c("full acc wilcoxon p", stats.wilcoxon(sda[sda.abs() > 1e-12]).pvalue, 0.135, 0.005)
    c("full ECE ABD-NB", se["ABD-NB"].mean(), 16.8, 0.1)
    c("full ECE GNB", se["GNB"].mean(), 22.4, 0.1)
    c("full ECE AODE", se["AODE"].mean(), 3.8, 0.1)
    c("full ECE LGBM", se["LGBM"].mean(), 1.6, 0.1)
    sra = sa.rank(axis=1, ascending=False).mean()
    srl = sl.rank(axis=1).mean()
    c("full acc rank ABD-NB", sra["ABD-NB"], 3.22, 0.02)
    c("full acc rank AODE", sra["AODE"], 2.17, 0.02)
    c("full acc rank GNB", sra["GNB"], 3.33, 0.02)
    c("full acc rank LGBM", sra["LGBM"], 1.28, 0.02)
    c("full LL rank LGBM", srl["LGBM"], 1.28, 0.02)
    c("full fit ABD-NB mean", st["ABD-NB"].mean(), 0.43, 0.02)
    c("full fit GNB mean", st["GNB"].mean(), 0.011, 0.002)
    c("full fit median ratio to GNB", (st["ABD-NB"] / st["GNB"]).median(), 39, 1.0)
    c("full fit median ratio to LGBM", (st["ABD-NB"] / st["LGBM"]).median(), 0.60, 0.02)
    c("letter fit ABD-NB", st.loc["letter", "ABD-NB"], 1.58, 0.05)
    c("letter fit LGBM", st.loc["letter", "LGBM"], 8.5, 0.2)
    for ds, a_abd, a_gnb, a_aode in (("ann-thyroid", 11.3, 11.5, 94.0),
                                     ("kr-vs-kp", 62.9, 63.0, 91.2),
                                     ("nursery-full", 64.5, 64.7, 92.6)):
        c(f"full {ds} ABD-NB acc", sa.loc[ds, "ABD-NB"], a_abd, 0.06)
        c(f"full {ds} GNB acc", sa.loc[ds, "GNB"], a_gnb, 0.06)
        c(f"full {ds} AODE acc", sa.loc[ds, "AODE"], a_aode, 0.06)

    # --- multinomial instantiation ---
    ca = load_csv("categorical").pivot_table("accuracy", "dataset", "model") * 100
    cl = load_csv("categorical").pivot_table("log_loss", "dataset", "model")
    c("cat mean dacc over Cat-NB", (ca["Cat-ABD"] - ca["Cat-NB"]).mean(), 1.23, 0.02)
    c("cat nursery Cat-ABD acc", ca.loc["nursery", "Cat-ABD"], 89.5, 0.06)
    c("cat LL rank Cat-ABD", cl.rank(axis=1).mean()["Cat-ABD"], 2.22, 0.02)
    c("cat LL rank Cat-NB", cl.rank(axis=1).mean()["Cat-NB"], 4.33, 0.02)
    c("cat acc rank AODE", ca.rank(axis=1, ascending=False).mean()["AODE"], 1.56, 0.02)

    df = pd.DataFrame(CHECKS, columns=["claim", "value", "paper", "status"])
    bad = df[df["status"] == "MISMATCH"]
    if len(bad):
        print(df.to_string(index=False))
        print("\nMISMATCHES:\n" + bad.to_string(index=False))
        sys.exit(1)
    print(f"ALL {len(df)} CLAIMS VERIFIED")


if __name__ == "__main__":
    main()
