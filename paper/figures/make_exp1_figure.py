#!/usr/bin/env python3
"""
Experiment 1 figures.

Compares amortized, unconditional-CEM, and per-threshold proposal learning as
the number of requested thresholds increases under a fixed simulation budget.

Produces the main ILE/coverage figure and appendix failure-probability curves.
All plotted metrics are read from stored experiment results; no simulation is run.

Usage:
    python3 paper/figures/make_exp1_figure.py \
        [--results-dir DIR] [--main-out DIR] [--appendix-out DIR]
"""

import argparse
import json
import os
import sys
from pathlib import Path
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configuration

KS = [10, 25, 50, 100]
CEM = "unconditional_cem"
ARMS = ["amortized", CEM, "perthreshold"]
CGV_ARMS = ["amortized", "perthreshold"]
N_SEEDS = 10

ARM_LABEL = {"amortized": "Amortized",
             CEM: "Unconditional CEM",
             "perthreshold": "Per-threshold"}

ARM_COLOR = {"amortized": "#4C72B0", CEM: "#55A868", "perthreshold": "#C44E52"}
TRUTH_COLOR = "#000000"

CEM_ROOT = os.environ.get(
    "CEM_BASELINE_DIR",
    str(Path(__file__).resolve().parents[1] / "results" / "exp1" / "cem_baseline"))
CEM_METRICS = os.path.join(CEM_ROOT, "metrics.json")
CEM_RAW     = os.path.join(CEM_ROOT, "raw")

EXPECTED_ILE = {
    ("amortized", 10): 0.014638, ("amortized", 25): 0.015059,
    ("amortized", 50): 0.015205, ("amortized", 100): 0.014879,
    ("perthreshold", 10): 0.462849, ("perthreshold", 25): 0.538495,
    ("perthreshold", 50): 0.676778, ("perthreshold", 100): 1.20003,
    # from cem_baseline/metrics.json
    (CEM, 10): 0.017743, (CEM, 25): 0.018427,
    (CEM, 50): 0.019041, (CEM, 100): 0.019045,
}
EXPECTED_COV = {
    ("amortized", 10): 1.0, ("amortized", 25): 1.0,
    ("amortized", 50): 1.0, ("amortized", 100): 1.0,
    ("perthreshold", 10): 1.0, ("perthreshold", 25): 1.0,
    ("perthreshold", 50): 0.96, ("perthreshold", 100): 0.555,
    (CEM, 10): 1.0, (CEM, 25): 1.0, (CEM, 50): 1.0, (CEM, 100): 1.0,
}

CEM_TABLE_4DP = {10: "0.0177", 25: "0.0184", 50: "0.0190", 100: "0.0190"}

def default_results_dir():
    """Released layout: <repo>/paper/results/exp1. Resolved from this file, not the CWD."""
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.get("EXP1_RESULTS_DIR")
    if env:
        return env
    cand = repo_root / "results" / "exp1"
    if cand.is_dir():
        return str(cand)
    sys.exit("could not locate results/exp1; pass --results-dir or set EXP1_RESULTS_DIR")

def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "axes.facecolor": "white",
        "axes.grid": False,

        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,

        "axes.edgecolor": "black",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,

        "lines.solid_capstyle": "round",
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "legend.edgecolor": "0.3",
        "legend.borderpad": 0.45,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.6,
        "legend.labelspacing": 0.35,

        # True vector output; text stays real text (editable), not outlines.
        "pdf.fonttype": 42,      # TrueType subset -> selectable/editable glyphs
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # keep <text> elements in the SVG
        "pdf.compression": 6,
        "figure.dpi": 150,
    })

# Data loading

def load_metrics(results_dir):
    """Per-cell ILE / coverage from the endpoint-snapped metrics.json."""
    with open(os.path.join(results_dir, "metrics.json")) as f:
        mr = json.load(f)
    if not mr.get("endpoint_snapped", False):
        sys.exit("metrics.json is not endpoint-snapped; refusing to plot")
    out = {}
    for arm in CGV_ARMS:
        for K in KS:
            c = mr["cells"][f"{arm}_K{K}"]
            out[(arm, K)] = {
                "ile_median": c["ile_primary_median"],
                "ile_per_seed": np.asarray(c["ile_primary_per_seed"], float),
                "cov_median": c["coverage_median"],
                "cov_per_seed": np.asarray(c["coverage_per_seed"], float),
                "cov_min": c["coverage_min"],
                "cov_max": c["coverage_max"],
                "floor": c["floor_primary"],
                "n_seeds": c["n_seeds"],
                "seeds": c["seeds"],
            }
    out.update(load_metrics_cem())
    return mr, out


def load_metrics_cem():
    """
    Per-cell ILE / coverage for the unconditional-CEM arm, from
    cem_baseline/metrics.json.
    """
    with open(CEM_METRICS) as f:
        cm = json.load(f)
    out = {}
    for K in KS:
        c = cm["cells"][f"{CEM}_K{K}"]
        seeds = list(c["seeds"])
        order = np.argsort(seeds)              # bank in ascending seed order
        ile = np.asarray(c["ile"], float)[order]
        cov = np.asarray(c["coverage"], float)[order]
        floors = set(c["floor"])
        assert len(floors) == 1, f"non-uniform 1/B floor at m={K}: {floors}"
        # the floor is never invoked for any CEM cell -- asserted
        assert list(c["n_pos"]) == [c["n_grid"]] * len(seeds), \
            f"CEM coverage is not complete at m={K}; the 1/B floor would bind"
        out[(CEM, K)] = {
            "ile_median": float(np.median(ile)),
            "ile_per_seed": ile,
            "cov_median": float(np.median(cov)),
            "cov_per_seed": cov,
            "cov_min": float(cov.min()),
            "cov_max": float(cov.max()),
            "floor": floors.pop(),
            "n_seeds": len(seeds),
            "seeds": [seeds[i] for i in order],
        }
    return out


def load_curves(results_dir):
    """
    Per-gamma estimates and analytic truth, per arm / K / seed, from raw/seed*.json.
    """
    curves = {}
    for arm in CGV_ARMS:
        for K in KS:
            gam = truth = None
            est = np.full((N_SEEDS, K), np.nan)
            for si, seed in enumerate(range(1, N_SEEDS + 1)):
                with open(os.path.join(results_dir, "raw", f"seed{seed}.json")) as f:
                    d = json.load(f)
                pts = d[arm][str(K)]["per_threshold"]
                g = np.array([p["gamma"] for p in pts], float)
                t = np.array([p["truth"] for p in pts], float)
                e = np.array([np.nan if p["estimate"] is None else p["estimate"]
                              for p in pts], float)
                order = np.argsort(g)
                if gam is None:
                    gam, truth = g[order], t[order]
                est[si] = e[order]
            curves[(arm, K)] = {"gamma": gam, "truth": truth, "est": est}
    curves.update(load_curves_cem(curves))
    return curves


def load_curves_cem(cgv_curves):
    """
    Per-gamma CEM estimates from cem_baseline/raw/seed*.json.
    """
    SNAP_TOL = 1e-10
    out = {}
    for K in KS:
        gam_ref = cgv_curves[("amortized", K)]["gamma"]
        est = np.full((N_SEEDS, K), np.nan)
        for si, seed in enumerate(range(1, N_SEEDS + 1)):
            with open(os.path.join(CEM_RAW, f"seed{seed}.json")) as f:
                d = json.load(f)
            pts = d[CEM][str(K)]["per_threshold"]
            g = np.array([p["gamma"] for p in pts], float)
            e = np.array([np.nan if p["estimate"] is None else p["estimate"]
                          for p in pts], float)
            order = np.argsort(g)
            g, e = g[order], e[order]
            dev = float(np.max(np.abs(g - gam_ref)))
            assert dev <= SNAP_TOL, (
                f"CEM grid differs from the conditioned grid at m={K}, "
                f"seed {seed}: max |dgamma| = {dev:.3e} > {SNAP_TOL:g}")
            est[si] = e
        out[(CEM, K)] = {"gamma": gam_ref,
                         "truth": cgv_curves[("amortized", K)]["truth"],
                         "est": est}
    return out

# (a) ILE vs K, (b) coverage vs K
# --------------------------------------------------------------------------

DODGE = {"amortized": 1.0 / 1.025, CEM: 1.0, "perthreshold": 1.025}

def jitter_log(K, n, arm, spread=0.040):
    """Deterministic multiplicative jitter for seed dots on a log x-axis."""
    x = float(K) * DODGE[arm]
    if n == 1:
        return np.array([x])
    return x * np.exp(spread * np.linspace(-1.0, 1.0, n))

def _style_k_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(8.4, 120)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(r"number of requested thresholds $m$")

def make_main_figure(canon, out_dir):
    set_style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.5, 2.75))
    fig.subplots_adjust(left=0.090, right=0.985, bottom=0.180, top=0.940, wspace=0.29)

    x = np.array(KS, float)

    for arm in ARMS:
        col = ARM_COLOR[arm]
        xd = x * DODGE[arm]

        # panel (a): integrated log-error, ILE@1/B (metrics.json)
        med = np.array([canon[(arm, K)]["ile_median"] for K in KS])
        lo = np.array([np.percentile(canon[(arm, K)]["ile_per_seed"], 25) for K in KS])
        hi = np.array([np.percentile(canon[(arm, K)]["ile_per_seed"], 75) for K in KS])
        axa.fill_between(xd, lo, hi, color=col, alpha=0.15, linewidth=0, zorder=1)
        axa.plot(xd, med, "-", color=col, lw=1.5, zorder=4,
                 marker="o", ms=3.6, mfc="white", mew=1.1, mec=col,
                 label=ARM_LABEL[arm])
        for K in KS:
            v = canon[(arm, K)]["ile_per_seed"]
            axa.plot(jitter_log(K, len(v), arm), v, linestyle="none",
                     marker="o", ms=2.0, color=col, alpha=0.32, mew=0, zorder=3)

        # panel (b): coverage
        med = np.array([canon[(arm, K)]["cov_median"] for K in KS]) * 100.0
        lo = np.array([np.percentile(canon[(arm, K)]["cov_per_seed"], 25) for K in KS]) * 100.0
        hi = np.array([np.percentile(canon[(arm, K)]["cov_per_seed"], 75) for K in KS]) * 100.0
        axb.fill_between(xd, lo, hi, color=col, alpha=0.15, linewidth=0, zorder=1)
        axb.plot(xd, med, "-", color=col, lw=1.5, zorder=4,
                 marker="o", ms=3.6, mfc="white", mew=1.1, mec=col)
        for K in KS:
            v = canon[(arm, K)]["cov_per_seed"] * 100.0
            axb.plot(jitter_log(K, len(v), arm), v, linestyle="none",
                     marker="o", ms=2.0, color=col, alpha=0.32, mew=0, zorder=3)

    _style_k_axis(axa)
    axa.set_yscale("log")
    axa.set_ylabel("integrated log-error (ILE)")
    # the gap between the two arms is widest on the right, so the legend sits
    # there rather than mid-left: no series is crowded and no band is covered
    axa.legend(loc="center right", bbox_to_anchor=(0.975, 0.47))

    _style_k_axis(axb)
    axb.set_ylim(0, 104)
    axb.set_yticks([0, 20, 40, 60, 80, 100])
    axb.set_ylabel("coverage (%)")

    save(fig, out_dir, "exp1_kscaling")
    plt.close(fig)

# Appendix figure
# --------------------------------------------------------------------------

def make_appendix_figure(curves, canon, out_dir):
    set_style()
    panel_ks = [10, 100]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.85), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.175, top=0.945,
                        wspace=0.09)

    ylo, yhi = 1.2e-7, 3e-3
    rug_lo, rug_hi = 1.35e-7, 1.95e-7

    for ax, K in zip(axes, panel_ks):
        gam = curves[("amortized", K)]["gamma"]
        truth = curves[("amortized", K)]["truth"]

        ax.plot(gam, truth, "-", color=TRUTH_COLOR, lw=0.8, zorder=9,
                label="Analytic truth")

        # ---- amortized ---------------------------------------------------
        est = curves[("amortized", K)]["est"]
        col = ARM_COLOR["amortized"]
        for si in range(est.shape[0]):
            ax.plot(gam, est[si], "-", color=col, lw=0.6, alpha=0.13, zorder=2)
    
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            amed = np.nanmedian(np.where(est > 0, est, np.nan), axis=0)
        ax.plot(gam, amed, "-", color=col, lw=2.0, alpha=0.9, zorder=6,
                label="Amortized (median)")

        miss = np.isnan(est).all(axis=0)
        if miss.any():
            ax.plot(gam[miss], truth[miss], linestyle="none", marker="o",
                    ms=3.0, mfc="white", mec=col, mew=0.8, zorder=10)

        # ---- unconditional CEM ---------------------------------------------
        est = curves[(CEM, K)]["est"]
        col = ARM_COLOR[CEM]
        for si in range(est.shape[0]):
            ax.plot(gam, est[si], "-", color=col, lw=0.6, alpha=0.13, zorder=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cmed = np.nanmedian(np.where(est > 0, est, np.nan), axis=0)

        ax.plot(gam, cmed, linestyle=(0, (4, 2)), color=col, lw=1.3, alpha=0.95,
                zorder=8, label="Unconditional CEM (median)")

        # ---- per-threshold -------------------------------------------------
        est = curves[("perthreshold", K)]["est"]
        col = ARM_COLOR["perthreshold"]
        pos = np.where(est > 0, est, np.nan)
        for si in range(est.shape[0]):
            ax.plot(gam, pos[si], linestyle="none", marker="o", ms=1.3,
                    color=col, alpha=0.20, mew=0, zorder=3)
        # all-NaN columns are expected (gammas where every seed returned zero)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pmed = np.nanmedian(pos, axis=0)
        ax.plot(gam, pmed, "-", color=col, lw=1.4, zorder=7,
                label="Per-threshold (median)")

        # zero-estimate rug, opacity proportional to the number of zero seeds
        n_zero = np.sum(est == 0.0, axis=0)
        for g, nz in zip(gam, n_zero):
            if nz > 0:
                ax.vlines(g, rug_lo, rug_hi, color=col, lw=0.9,
                          alpha=0.18 + 0.72 * (nz / N_SEEDS), zorder=5)

        ax.set_yscale("log")
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(-0.045, 1.19)
        ax.set_xlabel(r"robustness threshold $\gamma$")
        ax.set_title(rf"$m = {K}$", pad=4.0)

    axes[0].set_ylabel(r"failure probability $P(\rho \leq \gamma)$")

    # one compact legend; a proxy carries the rug, which has no line of its own
    h, l = axes[1].get_legend_handles_labels()
    h.append(matplotlib.lines.Line2D([], [], color=ARM_COLOR["perthreshold"],
                                     lw=0.9, alpha=0.8))
    l.append("Per-threshold zero estimates")
    axes[0].legend(h, l, loc="lower right", bbox_to_anchor=(1.0, 0.015),
                   fontsize=6.6, labelspacing=0.26, borderpad=0.40,
                   handlelength=1.5, handletextpad=0.5)

    save(fig, out_dir, "exp1_kscaling_curves")
    plt.close(fig)


# --------------------------------------------------------------------------

AUDIT_DIR = None

def save(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    # rasterized=False everywhere: lines, markers, axes and text stay vector
    for art in fig.findobj():
        try:
            art.set_rasterized(False)
        except Exception:
            pass

    fig.savefig(os.path.join(out_dir, stem + ".pdf"))
    fig.savefig(os.path.join(out_dir, stem + ".svg"))
    if AUDIT_DIR:
        os.makedirs(AUDIT_DIR, exist_ok=True)
        fig.savefig(os.path.join(AUDIT_DIR, stem + "_preview.png"), dpi=400)

# --------------------------------------------------------------------------

def audit(results_dir, mr, canon, curves, out_dir):
    print("=" * 78)
    print("EXP1 K-SCALING -- DATA AUDIT")
    print("=" * 78)

    print("\nSource files (read-only):")
    print(f"  root: {results_dir}")
    for f in ["config.json", "meta.json", "metrics.json"]:
        print(f"    {f:26s} {os.path.getsize(os.path.join(results_dir, f)):>8,d} B")
    for s in range(1, N_SEEDS + 1):
        p = os.path.join(results_dir, "raw", f"seed{s}.json")
        assert os.path.isfile(p), p
    print(f"    raw/seed1..seed{N_SEEDS}.json      "
          f"{sum(os.path.getsize(os.path.join(results_dir, 'raw', f'seed{s}.json')) for s in range(1, N_SEEDS+1)):>8,d} B")
    print("  NOT read: summary.json (its ILE averages over used grid points only), "
          "raw_state/")
    print("\n  Unconditional-CEM arm:")
    for f in (CEM_METRICS,):
        print(f"    {f}  {os.path.getsize(f):>9,d} B")
    print(f"    {CEM_RAW}/seed1..seed{N_SEEDS}.json  "
          f"{sum(os.path.getsize(os.path.join(CEM_RAW, f'seed{s}.json')) for s in range(1, N_SEEDS+1)):>9,d} B")
    cemcfg = json.load(open(os.path.join(CEM_ROOT, "config.json")))
    print(f"    estimator: {cemcfg['estimator']}")
    print(f"    density:   {cemcfg['density_evaluation']}")
    print(f"    grids:     {cemcfg['grid_source']}")

    cfg = json.load(open(os.path.join(results_dir, "config.json")))
    print(f"\nBudget/config: B_total={cfg['budget_total']}, Ks={cfg['Ks']}, "
          f"seeds={len(cfg['seeds'])}, problem={cfg['problem']}, d={cfg['dimension']}")

    print("\nMetric definition (metrics.json):")
    print(f"  endpoint_snapped        = {mr['endpoint_snapped']}   "
          f"(snap_tolerance = {mr['snap_tolerance']:g})")
    print(f"  rule                    = {mr['metric_definition']['rule']}")
    print(f"  ile                     = {mr['metric_definition']['ile']}")
    print(f"  coverage                = {mr['metric_definition']['coverage']}")
    print(f"  floor                   = {mr['metric_definition']['floor']}")

    print("\nSeeds per cell and plotted medians (metrics.json, ILE@1/B):")
    print(f"  {'arm':<13} {'K':>4} {'seeds':>6} {'floor 1/B':>10} "
          f"{'median ILE':>12} {'expected':>10} {'median cov':>11} "
          f"{'cov range':>14}")
    ok = True
    for arm in ARMS:
        for K in KS:
            c = canon[(arm, K)]
            e_ile, e_cov = EXPECTED_ILE[(arm, K)], EXPECTED_COV[(arm, K)]
            n_curve_seeds = curves[(arm, K)]["est"].shape[0]
            good = (c["n_seeds"] == N_SEEDS and n_curve_seeds == N_SEEDS
                    and abs(c["ile_median"] - e_ile) <= 5e-6 * max(1.0, e_ile)
                    and abs(c["cov_median"] - e_cov) <= 5e-4)
            ok &= good
            print(f"  {ARM_LABEL[arm]:<13} {K:>4} {c['n_seeds']:>6} "
                  f"{c['floor']:>10.1e} {c['ile_median']:>12.6f} "
                  f"{e_ile:>10.6f} {100*c['cov_median']:>10.1f}% "
                  f"{100*c['cov_min']:>6.1f}-{100*c['cov_max']:<6.1f}"
                  f"{'' if good else '   <-- MISMATCH'}")
    assert ok, "plotted medians do not match the expected metrics.json values"

    print("\nManuscript table cross-check (4 dp, unconditional CEM):")
    tab_ok = True
    for K in KS:
        got = f"{canon[(CEM, K)]['ile_median']:.4f}"
        want = CEM_TABLE_4DP[K]
        cov = f"{100*canon[(CEM, K)]['cov_median']:.1f}\\%"
        good = (got == want)
        tab_ok &= good
        print(f"  m={K:>4}  ILE {got}  (table: {want})  coverage {cov:>7}"
              f"{'' if good else '   <-- MISMATCH'}")
    assert tab_ok, "figure medians disagree with the manuscript table values"

    print("\n1/B floor invocations (must be zero for every CEM cell):")
    cm = json.load(open(CEM_METRICS))
    tot = 0
    for K in KS:
        c = cm["cells"][f"{CEM}_K{K}"]
        missing = sum(c["n_grid"] - n for n in c["n_pos"])
        tot += missing
        print(f"  m={K:>4}  floor 1/B = {c['floor'][0]:.1e}   "
              f"grid points needing the floor = {missing}")
    assert tot == 0, "the 1/B floor was invoked for a CEM cell"
    print(f"  total across all CEM cells = {tot}")

    # independent re-derivation of the metric from raw/, per arm
    print("\nIndependent re-derivation from raw/seed*.json under the metrics.json rule:")
    for arm in ARMS:
        for K in KS:
            cur = curves[(arm, K)]
            B = 1.0 / canon[(arm, K)]["floor"]
            t = cur["truth"]
            keep = np.isfinite(t) & (t > 0)
            est = cur["est"]
            valid = np.isfinite(est) & (est > 0)
            pt = np.where(valid, est, 1.0 / B)
            with np.errstate(divide="ignore", invalid="ignore"):
                ile = np.nanmean(np.abs(np.log(pt[:, keep]) - np.log(t[keep])), axis=1)
            cov = valid[:, keep].sum(axis=1) / keep.sum()
            d_ile = abs(float(np.median(ile)) - canon[(arm, K)]["ile_median"])
            d_cov = abs(float(np.median(cov)) - canon[(arm, K)]["cov_median"])
            tag = "exact" if d_ile < 1e-9 else f"delta={d_ile:.3e}"
            note = ""
            if arm == "amortized":
                note = ("  (raw record banks `null` at the snapped endpoint; "
                        "metrics.json recovers it -- see module docstring)")
            print(f"  {ARM_LABEL[arm]:<13} K={K:>4}  ILE {tag:<18} "
                  f"cov delta={d_cov:.3e}{note}")

    print("\nExecution:")
    print("  simulations run                     : 0 (no solver/estimator invoked)")
    print("  Julia processes started             : 0")
    print("  files written                       : figures only, under")
    print(f"                                        {out_dir}")
    print("  metric plotted                      : metrics.json and "
          "cem_baseline/metrics.json,")
    print("                                        endpoint-snapped, floor = 1/B")
    print("=" * 78)


def main():
    global AUDIT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--main-out", required=True,
                    help="manuscript figures/main directory")
    ap.add_argument("--appendix-out", required=True,
                    help="manuscript figures/appendix directory")
    ap.add_argument("--audit-dir", default=None,
                    help="where raster previews and the audit log go")
    a = ap.parse_args()

    results_dir = a.results_dir or default_results_dir()
    AUDIT_DIR = a.audit_dir

    mr, canon = load_metrics(results_dir)
    curves = load_curves(results_dir)

    make_main_figure(canon, a.main_out)
    make_appendix_figure(curves, canon, a.appendix_out)
    audit(results_dir, mr, canon, curves,
          f"{a.main_out} (main), {a.appendix_out} (appendix)")

    print("\nWrote:")
    for stem, d in (("exp1_kscaling", a.main_out),
                    ("exp1_kscaling_curves", a.appendix_out)):
        for ext in (".pdf", ".svg"):
            print(f"  {os.path.join(d, stem + ext)}")


if __name__ == "__main__":
    main()
