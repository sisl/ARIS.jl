#!/usr/bin/env python3
"""
Experiment 2 radial-shell discovery figure.

Shows the fraction of seeds that produce at least one finite positive estimate
on the evaluation grid for the amortized, per-threshold, and AMS methods.

The panels are indexed by nominal experiment budget. Actual evaluation counts
can differ across methods because oracle-tuning evaluations are included in the
reported cost, so this figure should not be interpreted as a budget-normalized
efficiency comparison.

Data are read from `results/exp2/metrics.json`; no simulation is run.

Usage:
    python3 paper/figures/make_exp2_figure.py \
        [--results-dir DIR] [--out-dir DIR]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configuration

BUDGETS = [25000, 50000]
DIMS = [2, 20]

# key -> (legend label, colour, marker, linestyle, zorder)
ARMS = {
    "arm_a": ("Amortized",            "#4C72B0", "o", "-",  3),
    "arm_c": ("AMS oracle",           "#55A868", "^", "-",  3),
    "arm_b": ("Per-threshold oracle", "#C44E52", "s", "--", 4),
}
LEGEND_ORDER = ["arm_a", "arm_b", "arm_c"]
DODGE = {"arm_a": 0.0, "arm_b": -0.55, "arm_c": +0.55}

EXPECTED = {
    (25000,  2, "arm_a"):  10.0, (25000,  2, "arm_b"): 100.0, (25000,  2, "arm_c"): 100.0,
    (25000, 20, "arm_a"):   0.0, (25000, 20, "arm_b"): 100.0, (25000, 20, "arm_c"): 100.0,
    (50000,  2, "arm_a"):  40.0, (50000,  2, "arm_b"): 100.0, (50000,  2, "arm_c"): 100.0,
    (50000, 20, "arm_a"):   0.0, (50000, 20, "arm_b"): 100.0, (50000, 20, "arm_c"): 100.0,
}
EXPECTED_N_SEEDS = 10

def cell_key(d, budget, arm):
    return f"d{d}_B{budget}|{arm}"

def default_results_dir():
    """Released layout: <repo>/paper/results/exp2. Resolved from this file, not the CWD."""
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.get("EXP2_RESULTS_DIR")
    if env:
        return env
    cand = repo_root / "results" / "exp2"
    if cand.is_dir():
        return str(cand)
    sys.exit("could not locate results/exp2; pass --results-dir or set EXP2_RESULTS_DIR")

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
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

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

# Data
# --------------------------------------------------------------------------

def load_discovery(results_dir):
    """discovery[(budget, d, arm)] = dict(n_any, n_seeds, rate_pct)."""
    with open(os.path.join(results_dir, "metrics.json")) as f:
        mr = json.load(f)
    out = {}
    for B in BUDGETS:
        for d in DIMS:
            for arm in ARMS:
                c = mr["cells"][cell_key(d, B, arm)]
                n_any, n = c["n_seeds_any_estimate"], c["n_seeds"]
                out[(B, d, arm)] = {"n_any": n_any, "n_seeds": n,
                                    "rate_pct": 100.0 * n_any / n}
    return mr, out

# Figure
# --------------------------------------------------------------------------

def make_figure(disc, out_dir):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.75), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.180, top=0.905,
                        wspace=0.09)

    handles = {}
    for ax, B in zip(axes, BUDGETS):
        for arm, (label, colour, marker, ls, z) in ARMS.items():
            x = np.array([d + DODGE[arm] for d in DIMS], float)
            y = np.array([disc[(B, d, arm)]["rate_pct"] for d in DIMS])
            ln, = ax.plot(x, y, ls, color=colour, lw=1.0, zorder=z,
                          dashes=(4.0, 2.0) if ls == "--" else (None, None),
                          marker=marker, ms=5.0, mfc="white", mew=1.2,
                          mec=colour, label=label)
            handles.setdefault(arm, (ln, label))

        colour = ARMS["arm_a"][1]
        for d in DIMS:
            v = disc[(B, d, "arm_a")]["rate_pct"]
            ax.annotate(f"{v:.0f}%", (d, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", va="bottom",
                        fontsize=7.5, color=colour, zorder=4)

        ax.set_xlim(-1.6, 23.6)
        ax.set_xticks(DIMS)
        ax.set_xticklabels([str(d) for d in DIMS])
        ax.set_xlabel(r"input dimension $d$")
        ax.set_title(rf"$B = {B:,}$".replace(",", "{,}"), pad=5.0)

    axes[0].set_ylim(-5, 105)
    axes[0].set_yticks([0, 20, 40, 60, 80, 100])
    axes[0].set_ylabel("discovery rate (%)")

    axes[0].legend([handles[a][0] for a in LEGEND_ORDER],
                   [handles[a][1] for a in LEGEND_ORDER],
                   loc="center", bbox_to_anchor=(0.5, 0.52))

    save(fig, out_dir, "exp2_shell_discovery")
    plt.close(fig)

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
    fig.savefig(os.path.join(out_dir, stem + "_preview.png"), dpi=400)

# Audit
# --------------------------------------------------------------------------

def audit(results_dir, mr, disc, out_dir):
    print("=" * 78)
    print("EXP2 SHELL -- DISCOVERY FIGURE DATA AUDIT")
    print("=" * 78)

    print("\nSource files (read-only):")
    print(f"  root: {results_dir}")
    for f in ["config.json", "meta.json", "metrics.json"]:
        print(f"    {f:26s} {os.path.getsize(os.path.join(results_dir, f)):>8,d} B")
    print("  NOT read: summary.json, raw/, raw_state/")

    cfg = json.load(open(os.path.join(results_dir, "config.json")))
    print(f"\nConfig: problem={cfg['problem']}, n_grid_points={cfg['n_grid_points']}, "
          f"n_seeds={cfg['n_seeds']}, arms={cfg['arms']}")
    print(f"  truth      : {cfg['truth']}")
    print(f"  discovery  : {cfg['discovery_policy']}")

    print("\nDiscovery rate = n_seeds_any_estimate / n_seeds "
          "(>=1 finite positive estimate on the 16-threshold grid):")
    print(f"  {'nominal B':>10} {'d':>4} {'arm':>22} {'seeds':>12} "
          f"{'rate':>7} {'expected':>9}")
    ok = True
    for B in BUDGETS:
        for d in DIMS:
            for arm in LEGEND_ORDER:
                label = ARMS[arm][0]
                r = disc[(B, d, arm)]
                exp = EXPECTED[(B, d, arm)]
                good = (abs(r["rate_pct"] - exp) < 1e-9
                        and r["n_seeds"] == EXPECTED_N_SEEDS)
                ok &= good
                print(f"  {B:>10,} {d:>4} {label:>22} "
                      f"{r['n_any']:>5}/{r['n_seeds']:<6} {r['rate_pct']:>6.1f}% "
                      f"{exp:>8.1f}%" + ("" if good else "   <-- MISMATCH"))
    assert ok, "discovery rates do not match the pre-registered values"

    amort = [(B, d, disc[(B, d, 'arm_a')]) for B in BUDGETS for d in DIMS]
    print("\n  amortized discovery recovered: "
          + ",  ".join(f"B={B:,}/d={d}: {r['n_any']}/{r['n_seeds']} "
                       f"({r['rate_pct']:.0f}%)" for B, d, r in amort))
    base_ok = all(disc[(B, d, a)]["rate_pct"] == 100.0
                  for B in BUDGETS for d in DIMS for a in ("arm_b", "arm_c"))
    print(f"  both oracle baselines at 100% in all four cells: "
          f"{'YES' if base_ok else 'NO'}")
    assert base_ok

    print("\nExecution:")
    print("  simulations run                     : 0 (no solver/estimator invoked)")
    print("  Julia processes started             : 0")
    print("  experiment artifacts modified       : 0 (all inputs opened read-only)")
    print(f"  files written                       : figures only, under\n"
          f"                                        {out_dir}")
    print("  quantity plotted                    : n_seeds_any_estimate / n_seeds "
          "(metrics.json)")
    print("  display-only devices                : horizontal dodge of "
          f"{DODGE['arm_b']:+.2f}/{DODGE['arm_c']:+.2f} in d for the two oracle")
    print("                                        arms (they coincide at 100%); "
          "amortized undodged.")
    print("                                        Amortized values annotated "
          "(10%, 0%, 40%, 0%). No axis")
    print("                                        label or plotted value is "
          "altered by either device.")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    results_dir = a.results_dir or default_results_dir()
    out_dir = a.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "exp2_shell")

    mr, disc = load_discovery(results_dir)
    make_figure(disc, out_dir)
    audit(results_dir, mr, disc, out_dir)

    print("\nWrote:")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {os.path.join(out_dir, f)}")


if __name__ == "__main__":
    main()
