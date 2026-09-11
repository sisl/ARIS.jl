#!/usr/bin/env python3
"""
Experiment 3 tail-race mode-discovery figure.

Compares robustness-conditioned and unconditional proposal learning on the
mode-switching benchmark under a matched simulation budget.

The figure reports two mode-structure outcomes across seeds:
    * fraction discovering all three failure lobes
    * fraction classified in the amplified regime

These panels characterize access to changing failure modes rather than
failure-probability estimation accuracy.

A small number of unconditional-baseline runs use the EMGMM fallback path;
the figure includes those runs in the aggregate, and the manuscript caption
reports this caveat.

Data are read from stored Experiment 3 results; no simulation is run.

Usage:
    python3 paper/figures/make_exp3_figure.py \
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

C2_VALUES = [16, 18, 20, 23]

ARMS = {
    "conditioned":   ("Conditioned",   "#4C72B0", "o"),
    "unconditional": ("Unconditional", "#C44E52", "s"),
}

EXPECTED_LOBES = {
    (16, "conditioned"): 6, (16, "unconditional"): 1,
    (18, "conditioned"): 3, (18, "unconditional"): 0,
    (20, "conditioned"): 5, (20, "unconditional"): 0,
    (23, "conditioned"): 1, (23, "unconditional"): 0,
}
EXPECTED_AMPLIFIED = {
    (16, "conditioned"): 7, (16, "unconditional"): 4,
    (18, "conditioned"): 3, (18, "unconditional"): 1,
    (20, "conditioned"): 2, (20, "unconditional"): 0,
    (23, "conditioned"): 1, (23, "unconditional"): 0,
}
EXPECTED_N_SEEDS = 10

# fallback-assisted runs as (c2, arm, seed), asserted against summary.json and raw/*.json
EXPECTED_FALLBACK_RUNS = [(16.0, "unconditional", 4), (18.0, "unconditional", 1),
                          (23.0, "unconditional", 8)]

# Count-label placement as (offset in points, ha, va), keyed by (panel, c2, arm). Conditioned
# labels default to above their marker and unconditional labels to below; these entries move
# labels that would otherwise touch a line or marker.
LABEL_PLACEMENT = {
    ("lobes", 18, "conditioned"):       ((0, -8), "center", "top"),     # below: V-shaped minimum
    ("lobes", 23, "conditioned"):       ((5, 5), "left", "bottom"),     # above-right: incoming line
    ("amplified", 18, "conditioned"):   ((0, -8), "center", "top"),     # below: steep incoming line
    ("amplified", 16, "unconditional"): ((0, 7), "center", "bottom"),   # above: outgoing line
}

PANELS = [
    ("lobes",     "seeds discovering all three lobes (%)"),
    ("amplified", "amplified-regime frequency (%)"),
]

def cell_key(c2, arm):
    return f"c2{float(c2)}_{arm}"

def default_results_dir():
    """Released layout: <repo>/paper/results/exp3. Resolved from this file, not the CWD."""
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.get("EXP3_RESULTS_DIR")
    if env:
        return env
    cand = repo_root / "results" / "exp3"
    if cand.is_dir():
        return str(cand)
    sys.exit("could not locate results/exp3; pass --results-dir or set EXP3_RESULTS_DIR")

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

def load_counts(results_dir):
    """counts[(c2, arm)] = dict(n_seeds, lobes, amplified, fallback)."""
    with open(os.path.join(results_dir, "summary.json")) as f:
        summary = json.load(f)

    # per-run fit_method from raw/, used only for the fallback-assisted check
    fallback_runs = []
    raw_dir = os.path.join(results_dir, "raw")
    n_raw = 0
    for fn in sorted(os.listdir(raw_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(raw_dir, fn)) as f:
            r = json.load(f)
        n_raw += 1
        if r["fit_method"] == "fallback_assisted":
            fallback_runs.append((r["c2"], r["arm"], r["seed"]))

    counts = {}
    for c2 in C2_VALUES:
        for arm in ARMS:
            c = summary["cells"][cell_key(c2, arm)]
            n = c["seed_accounting"]["n_completed"]
            counts[(c2, arm)] = {
                "n_seeds": n,
                "lobes": c["found_all_three_lobes_count"],
                "amplified": c["regime_counts"].get("amplified", 0),
                "regimes": c["regime_counts"],
                "fallback": c["fit_method_fallback_assisted_count"],
            }
    return summary, (n_raw, sorted(fallback_runs)), counts

def rate(counts, c2, arm, field):
    r = counts[(c2, arm)]
    return 100.0 * r[field] / r["n_seeds"]

# Figure
# --------------------------------------------------------------------------

def make_figure(counts, out_dir):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    fig.subplots_adjust(left=0.098, right=0.988, bottom=0.165, top=0.965,
                        wspace=0.30)

    x = np.array(C2_VALUES, float)

    for ax, (field, ylabel) in zip(axes, PANELS):
        for arm, (label, colour, marker) in ARMS.items():
            y = np.array([rate(counts, c2, arm, field) for c2 in C2_VALUES])
            ax.plot(x, y, "-", color=colour, lw=1.0, zorder=3,
                    marker=marker, ms=5.0, mfc="white", mew=1.2, mec=colour,
                    label=label)

            above = (arm == "conditioned")
            for c2 in C2_VALUES:
                n = counts[(c2, arm)][field]
                if n == 0:
                    continue
                default = ((0, 7), "center", "bottom") if above else ((0, -8), "center", "top")
                xytext, ha, va = LABEL_PLACEMENT.get((field, c2, arm), default)
                ax.annotate(f"{n}/{counts[(c2, arm)]['n_seeds']}",
                            (c2, rate(counts, c2, arm, field)),
                            textcoords="offset points",
                            xytext=xytext, ha=ha, va=va,
                            fontsize=7.2, color=colour, zorder=4)

        ax.set_xlim(15.0, 24.0)
        ax.set_xticks(C2_VALUES)
        ax.set_xticklabels([str(c) for c in C2_VALUES])
        ax.set_xlabel(r"tail-separation parameter $c_2$")
        ax.set_ylim(-6, 105)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_ylabel(ylabel)

    axes[0].legend(loc="upper right", bbox_to_anchor=(0.995, 0.99))

    save(fig, out_dir, "exp3_conditioning_modes")
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

def audit(results_dir, summary, fits, counts, out_dir):
    print("=" * 78)
    print("EXP3 TAIL-RACE -- CONDITIONING / MODE-DISCOVERY FIGURE AUDIT")
    print("=" * 78)

    print("\nSource files (read-only):")
    print(f"  root: {results_dir}")
    for f in ["summary.json"]:
        print(f"    {f:26s} {os.path.getsize(os.path.join(results_dir, f)):>8,d} B")
    n_raw, fallback_runs = fits
    print(f"    raw/*.json                 {n_raw} run records (field `fit_method` only)")
    print("  NOT read: raw_state/  (ILE/coverage are undefined for this "
          "single-threshold study)")

    cfg = summary["config"]
    keep = ["problem", "c1", "c2_values", "d", "target_threshold", "n_seeds",
            "budget_total", "arms"]
    print("\nConfig: " + ", ".join(f"{k}={cfg[k]}" for k in keep if k in cfg))

    n_assert = 0
    for field, expected, title in (
            ("lobes", EXPECTED_LOBES, "All-three-lobes discovery"),
            ("amplified", EXPECTED_AMPLIFIED, "Amplified-regime frequency")):
        print(f"\n{title} (count / seeds completed):")
        print(f"  {'c2':>4} {'arm':>15} {'count':>10} {'rate':>7} "
              f"{'expected':>9} {'fallback':>8}")
        for c2 in C2_VALUES:
            for arm, (label, _, _) in ARMS.items():
                r = counts[(c2, arm)]
                exp = expected[(c2, arm)]
                good = (r[field] == exp and r["n_seeds"] == EXPECTED_N_SEEDS)
                n_assert += 1
                assert good, f"{title} mismatch at c2={c2}, {arm}"
                print(f"  {c2:>4} {label:>15} {r[field]:>4}/{r['n_seeds']:<5} "
                      f"{rate(counts, c2, arm, field):>6.0f}% {exp:>7}/10 "
                      f"{r['fallback']:>8}")
    print(f"\n  assertions passed: {n_assert}/16 expected values "
          f"(8 all-three-lobes + 8 amplified-regime)")

    print("\nFull regime breakdown (context; only 'amplified' is plotted):")
    for c2 in C2_VALUES:
        for arm, (label, _, _) in ARMS.items():
            print(f"  c2={c2:<3} {label:<15} {counts[(c2, arm)]['regimes']}")

    print("\nFallback-assisted runs (fit_method_fallback_assisted_count, summary.json):")
    for c2 in C2_VALUES:
        for arm in ARMS:
            want = sum(1 for fc2, farm, _ in EXPECTED_FALLBACK_RUNS
                       if (fc2, farm) == (float(c2), arm))
            assert counts[(c2, arm)]["fallback"] == want, \
                f"fallback-assisted count mismatch at c2={c2}, {arm}"
    assert fallback_runs == EXPECTED_FALLBACK_RUNS, fallback_runs
    assert n_raw == len(C2_VALUES) * len(ARMS) * EXPECTED_N_SEEDS, n_raw
    for fc2, farm, s in fallback_runs:
        print(f"  c2={fc2:.0f} {farm} seed {s} "
              f"(summary.json count for the cell = {counts[(int(fc2), farm)]['fallback']}; "
              f"raw fit_method agrees)")
    print(f"  plotted aggregates INCLUDE all {len(fallback_runs)} fallback-assisted runs;")
    print(f"    they are NOT singled out visually anywhere in the figure.")
    print(f"  {n_raw - len(fallback_runs)} of {n_raw} runs use fit_method = em")

    print("\nExecution:")
    print("  simulations run                     : 0 (no solver/estimator invoked)")
    print("  Julia processes started             : 0")
    print("  raw experiment artifacts modified   : 0 (all inputs opened read-only;")
    print("                                        raw_state/ never opened)")
    print(f"  files written                       : figures only, under\n"
          f"                                        {out_dir}")
    print("  accuracy quantities plotted         : NONE (no P_hat/truth, ILE, "
          "relerr, fit ESS,")
    print("                                        gamma-min, runtime, small_mix, "
          "tail ESS)")
    print("  display-only devices                : NO horizontal dodge -- the two "
          "arms never")
    print("                                        coincide in either panel, so "
          "none is needed.")
    print("                                        Exact counts annotated as n/10 "
          "(conditioned above")
    print("                                        its markers, unconditional "
          "below, except LABEL_PLACEMENT); zero counts left")
    print("                                        unlabelled. Neither device "
          "alters a plotted value.")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    results_dir = a.results_dir or default_results_dir()
    out_dir = a.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "exp3_tailrace")

    summary, fits, counts = load_counts(results_dir)
    make_figure(counts, out_dir)
    audit(results_dir, summary, fits, counts, out_dir)

    print("\nWrote:")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {os.path.join(out_dir, f)}")


if __name__ == "__main__":
    main()
