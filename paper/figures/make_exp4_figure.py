#!/usr/bin/env python3
"""
Experiment 4 driving failure-probability figure.

Plots the estimated failure-probability curve and per-threshold estimator
availability for the 150-dimensional IDM crosswalk benchmark.

The amortized method reports pooled balance-MIS estimates across the full
threshold grid even when its adaptive schedule does not reach the target
threshold. Schedule reach and estimator availability are therefore distinct.

Charged evaluation counts differ across methods and are reported separately,
so the figure should not be interpreted as a matched-budget comparison.

The left panel shows median finite positive estimates against the reference
curve. The right panel shows the fraction of seeds with a finite positive
estimate at each threshold.

Data are read from stored Experiment 4 results; no simulation is run.

Usage:
    python3 paper/figures/make_exp4_figure.py \
        [--results-dir DIR] [--out-dir DIR]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configuration
# --------------------------------------------------------------------------

SEEDS = list(range(1, 11))
GAMMAS = [0.434, 0.38, 0.32, 0.26, 0.20, 0.14, 0.08, 0.00]

ARMS = {
    "arm_a_primary":      ("Amortized (25k)",     "#4C72B0", "o", "a_primary_sgm"),
    "arm_b_perthreshold": ("Per-threshold (48k)", "#C44E52", "s", "b_perthreshold"),
    "arm_c_ams":          ("AMS (115k median)",   "#55A868", "^", "c_ams"),
}
REF_COLOR = "#000000"
REF_LABEL = "26M-rollout reference"

# reference (26M rollouts) at the 8 thresholds, ascending gamma
EXPECTED_REFERENCE = [
    3.107692307692308e-05,
    3.107692307692308e-05,
    0.00011880769230769231,
    0.00021146153846153846,
    0.0004996538461538462,
    0.001432423076923077,
    0.003776423076923077,
    0.008663807692307693]

# threshold-wise count of seeds with a finite positive estimate, ascending gamma
EXPECTED_N_POS = {
    "arm_a_primary":       [10, 10, 10, 10, 10, 10, 10, 10],
    "arm_b_perthreshold":  [2, 1, 5, 8, 10, 10, 10, 10],
    "arm_c_ams":           [5, 5, 8, 8, 10, 10, 10, 10],
}
EXPECTED_AMORTIZED_DESC = [10, 10, 10, 10, 10, 10, 10, 10]

# aggregates from metrics.json
EXPECTED_AGG = {
    "arm_a_primary":       dict(ile=0.48376, cov=1.0, any=10, zero=0, B=25000.0),
    "arm_b_perthreshold":  dict(ile=3.49781, cov=0.6875, any=10, zero=0, B=48000.0),
    "arm_c_ams":           dict(ile=0.46460, cov=0.875, any=10, zero=0, B=115018.0),
}
EXPECTED_AMS_ILE_ANY = 0.464595
EXPECTED_AMS_ZERO_SEEDS = []
EXPECTED_REFERENCE_N = 26_000_000
EXPECTED_D = 150

def default_results_dir():
    """Released layout: <repo>/paper/results/exp4. Resolved from this file, not the CWD."""
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.get("EXP4_RESULTS_DIR")
    if env:
        return env
    cand = repo_root / "results" / "exp4"
    if cand.is_dir():
        return str(cand)
    sys.exit("could not locate results/exp4; pass --results-dir or set EXP4_RESULTS_DIR")

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
        "legend.fontsize": 8.0,

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
        "ytick.minor.size": 1.6,

        "lines.solid_capstyle": "round",
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "legend.edgecolor": "0.3",
        "legend.borderpad": 0.42,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.55,
        "legend.labelspacing": 0.32,

        # True vector output; text stays real text (editable), not outlines.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "pdf.compression": 6,
        "figure.dpi": 150,
    })

# Data
# --------------------------------------------------------------------------

def is_pos(e):
    """A usable estimate: present, numeric, finite, strictly positive."""
    return (e is not None and isinstance(e, (int, float))
            and math.isfinite(e) and e > 0)

def load(results_dir):
    with open(os.path.join(results_dir, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(results_dir, "summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(results_dir, "config.json")) as f:
        config = json.load(f)

    raw = {}
    for s in SEEDS:
        with open(os.path.join(results_dir, "raw", f"seed{s}.json")) as f:
            raw[s] = json.load(f)

    # threshold grid, ascending gamma; the grid is identical in every record
    grid = raw[SEEDS[0]]["gamma_grid"]
    order = np.argsort(np.asarray(grid, float))
    gam = np.asarray(grid, float)[order]
    assert sorted(grid, reverse=True) == GAMMAS, f"unexpected grid {grid}"

    refs = {tuple(p["truth"] for p in raw[s][arm]["per_threshold"])
            for s in SEEDS for arm in ARMS}
    assert len(refs) == 1, "reference curve differs across arms/seeds"
    reference = np.asarray(next(iter(refs)), float)[order]

    curves = {}
    for arm in ARMS:
        n_pos, med, per_seed_cov = [], [], []
        for i in range(len(gam)):
            vals = [raw[s][arm]["per_threshold"][order[i]]["estimate"] for s in SEEDS]
            good = [v for v in vals if is_pos(v)]
            n_pos.append(len(good))
            med.append(float(np.median(good)) if good else np.nan)
        for s in SEEDS:
            pts = raw[s][arm]["per_threshold"]
            per_seed_cov.append(sum(1 for p in pts if is_pos(p["estimate"])) / len(pts))
        curves[arm] = {"n_pos": np.asarray(n_pos), "median": np.asarray(med),
                       "coverage_per_seed": per_seed_cov}

    # AMS seeds with no usable estimate anywhere
    zero_seeds = [s for s in SEEDS
                  if not any(is_pos(p["estimate"])
                             for p in raw[s]["arm_c_ams"]["per_threshold"])]

    return dict(metrics=metrics, summary=summary,
                config=config, gam=gam, reference=reference, curves=curves,
                zero_seeds=zero_seeds)

# Figure
# --------------------------------------------------------------------------

def make_figure(D, out_dir):
    set_style()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.5, 3.05))
    fig.subplots_adjust(left=0.098, right=0.988, bottom=0.160, top=0.965,
                        wspace=0.28)

    gam, ref = D["gam"], D["reference"]

    # ---- left: failure-probability curve --------------------------------
    axl.plot(gam, ref, "-", color=REF_COLOR, lw=1.3, zorder=3, label=REF_LABEL)
    for arm, (label, colour, marker, _) in ARMS.items():
        y = D["curves"][arm]["median"]          # NaN where no seed is usable ->
        axl.plot(gam, y, "-", color=colour, lw=1.0, zorder=4,   # curve absent
                 marker=marker, ms=4.4, mfc="white", mew=1.1, mec=colour,
                 label=label)
    axl.set_yscale("log")
    axl.set_ylim(3e-12, 6e-2)
    axl.set_yticks([1e-11, 1e-9, 1e-7, 1e-5, 1e-3])
    axl.set_ylabel(r"failure probability $P(\rho \leq \gamma)$")

    # ---- right: per-threshold availability, denominator always 10 --------
    for arm, (label, colour, marker, _) in ARMS.items():
        y = 100.0 * D["curves"][arm]["n_pos"] / len(SEEDS)
        axr.plot(gam, y, "-", color=colour, lw=1.0, zorder=4,
                 marker=marker, ms=4.4, mfc="white", mew=1.1, mec=colour,
                 label=label)
    axr.set_ylim(-5, 105)
    axr.set_yticks([0, 20, 40, 60, 80, 100])
    axr.set_ylabel("seeds with estimate (%)")

    for ax in (axl, axr):
        ax.set_xlim(-0.03, 0.465)
        ax.set_xticks(sorted(GAMMAS))
        ax.set_xticklabels(["0", "0.08", "0.14", "0.20", "0.26", "0.32", "0.38", "0.434"])
        ax.set_xlabel(r"robustness threshold $\gamma$")

    # one shared legend, in the curve panel's lower right: for gamma >= 0.26
    # every curve lies above 1e-7, so that block is empty and nothing is hidden
    axl.legend(loc="lower right", bbox_to_anchor=(0.995, 0.02))

    save(fig, out_dir, "exp4_driving_curve")
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

def audit(results_dir, D, out_dir):
    metrics, gam, ref = D["metrics"], D["gam"], D["reference"]
    print("=" * 86)
    print("EXP4 IDM (d=150) -- DRIVING-CURVE FIGURE AUDIT")
    print("=" * 86)

    print("\n1. Source files (all opened read-only):")
    for f in ["metrics.json", "summary.json", "config.json"]:
        print(f"     {f:26s} {os.path.getsize(os.path.join(results_dir, f)):>8,d} B")
    nb = sum(os.path.getsize(os.path.join(results_dir, 'raw', f'seed{s}.json'))
             for s in SEEDS)
    print(f"     raw/seed1..seed10.json     {nb:>8,d} B   "
          f"(per-seed per-threshold estimates + reference; not banked elsewhere)")
    print("   NOT read: raw_state/, diagnostics/, meta.json")
    sa = D["summary"]["seed_accounting"]
    assert sa["n_completed"] == len(SEEDS) and sa["n_failed"] == 0, sa
    print(f"   seed accounting (summary.json): {sa['n_completed']} completed, "
          f"{sa['n_failed']} failed")
    rc = D["config"]["reference_check"]
    assert rc["reference_n"] == EXPECTED_REFERENCE_N and rc["reference_d"] == EXPECTED_D
    assert rc["all_match"] is True
    print(f"   reference provenance: n={rc['reference_n']:,} rollouts, d={rc['reference_d']}, "
          f"all_match={rc['all_match']}")

    print("\n2. Reference values used at all eight thresholds (26M rollouts):")
    for g, r in zip(gam, ref):
        print(f"     gamma={g:>5.3f}   P_ref = {r:.6e}")
    assert np.allclose(ref, EXPECTED_REFERENCE, rtol=0, atol=0), \
        "reference curve does not match the pre-registered values"
    print("     (gamma=0.08 and gamma=0.00 share a value: the empirical reference")
    print("      plateaus there -- no additional rollouts fall between them.)")

    print("\n3. Per method and threshold: finite-positive count and plotted median")
    print(f"     {'gamma':>6} | " + " | ".join(
        f"{ARMS[a][0].split(' (')[0]:>26}" for a in ARMS))
    for i, g in enumerate(gam):
        cells = []
        for arm in ARMS:
            n = D["curves"][arm]["n_pos"][i]
            m = D["curves"][arm]["median"][i]
            cells.append(f"{n:>2}/10  {'absent':>12}" if not np.isfinite(m)
                         else f"{n:>2}/10  {m:>12.5e}")
        print(f"     {g:>6.3f} | " + " | ".join(f"{c:>26}" for c in cells))
    n_assert = 0
    for arm in ARMS:
        got = list(int(v) for v in D["curves"][arm]["n_pos"])
        assert got == EXPECTED_N_POS[arm], f"{arm} counts {got}"
        n_assert += len(got)
    print(f"     asserted {n_assert} threshold-wise counts (3 arms x 8 thresholds)")

    print("\n4. Amortized availability (descending gamma):")
    desc = list(int(v) for v in D["curves"]["arm_a_primary"]["n_pos"])[::-1]
    assert desc == EXPECTED_AMORTIZED_DESC, desc
    print("     " + ",  ".join(f"gamma={g:.3f}: {n}/10"
                               for g, n in zip(GAMMAS, desc)))
    print("     -> " + ", ".join(f"{n}/10" for n in desc) + "  (matches EXPECTED_AMORTIZED_DESC)")
    reached = D["summary"]["arms"][ARMS["arm_a_primary"][3]]["reached_target_count"]
    print(f"     adaptive schedule reaches gamma=0 in {reached}/10 seeds (summary.json);")
    print("     the pooled balance-MIS estimate is reported at all 8 thresholds.")

    print("\n5. AMS threshold-wise counts, every seed retained in the denominator:")
    assert D["zero_seeds"] == EXPECTED_AMS_ZERO_SEEDS, D["zero_seeds"]
    ams = D["curves"]["arm_c_ams"]
    print("     " + ",  ".join(f"{g:.3f}: {n}/10" for g, n in zip(gam, ams["n_pos"])))
    print(f"     zero-coverage seeds derived from raw: {D['zero_seeds'] or 'none'} -- all seeds kept.")
    c = metrics["cells"]["arm_c_ams"]
    assert c["n_seeds"] == 10 \
        and c["n_seeds_any_estimate"] == EXPECTED_AGG["arm_c_ams"]["any"] \
        and c["n_seeds_zero_coverage"] == EXPECTED_AGG["arm_c_ams"]["zero"]
    print(f"     metrics.json agrees: n_seeds=10, any={c['n_seeds_any_estimate']}, "
          f"zero_coverage={c['n_seeds_zero_coverage']}")

    print("\n   Cross-check of the derivation against metrics.json:")
    for arm in ARMS:
        der = D["curves"][arm]["coverage_per_seed"]
        mc = metrics["cells"][arm]["coverage_per_seed"]
        assert der == mc, f"{arm}: derived coverage {der} != metrics.json {mc}"
        print(f"     {arm:22s} per-seed finite-positive/8 == "
              f"metrics.json coverage_per_seed  (exact, seed for seed)")
    print("   NOTE: summary.json's arms[*].ile_median is the ILE over USED cells")
    print("         only, a different quantity from ILE@1/B -- not used here.")

    print("\n   Aggregates (metrics.json):")
    print(f"     {'arm':>22} {'B charged':>11} {'ILE':>10} {'cov med':>9} "
          f"{'any':>6} {'zero':>6}")
    for arm, e in EXPECTED_AGG.items():
        c = metrics["cells"][arm]
        assert abs(c["ile_primary_median"] - e["ile"]) < 5e-6
        assert abs(c["coverage_median"] - e["cov"]) < 1e-9
        assert c["n_seeds_any_estimate"] == e["any"]
        assert c["n_seeds_zero_coverage"] == e["zero"]
        assert c["B_actual_median"] == e["B"]
        print(f"     {ARMS[arm][0]:>22} {int(c['B_actual_median']):>11,} "
              f"{c['ile_primary_median']:>10.5f} {100*c['coverage_median']:>8.0f}% "
              f"{c['n_seeds_any_estimate']:>5} {c['n_seeds_zero_coverage']:>6}")
    ams_any = metrics["cells"]["arm_c_ams"]["ile_conditional_on_any_estimate_median"]
    assert abs(ams_any - EXPECTED_AMS_ILE_ANY) < 5e-6
    print(f"     AMS ILE|any = {ams_any:.6f}")

    print("\n6. No 1/B floor value appears in the probability panel:")
    for arm in ARMS:
        floor = metrics["cells"][arm]["floor_primary_median"]
        med = D["curves"][arm]["median"]
        finite = med[np.isfinite(med)]
        assert not np.any(np.isclose(finite, floor, rtol=1e-9, atol=0)), arm
        n_absent = int(np.sum(~np.isfinite(med)))
        print(f"     {ARMS[arm][0]:>22}  floor 1/B = {floor:.5e}  "
              f"-> not among the {len(finite)} plotted points; "
              f"{n_absent} threshold(s) left ABSENT")
    print("     Every plotted point is a median over genuine finite positive")
    print("     estimates; missing estimates are never substituted.")

    print("\n7. Execution:")
    print("     simulations run                   : 0 (no solver/estimator invoked)")
    print("     Julia processes started           : 0")
    print("     artifacts modified                : 0 (all inputs opened read-only;")
    print("                                          raw/ read, never written)")
    print(f"     files written                     : figures only, under\n"
          f"                                          {out_dir}")
    print("     quantities plotted                : median finite positive estimate;")
    print("                                          finite-positive seed fraction")
    print("     NOT plotted                       : fit ESS, gamma-min, separation,")
    print("                                          runtime, GPD tails, tail ESS,")
    print("                                          defensive-mixture diagnostics,")
    print("                                          1/B floor-substituted estimates")
    print("     display-only devices              : none -- no dodge, no jitter, no")
    print("                                          annotation; budgets carried in")
    print("                                          the legend labels so the figure")
    print("                                          cannot read as matched-budget.")
    print("=" * 86)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    results_dir = a.results_dir or default_results_dir()
    out_dir = a.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "exp4_idm")

    D = load(results_dir)
    make_figure(D, out_dir)
    audit(results_dir, D, out_dir)

    print("\n8. Output paths:")
    for f in sorted(os.listdir(out_dir)):
        print(f"     {os.path.join(out_dir, f)}")


if __name__ == "__main__":
    main()
