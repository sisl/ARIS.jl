#!/usr/bin/env python3
"""Exp3 per-cell aggregates from the per-seed offline diagnostics.

Reads results/exp3/diagnostics/offline_exp3_per_seed.json (written from raw state by
offline_exp3_reconstruct.py) and writes results/exp3/diagnostics/exp3_derived_aggregates.json.
Every (c2, arm) cell gets the full 10-seed aggregate.

Usage: python paper/scripts/exp3_aggregates.py
"""
import json, math, os

import numpy as np

DIAG = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results", "exp3", "diagnostics"))
C2S, ARMS = (16.0, 18.0, 20.0, 23.0), ("conditioned", "unconditional")

phi = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))   # standard normal CDF
P1 = phi(-5.5)                                          # x1 lobe probability
PL = lambda c2: phi(-math.sqrt(c2))                     # each x2 lobe probability

def main():
    per_seed = json.load(open(os.path.join(DIAG, "offline_exp3_per_seed.json")))
    rows = [dict(r, fallback_assisted=(r["fit_method"] == "fallback_assisted"),
                 R_A=r["P_mode1"] / P1,
                 R_U=r["P_mode2_upper"] / PL(r["c2"]),
                 R_L=r["P_mode2_lower"] / PL(r["c2"]))
            for r in per_seed]
    agg = []
    for c2 in C2S:
        for arm in ARMS:
            for label, keep in (("full", lambda r: True),
                                ("clean_only", lambda r: not r["fallback_assisted"])):
                g = [r for r in rows if r["c2"] == c2 and r["arm"] == arm and keep(r)]
                if label == "clean_only" and len(g) == 10:
                    continue
                pv = np.array([r["P_over_truth"] for r in g])
                agg.append({
                    "c2": int(c2), "arm": arm, "aggregate": label, "n": len(g),
                    "median_P_hat": float(np.median([r["P_hat"] for r in g])),
                    "truth": g[0]["truth"],
                    "median_P_over_truth": float(np.median(pv)),
                    "log10_median_underestimation": float(-np.log10(np.median(pv))),
                    "iqr_lo": float(np.percentile(pv, 25)), "iqr_hi": float(np.percentile(pv, 75)),
                    "min": float(pv.min()), "max": float(pv.max()),
                    "n_above_truth": int((pv > 1).sum()),
                    "n_within_2x": int(((pv >= .5) & (pv <= 2)).sum()),
                    "n_within_10x": int(((pv >= .1) & (pv <= 10)).sum()),
                    "n_within_100x": int(((pv >= .01) & (pv <= 100)).sum()),
                    "all_three_lobes_ge1": sum(r["found_all_three_lobes"] for r in g),
                    "all_three_lobes_ge5": sum(1 for r in g if min(r["n_mode1"], r["n_mode2_upper"],
                                                                   r["n_mode2_lower"]) >= 5),
                    "amplified": sum(1 for r in g if r["regime_banked"] == "amplified"),
                    "locked": sum(1 for r in g if r["regime_banked"] == "locked"),
                    "trace": sum(1 for r in g if r["regime_banked"] == "trace"),
                    "collapsed": sum(1 for r in g if r["regime_banked"] == "collapsed"),
                    "fallback_assisted": sum(1 for r in g if r["fallback_assisted"]),
                    "n_seeds_R_U_ge_0p1": sum(1 for r in g if r["R_U"] >= 0.1),
                    "n_seeds_R_L_ge_0p1": sum(1 for r in g if r["R_L"] >= 0.1),
                    "n_seeds_both_x2_lobes_ge_0p1": sum(1 for r in g if r["R_U"] >= 0.1 and r["R_L"] >= 0.1),
                    "median_R_A": float(np.median([r["R_A"] for r in g])),
                })
    p = os.path.join(DIAG, "exp3_derived_aggregates.json")
    with open(p, "w") as f:
        json.dump(agg, f, indent=1)
        f.write("\n")
    print(f"wrote {p} ({len(agg)} rows)")


if __name__ == "__main__":
    main()
