#!/usr/bin/env python3
"""
Experiment 3 offline diagnostics.

Recomputes driver metrics and per-iteration mode-discovery diagnostics from
the stored raw-state bundle without rerunning the simulator.

The diagnostic output covers both conditioned and unconditional methods.
`fit_method` is read from the stored experiment record so fallback-assisted
runs retain the classification assigned by the driver.
"""

import json, os, sys, math
from pathlib import Path
import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
E3 = str(REPO / "results" / "exp3")
OUT = os.environ.get("EXP3_DIAGNOSTICS_OUT", str(REPO / "results" / "exp3" / "diagnostics"))
RAW_STATE = os.environ.get("RAW_STATE_DIR", os.path.join(E3, "raw_state"))
C1 = 5.5
C2S = [16.0, 18.0, 20.0, 23.0]
ARMS = ["conditioned", "unconditional"]
SEEDS = list(range(1, 11))

def truth_at0(c2):
    P1 = norm.cdf(-C1)
    P2 = 2.0 * norm.cdf(-math.sqrt(c2))
    return P1 + P2 - P1 * P2

def load_blocks(key):
    man = json.load(open(f"{RAW_STATE}/{key}.manifest.json"))
    buf = np.fromfile(f"{RAW_STATE}/{man['arrays_file']}", dtype="<f8")
    out = {}
    for b in man["blocks"]:
        name = b["name"].split("/")[-1]
        off = b["offset"] // 8
        n = b["nbytes"] // 8
        a = buf[off:off + n]
        if len(b["shape"]) == 2:
            # column-major (2, N) -> rows are samples after C-order (N,2) reshape
            a = a.reshape(b["shape"][1], b["shape"][0])
        out[name] = a
    return man, out

def analyse(c2, arm, seed):
    key = f"c2{c2}_{arm}_seed{seed}"
    man, B = load_blocks(key)
    X, rho, w, logq, it, dm = B["X"], B["rho"], B["w"], B["logq"], B["iter"], B["defensive"]
    x1, x2 = X[:, 0], X[:, 1]
    M = len(rho)
    truth = truth_at0(c2)

    fail = rho <= 0.0
    # driver's classify(): m1 == branch 1 (c1 - x1) is the active minimiser
    m1_all = (C1 - x1) <= (c2 - x2 ** 2)
    up_all = x2 > 0.0

    f_m1 = fail & m1_all
    f_m2u = fail & (~m1_all) & up_all
    f_m2l = fail & (~m1_all) & (~up_all)

    l1, l2u, l2l = int(f_m1.sum()), int(f_m2u.sum()), int(f_m2l.sum())
    P = float((fail * w).sum() / M)
    Pm1 = float(w[f_m1].sum() / M) if l1 else 0.0
    Pm2 = float(w[f_m2u | f_m2l].sum() / M) if (l2u + l2l) else 0.0
    Pm2u = float(w[f_m2u].sum() / M) if l2u else 0.0
    Pm2l = float(w[f_m2l].sum() / M) if l2l else 0.0

    # first-hit iteration per lobe (reconstructed for BOTH arms)
    def firsthit(mask):
        if not mask.any():
            return 0
        return int(it[mask].min())

    # final-iteration occupancy (last iteration only)
    last = it == it.max()
    fin = {
        "fin_n_m1": int((last & f_m1).sum()),
        "fin_n_m2u": int((last & f_m2u).sum()),
        "fin_n_m2l": int((last & f_m2l).sum()),
        "fin_n_fail": int((last & fail).sum()),
    }
    # last-5-iteration occupancy (retention)
    l5 = it >= it.max() - 4
    ret = {
        "l5_n_m1": int((l5 & f_m1).sum()),
        "l5_n_m2u": int((l5 & f_m2u).sum()),
        "l5_n_m2l": int((l5 & f_m2l).sum()),
    }

    niter = int(it.max())
    occ = {"m1": [], "m2u": [], "m2l": []}
    for k in range(1, niter + 1):
        sel = it == k
        occ["m1"].append(int((sel & f_m1).sum()))
        occ["m2u"].append(int((sel & f_m2u).sum()))
        occ["m2l"].append(int((sel & f_m2l).sum()))

    prox = {
        "prop_x1_ge_c1": float((x1 >= C1).mean()),
        "prop_x2_ge_sqrtc2": float((x2 >= math.sqrt(c2)).mean()),
        "prop_x2_le_msqrtc2": float((x2 <= -math.sqrt(c2)).mean()),
    }

    ess = lambda v: float(v.sum() ** 2 / (v ** 2).sum()) if len(v) else float("nan")

    row = json.load(open(f"{E3}/raw/{key}.json"))
    rec = {
        "c2": c2, "arm": arm, "seed": seed, "truth": truth,
        "M": M, "n_fail": int(fail.sum()),
        "n_mode1": l1, "n_mode2_upper": l2u, "n_mode2_lower": l2l,
        "P_hat": P, "P_over_truth": P / truth,
        "P_mode1": Pm1, "P_mode2": Pm2, "P_mode2_upper": Pm2u, "P_mode2_lower": Pm2l,
        "Pm2_over_truth": Pm2 / truth,
        "found_all_three_lobes": bool(l1 > 0 and l2u > 0 and l2l > 0),
        "firsthit_mode1": firsthit(f_m1),
        "firsthit_mode2_upper": firsthit(f_m2u),
        "firsthit_mode2_lower": firsthit(f_m2l),
        "n_defensive": int(dm.sum()),
        "n_nonfinite_w": int((~np.isfinite(w)).sum()),
        "n_zero_w": int((w == 0).sum()),
        "w_max": float(np.nanmax(w)),
        "tail_ess_all": ess(w[fail]) if int(fail.sum()) >= 2 else None,
        "tail_ess_mode2": ess(w[f_m2u | f_m2l]) if (l2u + l2l) >= 2 else None,
        "small_mix": row.get("small_mix"), "small_det": row.get("small_det"),
        "fit_method": row.get("fit_method"), "regime_banked": row.get("regime"),
        "banked_P_hat": row.get("P_hat"), "banked_P_over_truth": row.get("P_over_truth"),
        "banked_n_mode1": row.get("n_mode1"),
        "banked_n_mode2_upper": row.get("n_mode2_upper"),
        "banked_n_mode2_lower": row.get("n_mode2_lower"),
        "banked_found_all_three": row.get("found_all_three_lobes"),
        "banked_P_mode2": row.get("P_mode2"),
        "secs": row.get("secs"),
    }
    row_small_mix = row.get("small_mix") or 0
    rec["regime_recomputed"] = (
        "locked" if (l2u + l2l) == 0 else
        "amplified" if (Pm2 / truth) > 0.1 else
        "collapsed" if (row_small_mix > 50 and (l2u + l2l) > 1000) else "trace")
    rec.update(fin); rec.update(ret); rec.update(prox)
    rec["occ"] = occ
    return rec


def main():
    recs = []
    for c2 in C2S:
        for arm in ARMS:
            for s in SEEDS:
                recs.append(analyse(c2, arm, s))
                print(f"  done {c2} {arm} {s}", file=sys.stderr)
    json.dump(recs, open(f"{OUT}/offline_exp3_per_seed.json", "w"), indent=1)

    # flat CSV without the per-iteration vectors
    import csv
    keys = [k for k in recs[0] if k != "occ"]
    with open(f"{OUT}/offline_exp3_per_seed.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys)
        wtr.writeheader()
        for r in recs:
            wtr.writerow({k: r[k] for k in keys})
    print("wrote", len(recs), "rows")


if __name__ == "__main__":
    main()
