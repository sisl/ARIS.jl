#!/usr/bin/env python3
"""
Integrated log-error (ILE@1/B) and coverage for curve experiments.

For each threshold g, missing, zero, or non-finite estimates are replaced by
1/B, where B is the charged simulator-evaluation count for that method and
seed. Finite positive estimates are left unchanged.

    ILE      = mean |log P~(g) - log truth(g)|
    coverage = fraction of truth-positive grid points with finite positive estimates

The 1/B floor is used only for missing estimates; valid estimates are neither
raised to the floor nor upper-clamped. Alternative floors are reported only
as a sensitivity analysis.

Supports Experiments 1, 2, and 4. Experiment 3 is a single-threshold study and
is handled separately.

Usage:
    python paper/scripts/metrics.py exp2 exp4
    python paper/scripts/metrics.py --check exp2 exp4
    python paper/scripts/metrics.py --tables
    python paper/scripts/metrics.py exp1 --exp1-curves PATH
"""
import argparse, glob, json, math, os, statistics as st, sys

RESULTS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
FLOORS_APPENDIX = [1e-6, 1e-8]
SNAP_TOL = 1e-10
GROUP_ID = {"exp1": "exp1_kscaling", "exp2": "exp2_shell", "exp3": "exp3_tailrace", "exp4": "exp4_idm"}

METRIC_DEFINITION = {
    "rule": "P~ = P^ if finite and > 0, else 1/B; valid estimates never modified; no upper clamp",
    "ile": "mean over all grid points with truth > 0 of |log P~ - log truth|",
    "coverage": "fraction of grid points with a finite positive estimate",
    "floor": "1/B, where B is the charged simulator-evaluation count of the arm and seed",
    "appendix_floors_only": FLOORS_APPENDIX,
}

AMS_BUDGET = ("n_evals_all_configs",
              "all six oracle configurations are charged; the cost varies by seed with the number "
              "of AMS levels")
CURVE_GROUPS = {
    "exp2": {
        "cell": lambda r: f"d{r['d']}_B{r['budget']}",
        "arms": {
            "arm_a": ("n_evals_actual", "single adaptive run"),
            "arm_b": ("n_evals_actual", "every oracle candidate is charged (3 N per threshold)"),
            "arm_c": AMS_BUDGET,
        },
    },
    "exp4": {
        "cell": lambda r: "all",
        "arms": {
            "arm_a_primary": ("n_evals_actual", "single adaptive run"),
            "arm_b_perthreshold": ("n_evals_actual",
                                   "every oracle candidate is charged (2 N per threshold)"),
            "arm_c_ams": AMS_BUDGET,
        },
    },
}


def ile_and_coverage(points, floor):
    """points: iterable of (estimate, truth). Returns (ile, n_positive, n_grid)."""
    errs, npos, n = [], 0, 0
    for est, truth in points:
        if not (isinstance(truth, (int, float)) and truth > 0):
            continue
        n += 1
        good = isinstance(est, (int, float)) and math.isfinite(est) and est > 0
        if good:
            npos += 1
        errs.append(abs(math.log(est if good else floor) - math.log(truth)))
    return (st.mean(errs) if errs else None), npos, n


def med(v):
    v = [x for x in v if isinstance(x, (int, float)) and math.isfinite(x)]
    return st.median(v) if v else None

# ------------------------------------------------------------------------------------ 
def curve_metrics(exp):
    cfg = CURVE_GROUPS[exp]
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(RESULTS, exp, "raw", "*.json"))]
    rows = sorted((r for r in rows if r.get("ok")), key=lambda r: (cfg["cell"](r), r["seed"]))
    cells = {}
    for arm, (field, why) in cfg["arms"].items():
        buckets = {}
        for r in rows:
            buckets.setdefault(cfg["cell"](r), []).append(r)
        for cell, rs in buckets.items():
            per_seed = []
            for r in rs:
                a = r.get(arm)
                if not isinstance(a, dict) or "per_threshold" not in a:
                    continue
                B = a.get(field)
                if not B:
                    raise ValueError(f"{exp} {cell} {arm} seed {r['seed']}: no '{field}'")
                pts = [(p.get("estimate"), p.get("truth")) for p in a["per_threshold"]]
                ile, npos, n = ile_and_coverage(pts, 1.0 / B)
                alt = {f"{f:g}": ile_and_coverage(pts, f)[0] for f in FLOORS_APPENDIX}
                per_seed.append(dict(seed=r["seed"], B=B, ile=ile, n_pos=npos, n_grid=n,
                                     coverage=(npos / n if n else None), appendix=alt))
            if not per_seed:
                continue
            covs = [p["coverage"] for p in per_seed if p["coverage"] is not None]
            iles = [p["ile"] for p in per_seed if p["ile"] is not None]
            contrib = [p["ile"] for p in per_seed
                       if p["coverage"] and p["coverage"] > 0 and p["ile"] is not None]
            Bs = [p["B"] for p in per_seed]
            cells[arm if cell == "all" else f"{cell}|{arm}"] = {
                "cell": cell, "arm": arm,
                "B_actual_field": field, "B_actual_rationale": why,
                "B_actual_median": med(Bs), "B_actual_min": min(Bs), "B_actual_max": max(Bs),
                "B_actual_per_seed": Bs,
                "B_actual_constant_across_seeds": len(set(Bs)) == 1,
                "floor_primary_median": 1.0 / med(Bs),
                "n_seeds": len(per_seed),
                "ile_primary_median": med(iles),
                "ile_primary_per_seed": [p["ile"] for p in per_seed],
                "coverage_median": med(covs),
                "coverage_min": min(covs) if covs else None,
                "coverage_max": max(covs) if covs else None,
                "coverage_per_seed": [p["coverage"] for p in per_seed],
                "n_seeds_zero_coverage": sum(1 for c in covs if c == 0),
                "n_seeds_any_estimate": sum(1 for c in covs if c > 0),
                "n_seeds_full_coverage": sum(1 for c in covs if c >= 1.0),
                "ile_conditional_on_any_estimate_median": med(contrib),
                "n_grid": per_seed[0]["n_grid"],
                "appendix": {f"{f:g}": {"ile_median": med([p["appendix"][f"{f:g}"] for p in per_seed])}
                             for f in FLOORS_APPENDIX},
                "per_seed": per_seed,
            }
    return {"group": GROUP_ID[exp], "metric_definition": METRIC_DEFINITION,
            "budget_policy": {k: {"field": v[0], "rationale": v[1]} for k, v in cfg["arms"].items()},
            "cells": cells}

# ------------------------------------------------------------------------------------------
def cell_metrics(per_seed_points, n_evals_per_seed):
    """Exp1 cell: per_seed_points is a list (per seed) of lists of (estimate, truth)."""
    prim, cov, alt = [], [], {f: [] for f in FLOORS_APPENDIX}
    ngrid, floors = 0, []
    for pts, B in zip(per_seed_points, n_evals_per_seed):
        fl = 1.0 / B
        floors.append(fl)
        i, npos, n = ile_and_coverage(pts, fl)
        prim.append(i); cov.append(npos / n if n else None)
        ngrid = n
        for f in FLOORS_APPENDIX:
            alt[f].append(ile_and_coverage(pts, f)[0])
    covv = [c for c in cov if c is not None]
    contrib = [i for i, c in zip(prim, cov) if c is not None and c > 0]
    return {
        "ile_primary_median": med(prim), "ile_primary_per_seed": prim,
        "n_seeds_zero_coverage": sum(1 for c in covv if c == 0),
        "n_seeds_full_coverage": sum(1 for c in covv if c >= 1.0),
        "n_seeds_any_estimate": sum(1 for c in covv if c > 0),
        "ile_conditional_on_any_estimate_median": med(contrib),
        "coverage_median": med(cov), "coverage_per_seed": cov,
        "coverage_min": min(covv, default=None), "coverage_max": max(covv, default=None),
        "floor_primary": med(floors), "floors_per_seed": floors,
        "n_grid": ngrid, "n_seeds": len(prim),
        "appendix": {f"{f:g}": {"ile_median": med(alt[f])} for f in FLOORS_APPENDIX},
    }


def exp1_metrics(curves_path):
    rec = json.load(open(curves_path))
    nd = st.NormalDist()
    truthf = lambda g: nd.cdf(-(3.0 - g)) ** 2
    p_hi = nd.cdf(-3.0) ** 2

    def grid(K):
        lo, hi = math.log10(1e-3), math.log10(p_hi)
        gs = [3.0 + nd.inv_cdf(math.sqrt(10 ** (lo + (hi - lo) * i / (K - 1)))) for i in range(K)]
        return [0.0 if abs(g) < SNAP_TOL else g for g in gs]

    cells = {}
    for key, d in rec["cells"].items():
        K = d["K"]; G = grid(K)
        seeds = sorted(d["curves"].keys(), key=int)
        pts = [[(d["curves"][s][i], truthf(G[i])) for i in range(len(G))] for s in seeds]
        m = cell_metrics(pts, [1.0 / d["floor_budget"]] * len(seeds))
        m["arm"] = d["arm"]; m["K"] = K; m["seeds"] = [int(s) for s in seeds]
        cells[key] = m
    return {"group": GROUP_ID["exp1"], "endpoint_snapped": True, "snap_tolerance": SNAP_TOL,
            "cells": cells, "metric_definition": METRIC_DEFINITION}


# -----------------------------------------------------------------------------------------
def metrics_path(exp):
    return os.path.join(RESULTS, exp, "metrics.json")


def same(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present in only one file")
            else:
                same(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, (int, float)) and not isinstance(b, bool):
        if not math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0):
            out.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def num(x, sig=6):
    return "—" if x is None else f"{x:.{sig}g}"


def pct(x):
    return "—" if x is None else f"{100 * x:g}%"


def print_tables():
    m1 = json.load(open(metrics_path("exp1")))
    print("Exp1 (K-scaling): median ILE@1/B, median coverage")
    for key in sorted(m1["cells"], key=lambda k: (m1["cells"][k]["arm"], m1["cells"][k]["K"])):
        c = m1["cells"][key]
        print(f"  {c['arm']:<13} K={c['K']:<4} ILE {num(c['ile_primary_median'])}  "
              f"coverage {pct(c['coverage_median'])}")
    cem = json.load(open(os.path.join(RESULTS, "exp1", "cem_baseline", "metrics.json")))
    for key in sorted(cem["cells"], key=lambda k: int(k.rsplit("K", 1)[1])):
        print(f"  {'uncond. CEM':<13} K={key.rsplit('K', 1)[1]:<4} "
              f"ILE {num(med(cem['cells'][key]['ile']))}")
    for exp in ("exp2", "exp4"):
        m = json.load(open(metrics_path(exp)))
        print(f"\n{exp}: charged evals | median ILE@1/B | median coverage | full / any / zero")
        for key in sorted(m["cells"]):
            c = m["cells"][key]
            B = (f"{int(c['B_actual_median']):,}" if c["B_actual_constant_across_seeds"] else
                 f"{int(c['B_actual_median']):,} ({c['B_actual_min']:,}-{c['B_actual_max']:,})")
            print(f"  {key:<22} {B:>26}  {num(c['ile_primary_median']):>9}  "
                  f"{pct(c['coverage_median']):>7}  {c['n_seeds_full_coverage']}/{c['n_seeds']} "
                  f"{c['n_seeds_any_estimate']}/{c['n_seeds']} {c['n_seeds_zero_coverage']}/{c['n_seeds']}")
    s3 = json.load(open(os.path.join(RESULTS, "exp3", "summary.json")))
    print("\nexp3: all-three-lobe discovery | amplified | median P^/P | fallback-assisted runs")
    for key in sorted(s3["cells"]):
        c = s3["cells"][key]
        n = c["seed_accounting"]["n_completed"]
        print(f"  {key:<24} {c['found_all_three_lobes_count']}/{n}  "
              f"{c['regime_counts'].get('amplified', 0)}/{n}  {num(c['P_over_truth_median'], 3)}  "
              f"{c['fit_method_fallback_assisted_count']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("experiments", nargs="*", choices=["exp1", "exp2", "exp4"])
    ap.add_argument("--check", action="store_true", help="compare with the files; write nothing")
    ap.add_argument("--tables", action="store_true", help="print the result tables")
    ap.add_argument("--exp1-curves", help="endpoint curves from exp1_endpoint_curves.jl")
    args = ap.parse_args()
    if args.tables:
        print_tables()
        return 0
    status = 0
    for exp in args.experiments or ["exp2", "exp4"]:
        if exp == "exp1":
            if not args.exp1_curves:
                sys.exit("exp1 needs --exp1-curves (written from raw state by exp1_endpoint_curves.jl)")
            out = exp1_metrics(args.exp1_curves)
        else:
            out = curve_metrics(exp)
        p = metrics_path(exp)
        if args.check:
            diffs = same(out, json.load(open(p)))
            print(f"{exp}: {'matches ' + p if not diffs else f'{len(diffs)} differences'}")
            for d in diffs[:20]:
                print("   ", d)
            status |= bool(diffs)
        else:
            with open(p, "w") as f:
                json.dump(out, f, indent=2)
                f.write("\n")
            print(f"{exp}: wrote {p}")
    return status


if __name__ == "__main__":
    sys.exit(main())
