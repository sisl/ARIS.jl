# Reproducing experiments

Everything needed to reproduce the paper's reported results is under
[`paper/`](https://github.com/sisl/ARIS.jl/tree/main/paper):
drivers, configurations, seed-level results, metric and diagnostic scripts, figure generators and
figures.

!!! info "Authoritative source"
    [`paper/README.md`](https://github.com/sisl/ARIS.jl/blob/main/paper/README.md) is the
    reference for this material. This page summarises it and points at the parts that are easy to
    get wrong.

Run every command from the repository root.

## Environments

Two, and they are not interchangeable:

| environment | used by |
|---|---|
| `Project.toml` / `Manifest.toml` | Exp1, Exp2, Exp3, unconditional CEM |
| `paper/envs/idm/` | **Exp4 only** |

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=paper/envs/idm -e 'using Pkg; Pkg.instantiate()'
```

!!! note
    The Exp4 environment includes Git-based dependencies and may take noticeably longer to
    instantiate on a fresh machine. The Python figure and metric scripts need `numpy` and
    `matplotlib`.

## Nothing needs to be rerun

**None of the reported numbers requires a rerun.** The seed-level results are in the repository,
and the metric and figure scripts recompute the reported values from them. Start there.

## Reproducing metrics and tables

No simulation is involved.

```bash
python paper/scripts/metrics.py --check exp2 exp4
```

Recomputes the Exp2 and Exp4 `metrics.json` files from `raw/*.json` and compares them against the
released files. **Without `--check`, the script rewrites them.**

```bash
python paper/scripts/metrics.py --tables
```

Prints the result tables for all four experiments.

```bash
python paper/scripts/exp3_aggregates.py
```

Rebuilds `results/exp3/diagnostics/exp3_derived_aggregates.json` from the per-seed diagnostics.

Exp1's `metrics.json` needs the raw-state bundle — see
[below](#artifacts-that-need-the-raw-state-bundle).

## Reproducing figures

```bash
python paper/figures/make_exp1_figure.py --main-out OUTDIR --appendix-out OUTDIR
python paper/figures/make_exp2_figure.py --out-dir OUTDIR
python paper/figures/make_exp3_figure.py --out-dir OUTDIR
python paper/figures/make_exp4_figure.py --out-dir OUTDIR
```

Each generator reports its input files and validates plotted values against the stored results.

!!! note
    Regenerated PDF and SVG metadata may differ between runs even when the drawn content matches,
    so a byte comparison against the released figures is not the right check. The generators'
    own value validation is.

## Rerunning the experiments

Drivers write to `paper/rerun/<group>/` by default, so a rerun does **not** overwrite
`paper/results/`. Set `ARIS_OUTROOT` to choose another output root.

```bash
julia --project=.              paper/experiments/drivers/exp1_kscaling.jl [nseeds] [budget] [nodes]
julia --project=.              paper/experiments/drivers/exp2_shell.jl [nseeds]
julia --project=.              paper/experiments/drivers/exp3_tailrace.jl [nseeds] [with_sigma_arm(0|1)]
julia --project=paper/envs/idm paper/experiments/drivers/exp4_idm.jl [nseeds] [arms] [nodes]
julia --project=.              paper/experiments/cem_baseline/run_cem_arm.jl [--seeds 1,2,...] [--out DIR]
```

Note the different environment on the Exp4 line.

## What is where

```text
experiments/
  drivers/            exp{1,2,3,4}_*.jl, shared metric, raw-state and provenance modules,
                      and exp1_endpoint_curves.jl
  cem_baseline/       Exp1 unconditional-CEM baseline and its evaluator
  *.jl                balance-MIS, AMS, conditional-SGM, eta-mixture, EVT, sigma-smoothing
                      modules and the IDM reference generator
scripts/              metrics.py, exp3_aggregates.py, raw-state diagnostic generators
figures/
  make_exp*_figure.py figure generators
  output/             paper figures in PDF and SVG
results/
  reference/idm_hist.json   26M-rollout IDM reference histogram (Exp4)
  exp1/ ... exp4/           config.json, meta.json, raw/, summary.json, metrics.json, diagnostics/
envs/idm/             Exp4 Julia environment
manuscript/           snapshot of the manuscript source
```

Problem definitions live in
[`examples/problems/`](https://github.com/sisl/ARIS.jl/tree/main/examples/problems), not under
`paper/` — see [Example problems](problems.md).

### Result files

Each `results/expN/` holds `config.json` (run configuration), `meta.json` (Julia version, thread
counts, package versions, active environment), `raw/*.json` (one record per seed, or per cell for
Exp2 and Exp3), `summary.json`, and `metrics.json` for Exp1, Exp2 and Exp4.

!!! warning "`summary.json` is not the reported metric"
    Its `ile*` fields use only grid points with available estimates. The reported ILE is in
    `metrics.json`. Exp3 is a single-threshold study and reports P-hat/P from `summary.json`
    instead.

`diagnostics/` holds derived diagnostics supporting specific claims: per-seed sampling depth and
schedule reach (Exp2); per-seed mode statistics and cell aggregates including the clean-only
sensitivity aggregate (Exp3); pooled balance-MIS estimates, proposal-of-origin estimates, hit
counts, weight concentration and adaptive-schedule information (Exp4); and AMS level trajectories
per oracle configuration (Exp4).

## The metric

<div class="aris-math" markdown="0">
P̃(g) = P̂(g) if finite and &gt; 0, otherwise 1/B
<span class="aris-where">B is the charged simulator-evaluation count for that method and seed.</span>
</div>

<div class="aris-math" markdown="0">
ILE = mean over grid points with truth &gt; 0 of | log P̃(g) − log truth(g) |
<span class="aris-where">coverage = #{g : P̂(g) finite and &gt; 0} / #{g : truth(g) &gt; 0}</span>
</div>

Oracle-tuned baselines are charged for every candidate they evaluate. The 1/B floor replaces only
missing, zero or non-finite estimates; finite positive estimates are unchanged and no upper clamp
is applied. `metrics.json` also reports sensitivity for floors of 10⁻⁶ and 10⁻⁸.

## Artifacts that need the raw-state bundle

The drivers write per-sample raw-state bundles (samples, robustness values, weights, proposal
checkpoints). **These are not included in the release.** The derived artifacts and their
generators are:

| artifact | generator |
|---|---|
| `results/exp1/metrics.json` | `experiments/drivers/exp1_endpoint_curves.jl`, then `scripts/metrics.py exp1 --exp1-curves FILE` |
| `results/exp1/cem_baseline/metrics.json` | `experiments/cem_baseline/cem_metrics.jl` |
| `results/exp2/diagnostics/offline_exp2_diagnostics.json` | `scripts/offline_exp2_diagnostics.jl` |
| `results/exp3/diagnostics/offline_exp3_per_seed.json` | `scripts/offline_exp3_reconstruct.py` |
| `results/exp4/diagnostics/reindex/seed*.json` | `scripts/exp4_reindex.jl` |

The 26-million-rollout IDM reference was generated by `experiments/idm_reference.jl` and is stored
at `results/reference/idm_hist.json`.

## Third-party dependency

Exp4 uses `sisl/StructuredGaussianMixtures.jl` (MIT) with numerical-stability changes to the
MAP-EM fit. `paper/envs/idm` fetches it as a Git dependency from a fork, pinned to a specific
commit; the Manifest also records the commit's Git tree hash, which Pkg verifies on download.
[`NOTICE`](https://github.com/sisl/ARIS.jl/blob/main/NOTICE) records the upstream revision and the
local changes.

## Experiment names

The package is `ARIS`. The identifiers `exp1_kscaling`, `exp2_shell`, `exp3_tailrace` and
`exp4_idm` appear as `experiment_group` in `meta.json`, as `group` in `metrics.json`, and as the
default rerun directory names.

## Next

- [Paper and method](paper.md): what the results say, and their caveats.
- [Citation](citation.md).
