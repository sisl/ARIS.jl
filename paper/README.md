# Paper reproduction

This directory contains the experiment drivers, configurations, seed-level results, metric and
diagnostic scripts, figure generators, and figures needed to reproduce the paper's reported results.
The reusable library lives in [`../src`](../src).

## Layout

```text
experiments/
  drivers/                  exp{1,2,3,4}_*.jl, shared metric, raw-state, and provenance modules,
                            and exp1_endpoint_curves.jl
  cem_baseline/             Exp1 unconditional-CEM baseline and its evaluator
  *.jl                      balance-MIS, AMS, conditional-SGM, eta-mixture, EVT, sigma-smoothing
                            modules and the IDM reference generator
scripts/                    metrics.py, exp3_aggregates.py, and raw-state diagnostic generators
figures/
  make_exp*_figure.py       figure generators
  output/                   paper figures in PDF and SVG
results/
  reference/idm_hist.json   26M-rollout IDM reference histogram (Exp4)
  exp1/ ... exp4/           config.json, meta.json, raw/, summary.json, metrics.json, diagnostics/
envs/idm/                   Exp4 Julia environment
manuscript/                 snapshot of the canonical Overleaf manuscript source; build with
                            pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Problem definitions are in [`../examples/problems`](../examples/problems): `toy.jl` (Exp1),
`radial_shell.jl` (Exp2), `tailrace.jl` (Exp3), and `crosswalk.jl` (Exp4).

| Experiment | Driver | Results | Figure |
|---|---|---|---|
| Exp1, m-scaling | `experiments/drivers/exp1_kscaling.jl` | `results/exp1/` | `exp1_kscaling`, `exp1_kscaling_curves` |
| Exp1, unconditional CEM | `experiments/cem_baseline/run_cem_arm.jl` | `results/exp1/cem_baseline/` | same Exp1 figures |
| Exp2, radial shell | `experiments/drivers/exp2_shell.jl` | `results/exp2/` | `exp2_shell_discovery` |
| Exp3, multimodal tail | `experiments/drivers/exp3_tailrace.jl` | `results/exp3/` | `exp3_conditioning_modes` |
| Exp4, IDM driving | `experiments/drivers/exp4_idm.jl` | `results/exp4/` | `exp4_driving_curve` |

## Result files

Each `results/expN/` directory contains:

- `config.json` — run configuration;
- `meta.json` — Julia version, thread counts, package versions, and active environment;
- `raw/*.json` — one record per seed, or per cell for Exp2 and Exp3;
- `summary.json` — driver summary; its `ile*` fields use only grid points with available estimates
  and are not the reported ILE metric;
- `metrics.json` (Exp1, Exp2, Exp4) — reported ILE@1/B, coverage, and charged budgets. Exp3 is a
  single-threshold study and reports P-hat/P from `summary.json`;
- `diagnostics/` — derived diagnostics used to support statements in the paper:
  - `exp2/diagnostics/offline_exp2_diagnostics.json`: per-seed sampling depth and schedule reach;
  - `exp3/diagnostics/offline_exp3_per_seed.json` and
    `exp3_derived_aggregates.json`: per-seed mode statistics and cell aggregates, including the
    clean-only sensitivity aggregate;
  - `exp4/diagnostics/reindex/seed*.json`: pooled balance-MIS estimates, proposal-of-origin
    estimates, hit counts, weight concentration, and adaptive-schedule information;
  - `exp4/diagnostics/ams_levels/seed*.json`: AMS level trajectories for each oracle
    configuration.

## Metric

```text
P~(g) = P^(g)   if P^(g) is finite and > 0
      = 1/B     otherwise

ILE      = mean over all grid points with truth > 0 of |log P~(g) - log truth(g)|
coverage = #{g : P^(g) finite and > 0} / #{g : truth(g) > 0}
```

`B` is the charged simulator-evaluation count for that method and seed. Oracle-tuned baselines are
charged for every candidate they evaluate. The `1/B` floor replaces only missing, zero, or
non-finite estimates; finite positive estimates are left unchanged, and no upper clamp is applied.
`metrics.json` also reports sensitivity results for floors of `1e-6` and `1e-8`.

## Protocol notes

**Estimator.** The amortized curve uses pooled balance-MIS over all adaptive batches.
Proposal-of-origin adaptive importance sampling is unbiased under the conditional-history and
support assumptions stated in the paper. No unbiasedness guarantee is established for the pooled
adaptive balance-MIS estimator because later proposals depend on earlier samples.

**Exp1.** The final grid point is the target `gamma = 0`. Floating-point grid construction places
this endpoint slightly below zero, so it is snapped to zero before evaluation. The amortized endpoint is
rebuilt from the raw-state bundle by `experiments/drivers/exp1_endpoint_curves.jl`.

**Exp2.** The amortized method reports a pooled estimate at a grid threshold only when its adaptive
schedule has reached that threshold. Its coverage therefore measures schedule reach rather than
estimator availability. The per-threshold baseline uses a distinct deterministic random-number
stream for each threshold and candidate.

**Exp3.** The conditioned method fits the accumulated sample buffer, whereas the unconditional
baseline fits the current batch's elites. The comparison therefore evaluates the two full methods
rather than isolating conditioning alone. Three unconditional-baseline runs, one each at `c2 = 16`,
`18`, and `23`, use the EMGMM fallback path; they are included in the reported aggregates, with
clean-only sensitivity aggregates in `diagnostics/exp3_derived_aggregates.json`.

**Exp4.** The threshold grid is `{0.434, 0.38, 0.32, 0.26, 0.20, 0.14, 0.08, 0}` over 10 seeds.

| Method | Charged evaluations | ILE@1/B | Median coverage | Full | Any | Zero |
|---|---:|---:|---:|---:|---:|---:|
| Amortized | 25,000 | 0.48376 | 100% | 10/10 | 10/10 | 0/10 |
| Per-threshold | 48,000 | 3.49781 | 68.75% | 0/10 | 10/10 | 0/10 |
| AMS | 101,024–125,936 (median 115,018) | 0.46460 | 87.5% | 5/10 | 10/10 | 0/10 |

- *Amortized.* Estimates are reported at every grid threshold, while adaptive schedule reach is
  recorded separately. At the target threshold, all ten estimates are finite; target ESS has median
  5.8.
- *Per-threshold.* All candidate evaluations are charged. Zero selected estimates can arise because
  the relative-error oracle may prefer zero to a sufficiently large positive overshoot, so coverage
  is not purely a support measure.
- *AMS.* Uses `2.38/sqrt(d)` random-walk-Metropolis scaling. Grid estimates use exact adaptive
  levels when available and otherwise log-linear interpolation. Charged cost varies by seed because
  AMS stops after reaching the target.
- Amortized and AMS have similar median ILE; AMS uses about 4.6x as many simulator evaluations at
  the median.
- *Reference.* Probabilities are computed from the 26-million-rollout histogram at exact threshold
  boundaries.

## Environments

Run all commands from the repository root.

| Environment | Used by |
|---|---|
| `Project.toml` / `Manifest.toml` | Exp1, Exp2, Exp3, unconditional CEM |
| `paper/envs/idm/Project.toml` / `Manifest.toml` | Exp4 |

Instantiate the root environment with:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Instantiate the Exp4 environment with:

```bash
julia --project=paper/envs/idm -e 'using Pkg; Pkg.instantiate()'
```

*Note that the Exp4 environment includes Git-based dependencies and may take longer to instantiate on a fresh machine.

The Python figure and metric scripts require `numpy` and `matplotlib`.

## Reproducing metrics and tables

No simulation is involved.

```bash
python paper/scripts/metrics.py --check exp2 exp4
```

This recomputes the Exp2 and Exp4 `metrics.json` files from `raw/*.json` and compares them with the
released files. Without `--check`, the script rewrites them. Exp1's `metrics.json` requires the
raw-state bundle; see [Raw-state policy](#raw-state-policy).

```bash
python paper/scripts/metrics.py --tables
```

This prints the result tables for all four experiments.

```bash
python paper/scripts/exp3_aggregates.py
```

This rebuilds `results/exp3/diagnostics/exp3_derived_aggregates.json` from the per-seed diagnostics.

## Reproducing figures

```bash
python paper/figures/make_exp1_figure.py --main-out OUTDIR --appendix-out OUTDIR
```

```bash
python paper/figures/make_exp2_figure.py --out-dir OUTDIR
```

```bash
python paper/figures/make_exp3_figure.py --out-dir OUTDIR
```

```bash
python paper/figures/make_exp4_figure.py --out-dir OUTDIR
```

Each figure generator reports its input files and validates plotted values against the stored
results. Regenerated PDF/SVG metadata may differ between runs even when the drawn content matches.

## Rerunning the experiments

None of the reported numbers requires a rerun. Drivers write to `paper/rerun/<group>/` by default,
so reruns do not overwrite `paper/results/`. Set `ARIS_OUTROOT` to choose another output root.

```bash
julia --project=.              paper/experiments/drivers/exp1_kscaling.jl [nseeds] [budget] [nodes]
julia --project=.              paper/experiments/drivers/exp2_shell.jl [nseeds]
julia --project=.              paper/experiments/drivers/exp3_tailrace.jl [nseeds] [with_sigma_arm(0|1)]
julia --project=paper/envs/idm paper/experiments/drivers/exp4_idm.jl [nseeds] [arms] [nodes]
julia --project=.              paper/experiments/cem_baseline/run_cem_arm.jl [--seeds 1,2,...] [--out DIR]
```

The library includes optional flow and policy/MDP integrations, both disabled by default. None of
the paper experiments uses them.

## Raw-state policy

The drivers also write per-sample raw-state bundles containing samples, robustness values, weights,
and proposal checkpoints. These bundles are not included in the release. Derived artifacts and the
scripts used to generate them are included below:

| Artifact | Generator |
|---|---|
| `results/exp1/metrics.json` | `experiments/drivers/exp1_endpoint_curves.jl`, then `scripts/metrics.py exp1 --exp1-curves FILE` |
| `results/exp1/cem_baseline/metrics.json` | `experiments/cem_baseline/cem_metrics.jl` |
| `results/exp2/diagnostics/offline_exp2_diagnostics.json` | `scripts/offline_exp2_diagnostics.jl` |
| `results/exp3/diagnostics/offline_exp3_per_seed.json` | `scripts/offline_exp3_reconstruct.py` |
| `results/exp4/diagnostics/reindex/seed*.json` | `scripts/exp4_reindex.jl` |

The 26-million-rollout IDM reference was generated by `experiments/idm_reference.jl` and is stored
in `results/reference/idm_hist.json`.

## Third-party dependency

Exp4 uses `sisl/StructuredGaussianMixtures.jl` (MIT) with numerical-stability changes used by the
MAP-EM fit. `envs/idm` fetches it as a Git dependency from the fork
`https://github.com/lea-m-hadzic/StructuredGaussianMixtures.jl`, pinned to commit
`182b322e8f58b54c28c8bebf60466c047839c6af` (upstream `aef6158` plus one local commit). The
Manifest also records the commit's Git tree hash, which Pkg verifies on download.
[`../NOTICE`](../NOTICE) records the upstream revision and local changes.

## Names

The Julia package is `ARIS`. Experiment identifiers `exp1_kscaling`, `exp2_shell`,
`exp3_tailrace`, and `exp4_idm` appear as `experiment_group` in `meta.json`, as `group` in
`metrics.json`, and as the default rerun directory names.
