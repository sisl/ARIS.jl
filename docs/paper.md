# Paper and method

**Amortized Rare-Event Importance Sampling**<br>
Lea M. Hadzic, Harrison Delecki, and Mykel J. Kochenderfer — Stanford University

!!! note "Publication status"
    The manuscript is not yet published, and no venue, volume or DOI has been assigned. See
    [Citation](citation.md) for how to cite it in the meantime. The manuscript source is under
    [`paper/manuscript/`](https://github.com/sisl/ARIS.jl/tree/main/paper/manuscript).

The repository contains both a reusable library and the code behind this paper. This page says
which is which, and what the paper's results rest on.

## Package versus paper code

!!! warning "The boundary matters"
    Everything documented under [API reference](api-systems.md) is the **package** — what
    `using ARIS` gives you, in [`src/`](https://github.com/sisl/ARIS.jl/tree/main/src).

    The paper's experiment drivers, its **multi-threshold estimator**, its metric and diagnostic
    scripts, and its figure generators live under
    [`paper/`](https://github.com/sisl/ARIS.jl/tree/main/paper) and are **not** exposed as package
    API.

The most consequential instance:

| | package | paper |
|---|---|---|
| estimator | [`is_estimate`](api-estimators.md#is_estimate) — pooled **proposal-of-origin** | pooled **balance-MIS** over all adaptive batches |
| where | `src/utils.jl` | `paper/experiments/` |

Balance-MIS re-weights each sample against the mixture of *all* proposals used during the run,
rather than only the one it came from. If you are reproducing the paper's reported curves, use the
paper's drivers ([Reproducing experiments](reproduce.md)). If you are applying the method to your
own problem, `is_estimate` is the supported path.

## The method, in one paragraph

ARIS fits a joint model q(x, r) = q<sub>ψ</sub>(r) · q<sub>θ</sub>(x | r) over system inputs and
the robustness they produce, using an adaptive schedule that walks a conditioning threshold down
from the prior toward the target. Because the input model is conditional on robustness, truncating
q<sub>ψ</sub> at any threshold γ yields an importance-sampling proposal aimed at {ρ ≤ γ} — so one
fit serves a whole grid of thresholds without refitting or further simulator calls. See
[Amortized importance sampling](amortization.md) and [Methodology](methodology.md).

## The experiments

| experiment | problem | question |
|---|---|---|
| **Exp1** — m-scaling | [`toy.jl`](problems.md#unimodaltoy-the-corner) | how the estimate behaves as budget scales, against a closed form; also the unconditional-CEM comparison |
| **Exp2** — radial shell | [`radial_shell.jl`](problems.md#radialshell-high-dimensions) | a 20-dimensional, direction-free failure region; discovery and schedule reach |
| **Exp3** — multimodal tail | [`tailrace.jl`](problems.md#tailracetoy-the-multimodal-tail) | conditioning versus an unconditional baseline when failure is multimodal |
| **Exp4** — IDM driving | [`crosswalk.jl`](problems.md#adversarialcrosswalk-the-realistic-one) | the full amortization claim on a 150-dimensional simulator, over a threshold grid |

## The metric

Estimates span orders of magnitude, so accuracy is measured on a log scale:

<div class="aris-math" markdown="0">
P̃(g) = P̂(g) if P̂(g) is finite and &gt; 0,&nbsp; otherwise 1/B
<span class="aris-where">B is the charged simulator-evaluation count for that method and seed.</span>
</div>

<div class="aris-math" markdown="0">
ILE = mean over grid points with truth &gt; 0 of | log P̃(g) − log truth(g) |
<span class="aris-where">coverage = #{g : P̂(g) finite and &gt; 0} / #{g : truth(g) &gt; 0}</span>
</div>

The 1/B floor replaces only missing, zero or non-finite estimates; finite positive estimates are
left alone and no upper clamp is applied. `metrics.json` also reports sensitivity under floors of
10⁻⁶ and 10⁻⁸.

!!! info "Oracle-tuned baselines are charged in full"
    A baseline that is tuned by trying candidates is charged for **every** candidate it evaluated,
    not just the one finally selected. Without that, a tuned baseline gets free information.

## Reported results, Experiment 4

The threshold grid is {0.434, 0.38, 0.32, 0.26, 0.20, 0.14, 0.08, 0} over 10 seeds.

| Method | Charged evaluations | ILE@1/B | Median coverage | Full | Any | Zero |
|---|---:|---:|---:|---:|---:|---:|
| Amortized | 25,000 | 0.48376 | 100% | 10/10 | 10/10 | 0/10 |
| Per-threshold | 48,000 | 3.49781 | 68.75% | 0/10 | 10/10 | 0/10 |
| AMS | 101,024–125,936 (median 115,018) | 0.46460 | 87.5% | 5/10 | 10/10 | 0/10 |

!!! note "Read this table carefully"
    **Amortized and AMS have similar median ILE.** The amortized method's claim here is *cost*,
    not accuracy: AMS uses about 4.6× as many simulator evaluations at the median, and its cost
    varies by seed because it stops once it reaches the target.

    **Coverage is not a pure support measure.** For the per-threshold baseline, zero selected
    estimates can arise because the relative-error oracle may prefer zero to a sufficiently large
    positive overshoot.

    **Target ESS has median 5.8.** All ten amortized estimates at the target threshold are finite,
    but they rest on few effective samples — see
    [diagnostics](estimating.md#diagnostics).

    Reference probabilities come from a 26-million-rollout IDM histogram evaluated at exact
    threshold boundaries.

## Stated caveats

The paper's protocol notes record these; they are repeated here because they bear on how the
results should be read.

- **Estimator.** The amortized curve uses pooled balance-MIS over all adaptive batches.
  Proposal-of-origin adaptive importance sampling is unbiased under the conditional-history and
  support assumptions stated in the paper. **No unbiasedness guarantee is established** for the
  pooled adaptive balance-MIS estimator, because later proposals depend on earlier samples.
- **Exp1.** The final grid point is the target γ = 0. Floating-point grid construction places that
  endpoint slightly below zero, so it is snapped to zero before evaluation. The amortized endpoint
  is rebuilt from the raw-state bundle.
- **Exp2.** The amortized method reports a pooled estimate at a grid threshold **only when its
  adaptive schedule has reached it**, so its coverage measures schedule reach rather than
  estimator availability. The per-threshold baseline uses a distinct deterministic random-number
  stream per threshold and candidate.
- **Exp3.** The conditioned method fits the accumulated sample buffer, whereas the unconditional
  baseline fits the current batch's elites — so the comparison evaluates two complete methods
  rather than isolating conditioning alone. Three baseline runs (at c2 = 16, 18 and 23) used the
  EMGMM fallback path; they are included in the reported aggregates, with clean-only sensitivity
  aggregates reported alongside.
- **Exp4.** AMS uses 2.38/√d random-walk-Metropolis scaling. Grid estimates use exact adaptive
  levels where available and log-linear interpolation otherwise.

## Optional code paths

The library includes normalizing-flow and policy/MDP integrations behind the `CGV_ENABLE_FLOW` and
`CGV_ENABLE_POLICY` gates. Both are disabled by default and **none of the paper experiments uses
them** ([Installation](install.md#optional-integrations)).

## Raw-state policy

The drivers also write per-sample raw-state bundles — samples, robustness values, weights and
proposal checkpoints. **These are not included in the release.** Derived artifacts and the scripts
that generate them from the bundles are included; see
[Reproducing experiments](reproduce.md#artifacts-that-need-the-raw-state-bundle).

## Next

- [Reproducing experiments](reproduce.md): the commands.
- [Citation](citation.md): how to cite the work.
