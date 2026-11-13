# Estimating failure probabilities

Training produces a buffer. [`is_estimate`](api-estimators.md#is_estimate) turns it into a number.

```julia
train!(cv, sys)
p̂ = is_estimate(cv.buffer; threshold = 0.0)
```

## Definition

<div class="aris-math" markdown="0">
P̂(γ) = (1/N<sub>total</sub>) Σ<sub>i</sub> 1{ρ<sub>i</sub> ≤ γ} · w<sub>i</sub>
<span class="aris-where">over every sample in the buffer, including the first prior batch; w<sub>i</sub> is the raw proposal-of-origin weight recorded at sampling time.</span>
</div>

Three consequences follow:

- **It is not self-normalised.** The denominator is the sample count, not Σw. Each weight already
  refers to the proposal that produced it.
- **It pools every batch.** Earlier, higher-threshold batches contribute. They mostly contribute
  zeros at a low γ, which is correct — they are part of the sample count.
- **The weights do not depend on γ.** They were fixed when the sample was drawn. Changing
  `threshold` changes only the indicator, which is why one run answers many thresholds
  ([Working with multiple thresholds](multiple-thresholds.md)).

## Buffer contents

```julia
cv.buffer[:x]        # xdim × N   inputs
cv.buffer[:ρ]        # N          realised robustness
cv.buffer[:w]        # N          raw importance weights
cv.buffer[:r]        # N          conditioning value in force for that sample (+Inf for batch 1)
cv.buffer[:s]        # sdim × depth × N   simulated states
cv.buffer[:logpdf]   # N          log proposal density at the sample
length(cv.buffer)    # number of samples stored
```

## Diagnostics

Check these before reporting an estimate. A rare-event estimator typically fails by returning an
incorrect value rather than by raising an error.

```julia
using Statistics

ρs, ws = cv.buffer[:ρ], cv.buffer[:w]
γ = 0.0
hit = ρs .<= γ

println("estimate     ", is_estimate(cv.buffer; threshold = γ))
println("hits         ", count(hit), " / ", length(ρs))
println("ESS (all)    ", round(ess(ws); digits = 1))
println("ESS (hits)   ", count(hit) == 0 ? 0.0 : round(ess(ws[hit]); digits = 1))
println("max w share  ", round(maximum(ws[hit]; init = 0.0) / sum(ws[hit]; init = 1.0); digits = 3))
println("schedule to  ", minimum(cv.buffer[:r]))
println("deepest ρ    ", minimum(ρs))
```

| signal | what it means |
|---|---|
| **hits = 0** | the estimate is exactly 0. Not a small probability — no information. Lengthen the schedule or increase the budget |
| **ESS over hits is small** (single digits) | a handful of samples carry the estimate. The number is a draw from a high-variance estimator; do not quote it alone |
| **max weight share near 1** | one sample *is* the estimate |
| **schedule stopped above γ** | the proposal was never aimed at this region. See [reach](multiple-thresholds.md#reach) |
| **EM fell back** (`get_fit_stats()`) | the conditional model is not reliably fitted. See [fit diagnostics](api-models.md#fit-diagnostics) |

!!! warning "Effective sample size at the target"
    Overall ESS can look healthy while the samples that actually land in the failure region are
    dominated by one or two weights. On a hard problem the target ESS can be in the single
    digits even when the overall figure looks fine, so check it explicitly and report it.

## Variance and error bars

A single run gives a single draw. The package does not return a confidence interval, and one
should not be inferred from a single trained buffer: the samples are neither independent nor
identically weighted, so the usual weighted-variance formula understates the true spread.

The honest approach is **repeated independent runs**:

```julia
using Statistics

function replicate(sys; seeds = 1:10, γ = 0.0)
    map(seeds) do seed
        cv = ConditionalValidation(
            model = JointModel(Normal(), ConditionalEMGMM(2, 3, 3)),
            xdim = 2, depth = 1, sdim = 2,
            n_samples = 1000, n_iter = 25,
            sampling_strategy = QuantileSampling(0.2; target = γ),
        )
        train!(cv, sys; rng = MersenneTwister(seed))
        is_estimate(cv.buffer; threshold = γ)
    end
end

p̂s = replicate(sys)
println("median ", median(p̂s), "   spread ", extrema(p̂s))
```

Report the spread across seeds. Because these estimates range over orders of magnitude,
summarise them on a log scale: absolute log error against a known truth is the informative
measure, and a run that returns zero has to be handled explicitly rather than averaged in.

## Unbiasedness

!!! warning
    Importance sampling with proposal-of-origin weights is unbiased when the proposal covers the
    failure region: each sample is weighted against the density that actually produced it.

    Pooling across adaptive batches carries no such guarantee. `train!` fits each proposal from
    the samples the previous ones produced, so the batches are not independent draws. Treat a
    pooled estimate as a point estimate with unquantified bias, and report the spread across
    seeds rather than a single run.

## Comparison with baselines

All three baselines target ρ ≤ 0 and are described in
[Estimators and baselines](api-estimators.md).

```julia
# Unconditional cross-entropy method: same train!/is_estimate shape.
# Note the model is UNCONDITIONAL — an EMGMM over inputs alone, not a JointModel.
cem = CrossEntropyMethod(
    model = EMGMM(2, 2),              # 2 components over 2 input dimensions
    xdim = 2, depth = 1, sdim = 2,
    n_samples = 1000, n_iter = 25, n_elite = 100,
)
train!(cem, sys)
is_estimate(cem.buffer; threshold = 0.0)

# Adaptive multilevel splitting returns its own estimate.
p̂_ams, levels = ams(sys)

# Population Monte Carlo returns the fitted means, the kernel covariance, and its samples.
μ, Σ, samples = pmc(sys; dₓ = 2)
```

!!! note "Budget accounting"
    A comparison is only meaningful if every method is charged for every simulator evaluation it
    made, including the ones spent on tuning. `ams` runs a variable number of levels and stops
    when it reaches the target, so its cost varies per seed and has to be measured, not assumed.

## See also

- [Working with multiple thresholds](multiple-thresholds.md): reading a curve, and its caveats.
- [Estimators and baselines](api-estimators.md): signatures and return shapes.
