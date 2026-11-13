# Working with multiple thresholds

A single trained run yields estimates at any number of thresholds. This page covers how to obtain
them and the conditions under which they are valid.

## Estimating a threshold curve

```julia
using Printf

train!(cv, sys)

for γ in 2.0:-0.25:0.0
    @printf("γ = %5.2f   P̂ = %.4g\n", γ, is_estimate(cv.buffer; threshold = γ))
end
```

No refitting, no further simulator calls. The weights were fixed at sampling time and do not
depend on γ; only the indicator `ρᵢ ≤ γ` moves
([Methodology](methodology.md#the-estimator)).

On the toy system, where the truth is Φ(γ − γ₀)², you can check it directly:

```julia
for γ in 2.0:-0.5:0.0
    @printf("γ = %4.1f   P̂ = %-10.4g truth = %.4g\n",
            γ, is_estimate(cv.buffer; threshold = γ), cdf(Normal(), γ - 3.0)^2)
end
```

## Conditions for validity

A number comes back for **any** γ you ask about. Three separate conditions decide whether it means
anything, and none of them is checked for you.

### Reach

The training schedule descends from the prior toward `target` and stops where the budget runs out.
Thresholds below that point are extrapolation: the conditional model was never shown that region,
and the proposal was never aimed at it.

```julia
r_reached = minimum(cv.buffer[:r])     # deepest conditioning value actually used
ρ_deepest = minimum(cv.buffer[:ρ])     # deepest robustness actually observed

γ < r_reached && @warn "below the schedule's reach — extrapolating" γ r_reached
```

!!! warning "Reach and availability"
    `is_estimate` returns a number at any threshold, whether or not the schedule reached it, so
    "an estimate came back" says nothing about reach. If you report a curve, say which points the
    schedule actually reached and which are extrapolation.

### Support

The conditional model q<sub>θ</sub>(x | r) must cover the inputs that actually produce robustness
r. A mixture that has collapsed onto one failure mode will give confident underestimates at
*every* threshold — the error is in the model, not in any particular γ.

```julia
get_fit_stats()   # :em, :kmeans_fallback, :em_trial_error, :small_det, :small_mix, ...
```

Non-zero `:kmeans_fallback` or `:degenerate_fit_guard` means some iterations did not get a real EM
fit. Call `reset_fit_stats!()` before a run to scope the counters to it.

### Weight degeneracy

Each threshold has its own effective sample size, over the samples that land in its failure region.
Deep thresholds are supported by progressively fewer samples.

```julia
function threshold_report(buffer, γs)
    ρs, ws = buffer[:ρ], buffer[:w]
    r_reached = minimum(buffer[:r])
    for γ in γs
        hit = ρs .<= γ
        n = count(hit)
        @printf("γ = %6.3f  P̂ = %-11.4g hits = %5d  ESS = %7.1f  %s\n",
                γ, is_estimate(buffer; threshold = γ), n,
                n == 0 ? 0.0 : ess(ws[hit]),
                γ < r_reached ? "BEYOND REACH" : "")
    end
end

threshold_report(cv.buffer, 2.0:-0.25:0.0)
```

A row with few hits or single-digit ESS is a point you should not plot without saying so.

## Monotonicity

P(γ) is non-increasing in γ, and the estimator respects this by construction: the hit set
{ρᵢ ≤ γ} only grows as γ grows, and the weights are non-negative. So the estimated curve is
monotone **as a realisation** — which means monotonicity is *not* evidence that the curve is
correct. A collapsed proposal produces a perfectly monotone, perfectly wrong curve.

## Scope of `is_estimate`

!!! info "Scope"
    The package gives you `is_estimate(buffer; threshold)`, which is the proposal-of-origin
    estimator evaluated at whatever threshold you pass. Everything on this page is built from that
    one call plus the buffer's columns.

    Each weight refers only to the proposal that produced that sample, so the pooled average is
    a proposal-of-origin estimate rather than a multiple-importance-sampling one over the mixture
    of all proposals. That is a deliberate, documented limit of `is_estimate`, not an
    approximation it corrects for internally.

## Procedure

1. Set `target` to the deepest threshold you care about.
2. Train, with a `log_callback` that prints the conditioning value and hit count per iteration
   ([Training a proposal](training.md#logging-a-run)).
3. Confirm the schedule reached the target: `minimum(cv.buffer[:r])`.
4. Run `threshold_report` over your grid and drop, or mark, rows that are beyond reach or have
   negligible ESS.
5. Repeat over several seeds and report the spread, not a single run
   ([Estimating failure probabilities](estimating.md#variance-and-error-bars)).

## See also

- [Estimating failure probabilities](estimating.md): diagnostics in more detail.
