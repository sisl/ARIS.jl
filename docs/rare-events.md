# Rare-event estimation

## Problem statement

A system takes an input x drawn from a known distribution p, runs, and either fails or does not.
We want

<div class="aris-math" markdown="0">
P = ℙ<sub>x~p</sub>[ x causes failure ]
</div>

when P is small — 10⁻⁶ and below is routine in safety validation. The naive estimator draws N
inputs from p and counts failures. Its relative standard error is approximately

<div class="aris-math" markdown="0">
RSE ≈ √( (1 − P) / (N · P) ) ≈ 1 / √(N·P)
<span class="aris-where">so reaching 10% relative error at P = 10⁻⁶ needs on the order of 10⁸ simulator evaluations.</span>
</div>

If each evaluation is a physics rollout or a driving simulation, that budget is unavailable. Below
roughly N = 1/P the estimator returns exactly zero in most runs. A zero is not a small estimate: it
provides no information about P and admits no error bar.

## The robustness convention

ARIS does not work with a boolean failure flag. A system returns a real-valued **robustness** ρ:

!!! info "Convention"
    **Failure is ρ ≤ 0.** Positive robustness means the system succeeded, and the magnitude says
    by how much. More negative means a deeper failure.

This is the convention in `System.evaluate`, and every estimator in the package targets ρ ≤ 0
unless given another threshold. A real-valued robustness is required by the adaptive methods in
this package: a boolean failure flag carries no information until a failure has occurred, whereas ρ
orders non-failing inputs by their proximity to failure. That ordering is what the adaptive
schedule uses before any failure has been observed.

A **threshold** γ generalises the failure event to {ρ ≤ γ}, and the quantity of interest becomes

<div class="aris-math" markdown="0">
P(γ) = ℙ<sub>x~p</sub>[ ρ(x) ≤ γ ]
<span class="aris-where">which is non-increasing in γ; γ = 0 is the real failure probability, and γ > 0 are easier, nearer events.</span>
</div>

Thresholds above 0 are not failures — they are a sequence of progressively rarer events that
*approach* failure, and both the training schedule and the amortization claim are built on that
ladder.

## Importance sampling

Rather than sampling from p, draw from a **proposal** q that puts mass where failures are, and
correct for the mismatch:

<div class="aris-math" markdown="0">
P(γ) = 𝔼<sub>x~q</sub>[ 1{ρ(x) ≤ γ} · p(x)/q(x) ]
<span class="aris-where">estimated by P̂(γ) = (1/N) Σᵢ 1{ρ(xᵢ) ≤ γ} · wᵢ, with weights wᵢ = p(xᵢ)/q(xᵢ) and xᵢ ~ q.</span>
</div>

The estimator is unbiased for any q whose support covers the failure region. Its variance depends
on how closely q matches the variance-minimising proposal, which is p restricted to the failure
region and renormalised. That proposal cannot be sampled from directly, since constructing it
requires P(γ).

The practical problem is therefore the construction of a usable q in a region of the input space
from which no samples have yet been drawn.

!!! warning "Support requirement"
    If q assigns zero density anywhere the failure region has mass, the estimator is biased and no
    amount of sampling fixes it. A proposal that has collapsed onto one failure mode returns an
    underestimate with no indication of error. [Working with multiple
    thresholds](multiple-thresholds.md) covers the diagnostics.

## Adaptive proposals

The standard answer is to build q adaptively: sample, look at what came back, refit q toward the
failure region, repeat. The cross-entropy method does this, and the package ships it as a baseline
([`CrossEntropyMethod`](api-estimators.md#crossentropymethod)).

Adaptive methods work, but they produce a proposal that is specialised to **one** threshold. Ask
for a different γ and you start over, paying the full simulator budget again. When the quantity you
actually want is a *curve* — failure probability as a function of threshold, which is what a safety
case usually needs — that is a budget multiplied by the number of grid points.

ARIS attacks this by making the proposal conditional on robustness, so that one fit covers the
whole ladder. That is the subject of [Amortized importance sampling](amortization.md).

## Effective sample size

A weighted sample's worth is not its count. The package provides Kish's effective sample size:

<div class="aris-math" markdown="0">
ESS(w) = (Σᵢ wᵢ)² / Σᵢ wᵢ²
</div>

```julia
ess(cv.buffer[:w])
```

ESS collapsing toward 1 means a single sample carries the estimate, and the result should not be
believed regardless of how plausible the number looks. See
[Estimating failure probabilities](estimating.md#diagnostics).
