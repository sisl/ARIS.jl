# Amortized importance sampling

## The idea

An ordinary adaptive proposal is a distribution over inputs, q(x), fitted toward one failure
region. ARIS instead fits a **joint** distribution over inputs *and* the robustness value they
produce:

<div class="aris-math" markdown="0">
q(x, r) = q<sub>ψ</sub>(r) · q<sub>θ</sub>(x | r)
<span class="aris-where">q<sub>ψ</sub> is a univariate distribution over robustness; q<sub>θ</sub>(x | r) is a conditional model of the inputs that produce robustness r.</span>
</div>

In code that is exactly `JointModel(robustness_model, conditional_model)`:

```julia
model = JointModel(Normal(), ConditionalEMGMM(2, 3, 3))
#                  ───┬───   ──────────┬─────────
#                   q_ψ(r)          q_θ(x | r)
```

The conditional factor is the whole point. It does not answer "which inputs fail?" — it answers the
more general question **"which inputs produce robustness r?"**, for every r at once.

## Threshold reuse

Once you have q(x, r), a proposal aimed at the failure region {ρ ≤ γ} comes from conditioning,
not from refitting. Restrict robustness to r ≤ γ and keep the conditional factor as it is:

<div class="aris-math" markdown="0">
q<sup>γ</sup><sub>X</sub>(x) = ∫<sub>−∞</sub><sup>γ</sup> q<sub>ψ</sub>(r | r ≤ γ) · q<sub>θ</sub>(x | r) dr
<span class="aris-where">q<sub>ψ</sub>(· | r ≤ γ) is q<sub>ψ</sub> truncated above at γ and renormalised over (−∞, γ].</span>
</div>

Sampling from it is two steps, and this is literally what `rand(rng, ::JointModel, γ)` does:

1. draw r ~ truncated(q<sub>ψ</sub>; upper = γ)
2. draw x ~ q<sub>θ</sub>(· | r)

**No refitting. No simulator calls.** Changing γ changes only the truncation point of a univariate
distribution. That is the amortization: the expensive object — the conditional model q<sub>θ</sub>,
which is what the 25,000 evaluations paid for — is reused unchanged across every threshold.

!!! info "Why this is more than a trick"
    A per-threshold method spends its budget learning *one* slice of the input space. The
    conditional model spends the same budget learning how the input distribution **deforms** as
    robustness decreases. The ladder of thresholds is not overhead to be amortized away; it is the
    training signal that makes the conditional model learnable in the first place.

## The density, and why it is an integral

To compute importance weights you need q<sup>γ</sup><sub>X</sub>(x), and the equation above is an
integral with no closed form in general. The package evaluates it numerically, and uses the *same*
evaluator everywhere so that weights are consistent:

```julia
marginal_pdf(jm, γ, X)        # q^γ_X for every column of X
marginal_logpdf(jm, x, γ)     # log q^γ_X(x), one sample
```

The truncation convention matters and is deliberate: the sampler draws r over (−∞, γ], and the
evaluator normalises q<sub>ψ</sub> over the same interval, so sampler and density agree. Numerical
integration runs over [quantile(q<sub>ψ</sub>|<sub>≤γ</sub>, 10⁻¹²), γ] rather than literally to
−∞, which keeps that normalisation while preventing adaptive quadrature from stepping over a sharp
peak near γ.

Accuracy and method are tunable; see
[Models and distributions](api-models.md#marginal-density-evaluation).

## What this does not give you for free

Amortization is a statement about the *proposal*, not a guarantee about the *estimate*.

!!! warning "Three conditions"
    **Reach.** The training schedule walks the conditioning threshold down from the prior. A γ far
    below where the schedule stopped is a region the conditional model was never shown. The density
    integral still evaluates and `is_estimate` still returns a number — it is extrapolation, and it
    is not flagged for you.

    **Support.** q<sub>θ</sub>(x | r) must cover the inputs that actually produce robustness r. A
    conditional model that has collapsed onto one failure mode gives a confidently wrong answer at
    every threshold, not just one.

    **Pooling.** Estimates pool samples from every adaptive batch, each weighted by the proposal it
    came from. Later proposals depend on earlier samples, so the batches are not independent — see
    the unbiasedness caveat in [Estimating failure probabilities](estimating.md#unbiasedness).

[Working with multiple thresholds](multiple-thresholds.md) covers how to check each of these.

## Relationship to the baselines

| | proposal | thresholds per fit | in this package |
|---|---|---|---|
| Cross-entropy method | q(x), unconditional | one | [`CrossEntropyMethod`](api-estimators.md#crossentropymethod) |
| Population Monte Carlo | mixture of Gaussian kernels | one (ρ ≤ 0) | [`pmc`](api-estimators.md#pmc) |
| Adaptive multilevel splitting | MCMC over nested levels | one (ρ ≤ 0), via level decomposition | [`ams`](api-estimators.md#ams) |
| **ARIS** | q(x, r), conditional on robustness | **all of them** | [`JointModel`](api-models.md#jointmodel) + [`ConditionalValidation`](api-training.md#conditionalvalidation) |

AMS also produces intermediate levels, but they are a by-product of its level decomposition at a
fixed target, not a proposal you can re-aim at an arbitrary γ afterwards.

## Next

- [Methodology](methodology.md): the training loop that produces q<sub>ψ</sub> and q<sub>θ</sub>.
- [Working with multiple thresholds](multiple-thresholds.md): reading a curve off one run.
- [Models and distributions](api-models.md): the conditional models available.
