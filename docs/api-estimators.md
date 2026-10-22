# Estimators and baselines

Turning samples into a failure probability, and the three baselines the package ships for
comparison. The guide version is
[Estimating failure probabilities](estimating.md).

## `is_estimate`

```text
is_estimate(buffer; threshold = 0.0) -> Float64
```

The importance-sampling estimate of ℙ[ρ ≤ `threshold`] from a training buffer:

<div class="aris-math" markdown="0">
P̂(γ) = (1/N) Σ<sub>i</sub> 1{ρ<sub>i</sub> ≤ γ} · w<sub>i</sub>
<span class="aris-where">N is every row in the buffer, including the prior batch; w<sub>i</sub> is the raw proposal-of-origin weight.</span>
</div>

```julia
is_estimate(cv.buffer; threshold = 0.0)
```

It reads only `buffer[:w]` and `buffer[:ρ]`, so it works on any buffer that kept those columns.

- **Not self-normalised** — the denominator is the sample count, not Σw.
- **Pools every batch**, each sample weighted by the proposal that produced it.
- **Weights do not depend on `threshold`** — only the indicator moves, which is what makes
  [one run answer many thresholds](multiple-thresholds.md).

!!! warning "Unbiasedness"
    Proposal-of-origin adaptive importance sampling is unbiased under the conditional-history and
    support assumptions stated in the paper. No unbiasedness guarantee is established for the
    pooled adaptive estimator, because later proposals are fitted from earlier samples.

    The paper's multi-threshold curves use a pooled **balance-MIS** estimator implemented under
    [`paper/`](https://github.com/sisl/ARIS.jl/tree/main/paper), which is **not** part of the
    package API.

!!! note "Zero is not a small number"
    With no samples below the threshold, this returns exactly `0.0`. That is the absence of an
    estimate, not evidence of a tiny probability. Check `count(cv.buffer[:ρ] .<= γ)` before
    reporting ([diagnostics](estimating.md#diagnostics)).

## `ess`

```text
ess(weights) -> Float64
```

Kish's effective sample size, (Σw)² / Σw². The number to watch is the ESS **of the samples in the
failure region**, not of all weights:

```julia
hit = cv.buffer[:ρ] .<= 0.0
ess(cv.buffer[:w][hit])
```

---

All three baselines below target **ρ ≤ 0** and are the comparisons used in the paper.

## `CrossEntropyMethod`

```text
CrossEntropyMethod(; model, xdim, depth, sdim,
                     n_samples = 1000, n_iter = 1, n_elite = 100,
                     max_weight = 10.0, ignore_keys = [:r], ...)
```

The unconditional cross-entropy method: one proposal over inputs, refitted toward the failure
region. Trained with `train!` and read with `is_estimate`, exactly like
`ConditionalValidation`.

!!! warning "The model is unconditional"
    Unlike [`ConditionalValidation`](api-training.md#conditionalvalidation), the model here is a
    distribution over inputs **alone** — not a `JointModel`, and not a conditional model. It must
    support:

    ```text
    rand(rng, model, n::Int)          -> xdim × n matrix
    logpdf(model, X::AbstractMatrix)  -> Vector
    fit(model, X::AbstractMatrix, w::AbstractVector) -> model
    ```

    [`EMGMM`](api-models.md#emgmm) satisfies this.
    [`ConditionalGaussian`](api-models.md#conditionalgaussian) does **not** — its `rand` takes a
    conditioning value, so an integer sample count would be silently read as a conditioning value.

```julia
cem = CrossEntropyMethod(
    model = EMGMM(2, 2),
    xdim = 2, depth = 1, sdim = 2,
    n_samples = 1000, n_iter = 25, n_elite = 100,
)
train!(cem, sys)
is_estimate(cem.buffer; threshold = 0.0)
```

**Elite selection.** Each iteration fits the samples with ρ ≤ 0, or the `n_elite` lowest-ρ samples
if there are fewer than that. Weights are clamped to `max_weight` for the fit. The first (prior)
batch is fitted before the loop, so iteration 2 already samples from a fitted proposal.

`ignore_keys` defaults to `[:r]`, since an unconditional proposal has no conditioning value.

!!! note "One threshold per fit"
    `is_estimate` accepts a `threshold` here too, but the proposal was adapted toward ρ ≤ 0 only.
    Other thresholds are read off a proposal that was never aimed at them, which is precisely the
    limitation [amortization](amortization.md) addresses.

## `ams`

```text
ams(system; m = 100, m_elite = 10, k_max = 100, nmcmc = 10, σ = 0.5)
    -> (p̂fail::Float64, all_xs::Vector)
```

Adaptive multilevel splitting. Maintains a population of `m` particles, repeatedly discards all
but the `m_elite` best, resamples from the survivors, and moves each through `nmcmc` steps of
random-walk Metropolis with an isotropic Gaussian kernel of scale `σ`, targeting the prior
restricted to the current level. The estimate is the product of the per-level survival fractions.

```julia
p̂, levels = ams(sys)
```

Returns the estimate and the particle population at each level.

!!! warning "Per-sample interface, and variable cost"
    `ams` calls the **single-sample** `System.simulate` and `System.evaluate`, so your system must
    define those shapes ([Systems and interfaces](api-systems.md#per-sample-methods)).

    Its simulator cost is not fixed: it stops once a level reaches the target, so the number of
    evaluations varies by seed. Any budget comparison has to charge the realised count — the paper
    reports a median of roughly 4.6× the amortized method's budget on Experiment 4.

## `pmc`

```text
pmc(system; N = 8, T = 10, σ = 1.0, dₓ = 2, use_dm = false)
    -> (μ, Σ, (xs = ..., rs = ..., ws = ...))
```

Population Monte Carlo with `N` Gaussian kernels of covariance σ²I in `dₓ` dimensions, resampled
over `T` iterations toward the failure region.

```julia
μ, Σ, samples = pmc(sys; dₓ = 2)
```

| returned | is |
|---|---|
| `μ` | the kernel means after the final iteration, a vector of `N` vectors |
| `Σ` | the (fixed) kernel covariance, σ²I |
| the named tuple | `xs`, `rs` and `ws` pooled over all iterations |

!!! warning "`pmc` does not return a probability"
    Unlike `ams`, it returns its fitted state and samples, not an estimate. Its weights are already
    multiplied by the failure indicator for ρ ≤ 0, so a pooled estimate is the mean weight:

    ```julia
    μ, Σ, s = pmc(sys; dₓ = 2)
    p̂ = sum(s.ws) / length(s.ws)
    ```

    Note also that when a whole iteration produces no failures its weights are reset to 1/N to keep
    resampling alive; that is a sampler-level guard, and those iterations carry no information
    about the failure probability.

`use_dm = true` switches from standard multiple importance sampling to deterministic-mixture MIS,
weighting each draw against the mixture of all `N` kernels rather than only its own. `pmc` also
uses the **per-sample** system interface.

## Choosing a comparison

| | adapts | thresholds per run | returns |
|---|---|---|---|
| `ConditionalValidation` + `is_estimate` | conditional proposal | **all** | buffer → estimate at any γ |
| `CrossEntropyMethod` | unconditional proposal | one (ρ ≤ 0) | buffer → estimate |
| `ams` | MCMC over nested levels | one (ρ ≤ 0) | estimate + levels |
| `pmc` | mixture of kernels | one (ρ ≤ 0) | fitted state + samples |

A fair comparison charges every method for every simulator evaluation it made, including tuning
runs. See [Reproducing experiments](reproduce.md#the-metric).

## Next

- [Estimating failure probabilities](estimating.md): diagnostics and variance.
- [Paper and method](paper.md): how the baselines are used in the reported results.
