# Models and distributions

The proposal models, and the machinery that evaluates their densities. The conceptual version of
this page is [Amortized importance sampling](amortization.md).

## `JointModel`

```text
JointModel(robustness_model, trajectory_model)
```

The ARIS proposal: a joint distribution over inputs and robustness,

<div class="aris-math" markdown="0">
q(x, r) = q<sub>ψ</sub>(r) · q<sub>θ</sub>(x | r)
</div>

| field | is |
|---|---|
| `robustness_model` | a univariate `Distributions.jl` distribution over robustness (q<sub>ψ</sub>) |
| `trajectory_model` | a conditional model of inputs given robustness (q<sub>θ</sub>) |

```julia
model = JointModel(Normal(), ConditionalEMGMM(2, 3, 3))
```

!!! info "The robustness factor must be univariate"
    `fit` raises an error otherwise. It must also support
    `Distributions.fit(typeof(d), values, weights)` and `truncated(d; upper = γ)`.

### Sampling

```text
rand(rng, jm)                      -> (x, r)      untruncated draw from the joint
rand(rng, jm, γ::Real)             -> (x, r)      r ~ truncated(q_ψ; upper = γ), then x ~ q_θ(·|r)
rand(rng, jm, γ::Vector)           -> xs          one draw per entry of γ; returns inputs only
```

The vector form is what `train!` uses: each sample gets its own r drawn from the truncated
robustness model, then an input conditioned on it.

!!! warning "Different return shapes"
    `rand(rng, jm, γ::Real)` returns the **pair** `(x, r)`; `rand(rng, jm, γ::Vector)` returns an
    `xdim × length(γ)` matrix of inputs only.

### Density

```text
logpdf(jm, x::AbstractVector, γ::Real)                  -> Real
logpdf(jm, xs::AbstractMatrix, γs::AbstractVector)      -> Vector
```

These evaluate the **marginal** input density q<sup>γ</sup><sub>X</sub>, not the joint — that is
what importance weights need. Both delegate to
[`marginal_logpdf`](#marginal-density-evaluation). When every entry of `γs` is equal, the batch
form evaluates the integral once across all samples instead of per sample.

### Fitting

```text
Distributions.fit(jm, xs, rs, weights) -> JointModel
```

Fits the two factors separately and returns a **new** `JointModel`: the robustness model by
weighted MLE of its own type, the conditional model by its own `fit`.

## `ConditionalEMGMM`

```text
ConditionalEMGMM(n_components, n_features, condition_idx; diag_covs = false)
```

The default conditional model. For an input dimension `xdim`:

```julia
ConditionalEMGMM(k, xdim + 1, xdim + 1)
```

| argument | value | why |
|---|---|---|
| `n_components` | `k`, typically 2–3 | mixture components |
| `n_features` | `xdim + 1` | robustness is stacked onto the inputs |
| `condition_idx` | `xdim + 1` | the index of the robustness coordinate |
| `diag_covs` | `false` | diagonal component covariances when `true` |

It works by fitting a **joint** Gaussian mixture over `[x; r]` and conditioning it analytically on
`condition_idx`. A Gaussian mixture conditioned on one coordinate is again a Gaussian mixture, so
sampling and density evaluation of q<sub>θ</sub>(· | r) are closed-form once the joint fit exists.

```text
rand(rng, c_gmm, r::Real)        -> Vector          one input
rand(rng, c_gmm, r::Vector)      -> Matrix          one input per conditioning value
logpdf(c_gmm, X::Matrix, rs::Vector) -> Vector      log q_θ(x_i | r_i), column-wise
fit(c_gmm, X, rs, weights)       -> ConditionalEMGMM
```

!!! note "Jitter on the conditioning coordinate"
    `fit` adds Gaussian noise of scale 10⁻⁴ to the robustness values before the EM step. With
    `QuantileSampling` every sample in a batch shares one conditioning value, which would leave the
    joint mixture near-singular in that coordinate. The jitter is small relative to any useful
    threshold spacing.

## `EMGMM`

```text
EMGMM(n_components, n_features, diag_covs = false)
```

The unconditional mixture underneath `ConditionalEMGMM`, usable on its own.

```text
rand(rng, gmm, n)             -> n draws
logpdf(gmm, X::AbstractMatrix)
fit(gmm, X, weights)          -> EMGMM
```

`fit` runs five EM restarts from a weighted k-means initialisation and keeps the best by
log-likelihood. Components whose covariance determinant falls below 10⁻³ are inflated by 10⁻² I.
If every restart fails it falls back to the k-means initialisation; if that also fails it returns
the previous mixture unchanged, making the iteration a no-op rather than raising.

### Fit diagnostics

Those fallbacks are **counted, not printed**:

```julia
reset_fit_stats!()          # before the run
train!(cv, sys)
get_fit_stats()             # a Dict{Symbol,Int}
```

| key | meaning |
|---|---|
| `:em` | genuine EM fits |
| `:kmeans_fallback` | every EM restart failed; the k-means initialisation was used |
| `:em_trial_error` | individual EM restarts that errored |
| `:small_det` | components with near-singular covariance (then inflated) |
| `:small_mix` | mixture weights below 10⁻² |
| `:degenerate_fit_guard` | k-means also failed; the previous mixture was retained |

!!! warning
    A non-zero `:kmeans_fallback` or `:degenerate_fit_guard` means some iterations did not get a
    real fit. The run still completes and still returns an estimate. The paper records three
    Experiment 3 baseline runs that took this path and reports clean-only sensitivity aggregates
    alongside the headline numbers.

## `ConditionalGaussian`

```text
ConditionalGaussian(d::Int)
```

A single-Gaussian conditional model: a joint Gaussian over `[x; r]`, stored as its partitioned
blocks and conditioned in closed form.

```text
q(r)                                  -> MvNormal    the conditional at robustness r
rand(rng, q, r::Real)                 -> Vector
rand(rng, q, rs::Vector)              -> Matrix
logpdf(q, x::AbstractVector, r::Real) -> Real
fit(q, xs, rs, weights)               -> ConditionalGaussian
```

Cheap and stable, but unimodal. If the robustness values have standard deviation below 10⁻⁶ the
fit degenerates gracefully to an unconditional Gaussian over the inputs. A ridge of 10⁻³ I is
added to the conditional covariance.

!!! warning "Unimodal means unimodal"
    On a problem with genuinely separate failure modes this will collapse onto one of them and
    return a confident underestimate. See
    [`tailrace.jl`](problems.md#tailracetoy-the-multimodal-tail).

## `GaussianMixture`

```text
GaussianMixture(n_components, d::Int)
GaussianMixture(n_components, data::Matrix; init_strategy = "random")
```

A standalone mixture type holding `weights`, `means` (one per column) and `covs`
(`d × d × n_components`). Exported, and used by parts of the package that need an explicit mixture
representation rather than a `Distributions.MixtureModel`.

## `TruncatedJointModel`

```text
TruncatedJointModel(trajectory_model)
```

An alternative joint model that, instead of carrying a parametric robustness distribution, retains
the observed robustness values and weights and fits a **truncated** normal to those below the
requested threshold at evaluation time.

```text
rand(rng, jm, γ::Real)        -> x
logpdf(jm, x, γ::Real)        -> Real       per-sample adaptive quadrature over [−10, γ]
fit(jm, xs, rs, weights)      -> TruncatedJointModel
```

!!! note
    Its integration domain is bounded below at −10 rather than following the `MARGINAL_*`
    settings, and its density is evaluated per sample. `JointModel` is the path the examples and
    the paper's reported runs use.

## Marginal density evaluation

The canonical evaluator for

<div class="aris-math" markdown="0">
q<sup>γ</sup><sub>X</sub>(x) = ∫<sub>−∞</sub><sup>γ</sup> q<sub>ψ</sub>(r | r ≤ γ) · q<sub>θ</sub>(x | r) dr
</div>

All estimator-side density calls route through it, so weights are numerically consistent.

```text
marginal_pdf(jm, γ::Real, X::AbstractMatrix; method, rtol, lower, nodes) -> Vector{Float64}
marginal_pdf(jm, γ::Real, x::AbstractVector; kwargs...)                  -> Vector{Float64}
marginal_logpdf(jm, x::AbstractVector, γ::Real; kwargs...)               -> Real
```

!!! warning "Argument order differs"
    `marginal_pdf(jm, γ, x)` takes the threshold **second**; `marginal_logpdf(jm, x, γ)` takes it
    **third**, matching `logpdf(jm, x, γ)`.

### Settings

Global `Ref`s, so one setting governs every call site:

| ref | default | meaning |
|---|---|---|
| `MARGINAL_METHOD` | `:batch` | `:batch` (vector-valued adaptive quadrature), `:adaptive` (per sample), `:grid` (fixed grid) |
| `MARGINAL_RTOL` | `1e-3` | relative tolerance for `:batch` and `:adaptive` |
| `MARGINAL_LOWER` | `-Inf` | `-Inf` normalises q<sub>ψ</sub> over (−∞, γ], matching the sampler; a finite value uses [lower, γ] |
| `MARGINAL_NODES` | `1000` | nodes, `:grid` only |

```julia
ARIS.MARGINAL_RTOL[] = 1e-4     # tighter weights, slower
```

Integration runs over [quantile(q<sub>ψ</sub>|<sub>≤γ</sub>, 10⁻¹²), γ] rather than literally to
−∞, which keeps the (−∞, γ] normalisation while preventing adaptive quadrature from stepping over
a sharp peak near γ.

### Checking and recording

```text
marginal_reference(jm, γ, X; rtol = 1e-10) -> Vector{Float64}
marginal_settings()                        -> Dict
```

`marginal_reference` is a high-accuracy per-sample integral for measuring the canonical
evaluator's relative error — too slow for production. `marginal_settings` returns the active
configuration for recording in run provenance.

### `conditional_logpdf_at`

```text
conditional_logpdf_at(traj, r::Real, X::AbstractMatrix) -> Vector
```

log q<sub>θ</sub>(x | r) for every column of `X` at a **single** conditioning value, building the
conditional distribution once. Model types add methods here; the fallback works for anything
supporting `logpdf(traj, X, rs)`.

## Next

- [Training and sampling](api-training.md): running a model against a system.
- [Estimators and baselines](api-estimators.md): turning a buffer into a number.
