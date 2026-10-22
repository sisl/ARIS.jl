# Quickstart

This page runs one complete estimate: define a system, train an amortized proposal, and read a
failure probability off the result. It assumes the environment is instantiated
([Installation](install.md)).

## The problem

The bundled toy system is

<div class="aris-math" markdown="0">
ρ(x) = γ₀ − min(x₁, x₂),&nbsp;&nbsp; x ~ 𝒩(0, I₂)
<span class="aris-where">so that ℙ[ρ ≤ γ] = Φ(γ − γ₀)², a closed form to check the estimate against.</span>
</div>

With γ₀ = 3 and a target threshold of γ = 0, the failure probability is Φ(−3)² ≈ 1.8 × 10⁻⁶.
Drawn straight from the prior, the 25,000 evaluations used below would be expected to turn up
**0.05 failures** — which is the whole difficulty.

## The run

```julia
using ARIS
using Distributions, Random, LinearAlgebra, Printf

include(joinpath("examples", "problems", "toy.jl"))   # defines UnimodalToy

const γ₀ = 3.0
const γ_target = 0.0

sys = UnimodalToy(γ₀)
truth = cdf(Normal(), γ_target - γ₀)^2

Random.seed!(1)

# The proposal is a joint model over (x, ρ): a marginal over the robustness value, and a
# conditional GMM over inputs given it.
model = JointModel(Normal(), ConditionalEMGMM(2, 3, 3))

cv = ConditionalValidation(
    model = model,
    xdim = 2, depth = 1, sdim = 2,
    n_samples = 1000, n_iter = 25,
    sampling_strategy = QuantileSampling(0.2; target = γ_target),
    max_weight = 10.0,
)

train!(cv, sys)

p̂ = is_estimate(cv.buffer; threshold = γ_target)

@printf("evaluations:  %d\n", length(cv.buffer[:ρ]))
@printf("estimate:     %.4g\n", p̂)
@printf("truth:        %.4g\n", truth)
```

This is [`examples/basic_usage.jl`](https://github.com/sisl/ARIS.jl/blob/main/examples/basic_usage.jl),
which you can run directly:

```bash
julia --project=. examples/basic_usage.jl
```

## What it prints

```text
evaluations:  25000
estimate:     2.245e-06
truth:        1.822e-06
```

!!! note "About this number"
    `Random.seed!(1)` makes the run reproducible for a fixed Julia version and dependency set;
    the values above come from Julia 1.12.6 with the pinned `Manifest.toml`. A different Julia
    version, a different seed, or a resolved (rather than pinned) environment will move the
    estimate. Treat it as the right order of magnitude, not a golden value — a single run of a
    stochastic estimator is one draw, not a converged answer.

## Reading the call

| piece | what it does |
|---|---|
| `JointModel(Normal(), ConditionalEMGMM(2, 3, 3))` | the proposal: a `Normal` over robustness, and a 2-component conditional GMM over inputs given robustness. The `3, 3` are `xdim + 1` features and the index of the conditioning variable — see [Models and distributions](api-models.md#conditionalemgmm) |
| `xdim = 2, depth = 1, sdim = 2` | input dimension, simulation depth, and state dimension, matching `UnimodalToy` |
| `n_samples = 1000, n_iter = 25` | 25 batches of 1000 — a **25,000-evaluation** simulator budget |
| `QuantileSampling(0.2; target = 0.0)` | the schedule: each batch conditions on the 20% quantile of the previous batch's robustness, floored at the target |
| `max_weight = 10.0` | clamps weights **for fitting only**; the buffer keeps raw weights for estimation |
| `cv.buffer` | every evaluated sample, with its robustness, weight and conditioning value |
| `is_estimate(buffer; threshold)` | the importance-sampling estimate of ℙ[ρ ≤ threshold] |

## The thing worth noticing

`is_estimate` takes a `threshold` argument, and nothing about the trained run was specific to
γ = 0. The same buffer answers other thresholds:

```julia
for γ in (0.0, 0.5, 1.0, 1.5)
    @printf("γ = %.1f   P̂ = %.3g   truth = %.3g\n",
            γ, is_estimate(cv.buffer; threshold = γ), cdf(Normal(), γ - γ₀)^2)
end
```

That is the amortization claim, and it has real conditions attached — a threshold the schedule
never reached is not estimated reliably just because the call returns a number. See
[Working with multiple thresholds](multiple-thresholds.md).

## Next

- [Defining a system](defining-a-system.md): put your own problem behind the `System` interface.
- [Training a proposal](training.md): choosing the model, the schedule, and the budget.
- [Methodology](methodology.md): what the training loop actually does, batch by batch.
