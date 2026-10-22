# Basic usage

[`examples/basic_usage.jl`](https://github.com/sisl/ARIS.jl/blob/main/examples/basic_usage.jl) is
a complete, runnable estimate against a problem with a closed-form answer. This page walks it
through.

```bash
julia --project=. examples/basic_usage.jl
```

## The script

```julia
using ARIS
using Distributions, Random, LinearAlgebra, Printf

include(joinpath(@__DIR__, "problems", "toy.jl"))

const γ₀ = 3.0
const γ_target = 0.0

sys = UnimodalToy(γ₀)
truth = cdf(Normal(), γ_target - γ₀)^2

Random.seed!(1)

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

@printf("\nevaluations:  %d\n", length(cv.buffer[:ρ]))
@printf("estimate:     %.4g\n", p̂)
@printf("truth:        %.4g\n", truth)
```

## Output

```text
evaluations:  25000
estimate:     2.245e-06
truth:        1.822e-06
```

!!! note
    Produced with Julia 1.12.6 and the pinned `Manifest.toml`. `Random.seed!(1)` fixes the run for
    a given Julia version and dependency set; other versions or seeds will give different numbers.
    One run of a stochastic estimator is one draw — see
    [Estimating failure probabilities](estimating.md#variance-and-getting-an-error-bar).

## Why this problem

ρ(x) = 3 − min(x₁, x₂) with x ~ 𝒩(0, I₂) is a **unimodal** rare event with an exact answer:

<div class="aris-math" markdown="0">
ℙ[ρ ≤ γ] = ℙ[min(x₁,x₂) ≥ 3 − γ] = Φ(γ − 3)²
<span class="aris-where">since the two coordinates are independent; at γ = 0 this is Φ(−3)² ≈ 1.822 × 10⁻⁶.</span>
</div>

At that probability, 25,000 draws from the prior would be expected to produce **0.05** failures.
Naive Monte Carlo would return zero essentially every time. So the comparison is not "is the
estimate good" but "is there an estimate at all".

## Extending the script

### Read the whole curve

The run was not specific to γ = 0, so the same buffer answers other thresholds at no extra cost:

```julia
for γ in 2.0:-0.5:0.0
    @printf("γ = %4.1f   P̂ = %-11.4g truth = %.4g\n",
            γ, is_estimate(cv.buffer; threshold = γ), cdf(Normal(), γ - γ₀)^2)
end
```

See [Working with multiple thresholds](multiple-thresholds.md), including the reach and support
conditions this glosses over.

### Watch it train

```julia
cv = ConditionalValidation(
    model = JointModel(Normal(), ConditionalEMGMM(2, 3, 3)),
    xdim = 2, depth = 1, sdim = 2,
    n_samples = 1000, n_iter = 25,
    sampling_strategy = QuantileSampling(0.2; target = γ_target),
    log_callback = (cv, D) -> @printf("  r = %7.4f   min ρ = %7.4f   hits = %4d   ESS = %6.1f\n",
                                      maximum(D[:r]), minimum(D[:ρ]),
                                      count(<=(0.0), D[:ρ]), ess(D[:w])),
)
train!(cv, sys)
```

### Check the diagnostics

```julia
reset_fit_stats!()        # before train!
train!(cv, sys)

ρs, ws = cv.buffer[:ρ], cv.buffer[:w]
hit = ρs .<= γ_target
@printf("hits %d   ESS(hits) %.1f   schedule reached %.4f\n",
        count(hit), count(hit) == 0 ? 0.0 : ess(ws[hit]), minimum(cv.buffer[:r]))
println(get_fit_stats())
```

### Swap in a single Gaussian

```julia
model = JointModel(Normal(), ConditionalGaussian(2))
```

Cheaper and more stable, and adequate here because the failure region is a single corner. It is
the wrong choice on a multimodal problem — see
[`tailrace.jl`](problems.md#tailracetoy-the-multimodal-tail).

### Try your own system

Replace `UnimodalToy` with any `System.SystemParameters` subtype
([Defining a system](defining-a-system.md)), and match `xdim`, `sdim`, `depth` and the
conditional model's feature count to it.

## Next

- [Example problems](problems.md): the other three systems that ship with the package.
- [Training a proposal](training.md): what each option does.
