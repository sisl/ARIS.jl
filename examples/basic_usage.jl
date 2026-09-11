# Basic usage: estimate a rare-event probability with an amortized conditional proposal.
#
# The toy system is ρ(x) = γ₀ − min(x₁, x₂) with x ~ N(0, I₂), so the failure event ρ ≤ γ
# has the closed form P(ρ ≤ γ) = Φ(γ − γ₀)². At the target γ = 0 that is ≈ 1.8 × 10⁻⁶: the
# 25,000 evaluations used below would, drawn from the prior, be expected to turn up 0.05
# failures.
#
#   julia --project=. examples/basic_usage.jl

using ARIS
using Distributions, Random, LinearAlgebra, Printf

include(joinpath(@__DIR__, "problems", "toy.jl"))

const γ₀ = 3.0
const γ_target = 0.0

sys = UnimodalToy(γ₀)
truth = cdf(Normal(), γ_target - γ₀)^2

Random.seed!(1)

# The proposal is a joint model over (x, ρ): a marginal over the robustness value ρ, and a
# conditional GMM over inputs given ρ. Conditioning on ρ is what lets one fit track the
# threshold as it moves, instead of being specialised to a fixed one.
model = JointModel(Normal(), ConditionalEMGMM(2, 3, 3))

# QuantileSampling walks the training threshold down toward `target` by taking the 20%
# quantile of each batch's robustness values, so the proposal is never asked to jump
# straight to the rare set. 25 iterations × 1000 samples = 25,000 evaluations.
cv = ConditionalValidation(
    model = model,
    xdim = 2, depth = 1, sdim = 2,
    n_samples = 1000, n_iter = 25,
    sampling_strategy = QuantileSampling(0.2; target = γ_target),
    max_weight = 10.0,
)

train!(cv, sys)

# The buffer holds every evaluated sample with its importance weight; `is_estimate` reads
# the importance-sampling estimate of P(ρ ≤ γ) off it.
p̂ = is_estimate(cv.buffer; threshold = γ_target)

@printf("\nevaluations:  %d\n", length(cv.buffer[:ρ]))
@printf("estimate:     %.4g\n", p̂)
@printf("truth:        %.4g\n", truth)
