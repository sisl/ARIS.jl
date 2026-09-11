# σ-smoothed proposal wrapper.
#
# Smoothing is applied only during fitting:
#   * soft inclusion weights retain samples near the current robustness threshold;
#   * robustness jitter is applied only when fitting the conditional model q_θ(x | r).
#
# Sampling and density evaluation delegate to the wrapped proposal, so σ changes
# proposal fitting but not the estimator form.

using ARIS
using Distributions, Random, Statistics

const SOFT_FLOOR = 0.02   # minimum soft-inclusion weight (prevents weighted-EM starvation → nothing)

mutable struct SigmaSmoothed{B}
    base::B          # JointModel
    σ::Float64
    q::Float64       # schedule quantile — sets the soft-inclusion threshold γ ≈ quantile(ρ, q)
end
SigmaSmoothed(base, σ; q::Float64=0.15) = SigmaSmoothed(base, σ, q)

# Sampling and density evaluation delegate to the wrapped proposal.
Base.rand(rng::AbstractRNG, m::SigmaSmoothed, γ) = rand(rng, m.base, γ)
Distributions.logpdf(m::SigmaSmoothed, X, γ) = logpdf(m.base, X, γ)

function Distributions.fit(m::SigmaSmoothed, xs, rs, ws)
    jm = m.base; σ = m.σ
    rs = Float64.(rs); ws = Float64.(ws)

    # Soft inclusion around the current robustness threshold.
    # The floor prevents the weighted fit from losing numerical support.
    γ = quantile(rs, m.q)
    soft = max.(cdf.(Normal(), (γ .- rs) ./ σ), SOFT_FLOOR)
    ws_soft = ws .* soft

    # Fit the robustness marginal on the original robustness values.
    robustness = Distributions.fit(typeof(jm.robustness_model), rs, ws_soft)

    # Jitter robustness values only when fitting the conditional model.
    rs_jit = rs .+ σ .* randn(length(rs))
    trajectory = Distributions.fit(jm.trajectory_model, xs, rs_jit, ws_soft)

    return SigmaSmoothed(JointModel(robustness, trajectory), σ, m.q)
end
