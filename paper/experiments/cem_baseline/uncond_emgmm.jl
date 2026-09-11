# Unconditional EMGMM adapter for the Experiment 1 CEM baseline.
#
# `UncondEMGMM` exposes an unconditional EMGMM through the same model interface
# used by `ConditionalValidation`. Conditioning arguments are ignored, while
# sampling, fitting, checkpointing, and balance-MIS evaluation use the shared
# code path.
#
# Proposal densities are evaluated directly without quadrature.

using ARIS
using Distributions, Random, LinearAlgebra

"""
    UncondEMGMM(g::EMGMM)

Unconditional GMM proposal presented through the conditional model interface.
The robustness condition is accepted and discarded at every entry point.
"""
struct UncondEMGMM
    g::EMGMM
end

UncondEMGMM(n_components::Int, n_features::Int) = UncondEMGMM(EMGMM(n_components, n_features))

# --- optional fit instrumentation (off by default) --------------------------
const UNCOND_FIT_LOG = Vector{NamedTuple}()
const UNCOND_TRACE   = Ref(false)

# Sampling ignores the conditioning values.
Base.rand(rng::AbstractRNG, m::UncondEMGMM, conditions::AbstractVector) =
    rand(rng, m.g.mixture, length(conditions))

# Density evaluation is unconditional.
Distributions.logpdf(m::UncondEMGMM, xs::AbstractMatrix, conditions::AbstractVector) =
    logpdf(m.g.mixture, xs)

# Fit the unconditional proposal using the shared adaptive-training interface.
function Distributions.fit(m::UncondEMGMM, X::AbstractMatrix,
                           rhos::AbstractVector, ws::AbstractVector)
    UNCOND_TRACE[] && push!(UNCOND_FIT_LOG,
        (n_samples_fitted = size(X, 2), n_weights = length(ws),
         max_weight_seen = maximum(ws), rho_arg_len = length(rhos)))
    return UncondEMGMM(Distributions.fit(m.g, X, ws))
end

# Unconditional proposal densities are evaluated directly.
proposal_pdf(m::UncondEMGMM, γ, X, prior; kw...) = pdf(m.g.mixture, X)

Distributions.logpdf(m::UncondEMGMM, xs::AbstractMatrix) = logpdf(m.g.mixture, xs)
n_components(m::UncondEMGMM) = length(m.g.mixture.components)

# Direct density evaluation for bare unconditional EMGMM checkpoints.
proposal_pdf(m::EMGMM, γ, X, prior; kw...) = pdf(m.mixture, X)
