# Conditional low-rank-plus-diagonal Gaussian mixture proposal.
#
# Uses StructuredGaussianMixtures for weighted fitting and conditioning on the
# appended robustness dimension. The `rank` parameter controls the low-rank
# covariance capacity.
using StructuredGaussianMixtures
const SGM = StructuredGaussianMixtures
import Distributions

mutable struct ConditionalSGM
    mixture
    condition_idx::Int
    k::Int
    rank::Int
end
ConditionalSGM(k::Int, n_features::Int, condition_idx::Int; rank::Int=10) =
    ConditionalSGM(nothing, condition_idx, k, min(rank, n_features-1))

function Distributions.fit(c::ConditionalSGM, X::AbstractMatrix, rs::AbstractVector, weights::AbstractVector)
    X = Float64.(X); rs = Float64.(rs) .+ 1e-4 .* randn(size(rs)); weights = Float64.(weights)
    X_r = vcat(X, rs')                                   # (d+1, N); condition_idx = d+1
    mix = SGM.fit(SGM.FactorEM(c.k, c.rank), X_r, weights)
    return ConditionalSGM(mix, c.condition_idx, c.k, c.rank)
end

_cond(c, r) = SGM.predict(c.mixture, [r], [c.condition_idx], collect(1:c.condition_idx-1))
Distributions.logpdf(c::ConditionalSGM, X::AbstractMatrix, rs::AbstractVector) =
    [Distributions.logpdf(_cond(c, rs[i]), X[:, i]) for i in 1:size(X, 2)]
Distributions.rand(c::ConditionalSGM, r::AbstractVector) = hcat([rand(_cond(c, r[i])) for i in eachindex(r)]...)
Distributions.rand(rng::Distributions.AbstractRNG, c::ConditionalSGM, r::AbstractVector) =
    hcat([rand(rng, _cond(c, r[i])) for i in eachindex(r)]...)
