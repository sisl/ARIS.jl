# Defensive-mixture proposal:
#
#   q_η = (1 - η) q_adapt + η p
#
# where the defensive component is the prior p. This bounds importance weights
# by 1/η and guarantees prior support while retaining the adaptive proposal.
#
# The wrapper preserves the existing training interface:
#   * `rand` samples from the adaptive or defensive component;
#   * `logpdf` evaluates the mixture density;
#   * `fit` updates only the adaptive component.

using ARIS
using Distributions, Random

@inline _logaddexp(a, b) = (m = max(a, b); m + log1p(exp(-abs(a - b))))

mutable struct EtaMixture{B,P}
    base::B
    prior::P
    η::Float64
    decouple::Bool
    last_defensive::Vector{Bool}
end
EtaMixture(base, prior, η; decouple::Bool=true) = EtaMixture(base, prior, η, decouple, Bool[])

function Base.rand(rng::AbstractRNG, m::EtaMixture, γ::AbstractVector)
    Xbase = rand(rng, m.base, γ)
    n = size(Xbase, 2)
    defensive = falses(n)
    for i in 1:n
        if rand(rng) < m.η
            Xbase[:, i] = rand(rng, m.prior)  
            defensive[i] = true
        end
    end
    m.last_defensive = defensive
    return Xbase
end

# For defensive mixtures, adaptation can exclude defensive draws while
# estimation continues to use the full mixture.
ARIS.defensive_mask(m::EtaMixture, n::Int) =
    length(m.last_defensive) == n ? m.last_defensive : falses(n)
ARIS.decouple_adaptation(m::EtaMixture) = m.decouple

function Distributions.logpdf(m::EtaMixture, X::AbstractMatrix, γ::AbstractVector)
    lb = logpdf(m.base, X, γ)
    lp = logpdf(m.prior, X)
    la = log(1 - m.η); lc = log(m.η)
    return [_logaddexp(la + lb[i], lc + lp[i]) for i in eachindex(lb)]
end

function Distributions.fit(m::EtaMixture, xs, rs, ws)
    base2 = Distributions.fit(m.base, xs, rs, ws)
    return EtaMixture(base2, m.prior, m.η, m.decouple, m.last_defensive)
end
