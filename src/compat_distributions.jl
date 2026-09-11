# Compatibility shims for Distributions.jl Gaussian fitting.
#
# ExpectationMaximization fits mixture components using their concrete
# distribution type. For some concrete `MvNormal` subtypes, Distributions'
# weighted fitting dispatch is incomplete or ambiguous. The methods below
# provide the missing full-covariance fallback and resolve subtype-specific
# overlaps while preserving their covariance structure.

import Distributions: fit_mle

# Weighted full-covariance fallback for concrete `MvNormal` subtypes.
# This always returns a full covariance and is therefore not appropriate for
# diagonal or isotropic fits.
function fit_mle(::Type{D}, x::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}) where {D<:Distributions.MvNormal}
    sw = sum(w)
    μ = Vector(float.((x * w) ./ sw))
    xc = x .- μ
    S = Matrix((xc .* w') * xc' ./ sw)
    S = (S + S') / 2 + 1e-10 * I          # symmetrize + tiny jitter for positive-definiteness
    return MvNormal(μ, S)
end

function fit_mle(::Type{D}, x::AbstractMatrix{<:Real}) where {D<:Distributions.MvNormal}
    n = size(x, 2)
    μ = Vector(float.(vec(sum(x, dims = 2)) ./ n))
    xc = x .- μ
    S = Matrix(xc * xc' ./ n)
    S = (S + S') / 2 + 1e-10 * I
    return MvNormal(μ, S)
end

# Resolve the weighted `MvNormal` dispatch overlap by forwarding to the
# full-covariance fallback above.

fit_mle(::Type{Distributions.MvNormal}, x::AbstractMatrix{Float64}, w::AbstractVector{Float64}) =
    invoke(fit_mle,
           Tuple{Type{D}, AbstractMatrix{<:Real}, AbstractVector{<:Real}} where {D<:Distributions.MvNormal},
           Distributions.MvNormal, x, w)

# Resolve weighted subtype dispatch overlaps while preserving covariance
# structure: FullNormal uses the full-covariance fallback, whereas DiagNormal
# and IsoNormal use Distributions' specialized estimators.

fit_mle(::Type{Distributions.FullNormal}, x::AbstractMatrix{Float64}, w::AbstractVector{<:Real}) =
    invoke(fit_mle,
           Tuple{Type{D}, AbstractMatrix{<:Real}, AbstractVector{<:Real}} where {D<:Distributions.MvNormal},
           Distributions.FullNormal, x, w)

fit_mle(::Type{Distributions.DiagNormal}, x::AbstractMatrix{Float64}, w::AbstractVector{<:Real}) =
    invoke(fit_mle, Tuple{Type{Distributions.DiagNormal}, AbstractMatrix{Float64}, AbstractVector},
           Distributions.DiagNormal, x, w)

fit_mle(::Type{Distributions.IsoNormal}, x::AbstractMatrix{Float64}, w::AbstractVector{<:Real}) =
    invoke(fit_mle, Tuple{Type{Distributions.IsoNormal}, AbstractMatrix{Float64}, AbstractVector},
           Distributions.IsoNormal, x, w)
