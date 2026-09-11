using LinearAlgebra
using ARIS
using Distributions

mutable struct TailRaceToy <: System.SystemParameters
    c1::Float64
    c2::Float64
    d::Int          # total dimension (≥2); dims 3..d are nuisance (not in ρ)
end
TailRaceToy(; c1=5.5, c2=23.0, d=2) = TailRaceToy(c1, c2, d)

function System.evaluate(s::TailRaceToy, x::AbstractVector; kwargs...)
    return min(s.c1 - x[1], s.c2 - x[2]^2)
end

function System.evaluate(s::TailRaceToy, x::Array{T,3}; kwargs...) where T<:Real
    x = x[:, 1, :]
    return map(c -> System.evaluate(s, c), eachcol(x))
end

System.simulate(s::TailRaceToy, x::Vector) = x
System.simulate(s::TailRaceToy, x::AbstractMatrix) = reshape(x, s.d, 1, size(x, 2))

System.prior(s::TailRaceToy) = MvNormal(zeros(s.d), I)
System.get_depth(s::TailRaceToy) = 1
System.get_xdim(s::TailRaceToy) = s.d
System.get_sdim(s::TailRaceToy) = s.d
