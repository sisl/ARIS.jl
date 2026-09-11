using LinearAlgebra
using ARIS
using Distributions

mutable struct UnimodalToy <: System.SystemParameters
    γ::Float64
end

function System.evaluate(sparams::UnimodalToy, x::AbstractVector; kwargs...)
    d(p) = sparams.γ - min(p[1], p[2])
    return d(x)
end

function System.evaluate(sparams::UnimodalToy, x::Array{T, 3}; kwargs...) where T<: Real
    x = x[:, 1, :]
    return map(x->System.evaluate(sparams, x), eachcol(x))
end

function System.simulate(sparams::UnimodalToy, x::Vector)
    return x
end

function System.simulate(sparams::UnimodalToy, x::AbstractMatrix)
    return reshape(x, 2, 1, size(x, 2))
end


function System.prior(sparams::UnimodalToy)
    return MvNormal([0.0, 0.0], I)
end

function System.get_depth(sparams::UnimodalToy)
    return 1
end

function System.get_xdim(sparams::UnimodalToy)
    return 2
end

function System.get_sdim(sparams::UnimodalToy)
    return 2
end