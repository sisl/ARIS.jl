using LinearAlgebra
using ARIS
using Distributions

mutable struct RadialShell <: System.SystemParameters
    r_crit::Float64
    d::Int
end
RadialShell(; r_crit=7.5, d=20) = RadialShell(r_crit, d)

System.evaluate(s::RadialShell, x::AbstractVector; kwargs...) = s.r_crit - norm(x)
function System.evaluate(s::RadialShell, x::Array{T,3}; kwargs...) where {T<:Real}
    x = x[:, 1, :]
    return map(c -> System.evaluate(s, c), eachcol(x))
end

System.simulate(s::RadialShell, x::Vector) = x
System.simulate(s::RadialShell, x::AbstractMatrix) = reshape(x, s.d, 1, size(x, 2))

System.prior(s::RadialShell) = MvNormal(zeros(s.d), I)
System.get_depth(s::RadialShell) = 1
System.get_xdim(s::RadialShell) = s.d
System.get_sdim(s::RadialShell) = s.d

# closed-form curve (χ²_d upper tail), for the harness truth
shell_truth(s::RadialShell, γ) = ccdf(Chisq(s.d), (s.r_crit - γ)^2)
