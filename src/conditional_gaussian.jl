mutable struct ConditionalGaussian
    μ1::Vector{Float64}
    μ2::Float64
    Σ11::Matrix{Float64}
    Σ12::Matrix{Float64}
    Σ21::Matrix{Float64}
    Σ22_inv::Matrix{Float64}
end

function ConditionalGaussian(d::Int64)
    # Initialize d-dimensional, zero-mean, identity covariance
    μ1 = zeros(d)
    μ2 = 0.0
    Σ11 = diagm(ones(d))
    Σ12 = zeros(d, 1)
    Σ21 = zeros(1, d)
    Σ22_inv = reshape([1.0], 1, 1)
    return ConditionalGaussian(μ1, μ2, Σ11, Σ12, Σ21, Σ22_inv)
end

function (q::ConditionalGaussian)(r)
    μ1 = q.μ1
    μ2 = q.μ2
    Σ11 = q.Σ11
    Σ12 = q.Σ12
    Σ21 = q.Σ21
    Σ22_inv = q.Σ22_inv

    # Compute the conditional mean and covariance
    μ1_2 = μ1 + Σ12 * Σ22_inv * (r - μ2)
    Σ1_2 = Σ11 - Σ12 * Σ22_inv * Σ21

    # Add some bias
    Σ1_2 += 1e-3*I

    # Return the conditional distribution
    return MvNormal(vec(μ1_2), Symmetric(Σ1_2))
end

function Base.rand(rng::AbstractRNG, q::ConditionalGaussian, r::Real)
    return rand(rng, q(r))
end

function Base.rand(rng::AbstractRNG, q::ConditionalGaussian, rs::Vector)
    return hcat([rand(rng, q(r)) for r in rs]...)
end

function Distributions.logpdf(q::ConditionalGaussian, x::AbstractVector, r::Real)
    return logpdf(q(r), x)
end

function Distributions.logpdf(q::ConditionalGaussian, xs::AbstractMatrix, rs::Vector)
    return [logpdf(q(r), x) for (x, r) in zip(eachcol(xs), rs)]
end

function Distributions.fit(q::Type{ConditionalGaussian}, xs, rs, weights)
    # Fallback to unconditional Gaussian behavior
    
    xs = Float64.(xs)
    rs = Float64.(rs)
    if std(rs) <= 1e-6
        data_joint = Distributions.fit(MvNormal, xs, weights)
        μ1 = data_joint.μ
        Σ11 = data_joint.Σ
        Σ12 = zeros(size(Σ11, 1), 1)
        Σ21 = zeros(1, size(Σ11, 1))
        Σ22_inv = zeros(1, 1)
        μ2 = 0.0
        return ConditionalGaussian(μ1, μ2, Σ11, Σ12, Σ21, Σ22_inv)
    end

    data = vcat(xs, rs')
    joint = Distributions.fit(MvNormal, Float64.(data), Float64.(weights))

    cidx = length(joint)

    # Get the indices of the variables we're not conditioning on
    uidx = setdiff(1:length(joint), cidx)

    # Partition the mean
    μ1 = joint.μ[uidx]
    μ2 = joint.μ[cidx]

    # Partition the covariance matrix
    Σ11 = joint.Σ[uidx, uidx]
    Σ22 = joint.Σ[cidx, cidx]
    Σ12 = reshape(joint.Σ[uidx, cidx], length(uidx), 1)
    Σ21 = reshape(joint.Σ[cidx, uidx], 1, length(uidx))

    L = cholesky(Σ22).L
    Σ22_inv = L' \ (L \ I)
    
    return ConditionalGaussian(μ1, μ2, Σ11, Σ12, Σ21, Σ22_inv)

end


function Distributions.fit(q::ConditionalGaussian, xs::AbstractMatrix, rs, weights)
    return fit(ConditionalGaussian, xs, rs, weights)
end

