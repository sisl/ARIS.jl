@inline function bslice(v, i)
    nd = ndims(v)
    if nd == 2
        return view(v,:,i)
    elseif nd == 3
        return view(v, :, :, i)
    elseif nd == 4
        return view(v, :, :, :, i)
    elseif nd == 1
        return v[i]
    else
        return view(v, ntuple(x->:, nd-1)..., i)
    end
end

function is_estimate(buffer; threshold=0.0)
    ws = buffer[:w]
    ρs = buffer[:ρ]
    fs = map(r->r<=threshold, ρs)
    return mean(fs .* ws)
end

# Kish effective sample size of a weight vector:
#   ess(w) = (Σw)² / Σw²
function ess(weights)
    return sum(weights)^2 / sum(weights.^2)
end

function conditional(mvn::MvNormal, cidx::Vector{Int}, cval::Vector{Float64})
    # Get the indices of the variables we're not conditioning on
    uidx = setdiff(1:length(mvn), cidx)

    # Partition the mean
    μ1 = mvn.μ[uidx]
    μ2 = mvn.μ[cidx]

    # Partition the covariance matrix
    Σ11 = mvn.Σ[uidx, uidx]
    Σ22 = mvn.Σ[cidx, cidx]
    Σ12 = mvn.Σ[uidx, cidx]
    Σ21 = mvn.Σ[cidx, uidx]
    
    # Σ22_inv = inv(Σ22)
    L = cholesky(Σ22).L
    Σ22_inv = L' \ (L \ I)

    # Compute the conditional mean and covariance
    μ1_2 = μ1 + Σ12 * Σ22_inv * (cval - μ2)
    Σ1_2 = Σ11 - Σ12 * Σ22_inv * Σ21

    # Add some bias
    Σ1_2 += 1e-3*I

    # Return the conditional distribution
    return MvNormal(μ1_2, Symmetric(Σ1_2))
end

function marginal(d::MvNormal, dims::Vector{Int})
    μ = mean(d)[dims]
    Σ = cov(d)[dims, dims]
    return MvNormal(μ, Σ)
end

function marginal(d::MvNormal, dim::Int)
    μ = mean(d)[dim]
    Σ = cov(d)[dim, dim]
    return Normal(μ, Σ)
end