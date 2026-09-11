# Gaussian Mixture Model structure
mutable struct GaussianMixture{T<:Real}
    n_components::Int
    weights::Vector{T}
    means::Matrix{T}  # each column is a mean vector
    covs::Array{T, 3} # dimensions × dimensions × n_components
    
    function GaussianMixture{T}(n_components::Int, 
                             weights::Vector{T}, 
                             means::Matrix{T}, 
                             covs::Array{T, 3}) where T<:Real
        @assert n_components == length(weights) == size(means, 2) == size(covs, 3)
        @assert size(means, 1) == size(covs, 1) == size(covs, 2)
        weights = weights ./ sum(weights)
        @assert isapprox(sum(weights), one(T), atol=1e-10)
        
        new(n_components, weights, means, covs)
    end
end

# Constructor for creating a model from default means and covariances
# Input is number of components, data dimensions
function GaussianMixture(n_components::Int, d::Int)
    weights = ones(n_components) / n_components
    means = zeros(d, n_components)
    covs = zeros(d, d, n_components)
    for k in 1:n_components
        covs[:,:,k] = Matrix{Float64}(I, d, d)
    end
    return GaussianMixture{Float64}(n_components, weights, means, covs)
end


# Constructor for creating a model from data
function GaussianMixture(n_components::Int, data::Matrix{T}; 
                      init_strategy="random") where T<:Real
    d, n = size(data)
    
    # Initialize with random assignment
    if init_strategy == "random"
        idx = rand(1:n_components, n)
        assignments = zeros(Int, n, n_components)
        for i in 1:n
            assignments[i, idx[i]] = 1
        end
    else
        error("Initialization strategy not implemented")
    end
    
    # Initial weights, means, and covariances
    weights = ones(T, n_components) / n_components
    means = zeros(T, d, n_components)
    covs = zeros(T, d, d, n_components)
    
    for k in 1:n_components
        # Initialize with identity covariance
        covs[:,:,k] = Matrix{T}(I, d, d)
    end
    
    return GaussianMixture{T}(n_components, weights, means, covs)
end

# Fit the GMM to data with optional weights
function Distributions.fit(model::GaussianMixture{T}, 
             data::Matrix{T}, 
             weights::Vector{T}=ones(T, size(data, 2)); 
             tol::T=1e-5, 
             max_iter::Int=500, 
             verbose::Bool=true) where T<:Real
    
    d, n = size(data)
    weights = reshape(weights, 1, n)

    # Reset means, covs, and weights
    n_components = size(model.means, 2)
    model = GaussianMixture(n_components, data)
    
    # Initialize cluster assignments randomly
    R = initialization(data, model.n_components)
    
    # Prepare for EM iterations
    log_likelihood = fill(-Inf, max_iter)
    converged = false
    t = 0
    
    # EM algorithm
    final_mix_weights = zeros(T, model.n_components)
    final_means = zeros(T, size(data, 1), model.n_components)
    final_covs = zeros(T, size(data, 1), size(data, 1), model.n_components)
    while !converged && t < max_iter
        t += 1
        
        # Remove empty components
        label = [argmax(R[i,:]) for i in 1:n]
        u = unique(label)
        if length(u) != model.n_components
            R = R[:, u]
        end
        
        # M-step: Update parameters
        means, covs, mix_weights = maximization(data, weights, R)
        final_mix_weights .= mix_weights
        final_means .= means
        final_covs .= covs
        # E-step: Update responsibilities
        R, log_likelihood[t] = expectation(data, weights, means, covs, mix_weights)
        
        # Check convergence
        if t > 1
            diff = log_likelihood[t] - log_likelihood[t-1]
            eps = abs(diff)
            converged = (eps < tol * abs(log_likelihood[t]))
        end
    end
    
    if converged && verbose
        println("Converged in $t steps.")
    elseif verbose
        println("Not converged in $max_iter steps.")
    end
    
    # Create and return a new model with the fitted parameters
    return GaussianMixture{T}(model.n_components, final_mix_weights, final_means, final_covs)
end

# Initialize cluster assignments randomly
function initialization(data::Matrix{T}, n_components::Int) where T<:Real
    d, n = size(data)
    
    # Random initialization
    idx = rand(1:n, n_components)
    m = data[:, idx]
    
    # Assign each point to closest center
    label = zeros(Int, n)
    for i in 1:n
        distances = [dot(data[:,i] - m[:,j], data[:,i] - m[:,j]) for j in 1:n_components]
        label[i] = argmin(distances)
    end
    
    # Ensure we have the desired number of components
    u = unique(label)
    while length(u) != n_components
        idx = rand(1:n, n_components)
        m = data[:, idx]
        
        for i in 1:n
            distances = [dot(data[:,i] - m[:,j], data[:,i] - m[:,j]) for j in 1:n_components]
            label[i] = argmin(distances)
        end
        
        u = unique(label)
    end
    
    # Convert to one-hot encoding
    R = zeros(T, n, n_components)
    for i in 1:n
        R[i, label[i]] = 1
    end
    
    return R
end

# Expectation step
function expectation(data::Matrix{T}, 
                     weights::Matrix{T}, 
                     means::Matrix{T}, 
                     covs::Array{T, 3}, 
                     mix_weights::Vector{T}) where T<:Real
    d, n = size(data)
    k = length(mix_weights)
    
    logpdf = zeros(T, n, k)
    for i in 1:k
        logpdf[:, i] = log_gaussian_pdf(data, means[:, i], covs[:, :, i])
    end
    
    logpdf .+= log.(mix_weights)'
    log_sum = logsumexp(logpdf, 2)
    log_likelihood = sum(weights .* log_sum) / sum(weights)
    log_responsibilities = logpdf .- log_sum
    
    return exp.(log_responsibilities), log_likelihood
end

# Maximization step
function maximization(data::Matrix{T}, 
                      weights::Matrix{T}, 
                      responsibilities::Matrix{T}) where T<:Real
    
    d, n = size(data)
    k = size(responsibilities, 2)
    
    # Explicitly reshape weights to be a column vector that can broadcast with each row of responsibilities
    weighted_resp = responsibilities .* reshape(weights, :, 1)
    
    # Component weights
    nk = sum(weighted_resp, dims=1)[:]
    
    # Prevent division by zero
    if any(nk .== 0)
        nk .+= 1e-6
    end
    
    mix_weights = nk / sum(weights)
    means = (data * weighted_resp) ./ nk'
    
    # Covariance matrices
    covs = zeros(T, d, d, k)
    sqrt_resp = sqrt.(weighted_resp)
    
    for i in 1:k
        X_centered = data .- means[:, i]
        X_weighted = X_centered .* sqrt_resp[:, i]'
        covs[:, :, i] = (X_weighted * X_weighted') / nk[i]
        
        # Add small regularization for numerical stability
        covs[:, :, i] += I * 1e-6
    end
    
    return means, covs, mix_weights
end

# Log of Gaussian PDF
function log_gaussian_pdf(data::Matrix{T}, 
                         mean::Vector{T}, 
                         cov::Matrix{T}) where T<:Real
    
    d = MvNormal(mean, cov)
    return logpdf(d, data)
end

# Numerically stable log-sum-exp
function logsumexp(x::Matrix{T}, dim::Int=1) where T<:Real
    # Find max for numerical stability
    y = maximum(x, dims=dim)
    x_shifted = x .- y
    s = y .+ log.(sum(exp.(x_shifted), dims=dim))
    
    # Handle edge cases
    not_finite = .!isfinite.(y)
    if any(not_finite)
        idx = findall(not_finite)
        s[idx] .= y[idx]
    end
    
    return s
end

# Distributions.jl-like interface
function Distributions.logpdf(gmm::GaussianMixture{T}, x::AbstractMatrix{T}) where T<:Real
    n_samples = size(x, 2)
    log_probs = zeros(T, n_samples)
    
    for i in 1:n_samples
        component_logprobs = zeros(T, gmm.n_components)
        for k in 1:gmm.n_components
            component_logprobs[k] = log(gmm.weights[k]) + 
                                   log_gaussian_pdf(x[:, i:i], gmm.means[:, k], gmm.covs[:, :, k])[1]
        end
        log_probs[i] = logsumexp(reshape(component_logprobs, 1, :))[1]
    end
    
    return log_probs
end

Distributions.pdf(gmm::GaussianMixture{T}, x::AbstractMatrix{T}) where T<:Real = exp.(logpdf(gmm, x))

function Distributions.rand(rng::Random.AbstractRNG, gmm::GaussianMixture{T}, n::Int=1) where T<:Real
    d = size(gmm.means, 1)
    X = zeros(T, d, n)
    
    # Sample component indices based on weights
    component_indices = rand(Categorical(gmm.weights), n)
    
    # Sample from each selected Gaussian component
    for i in 1:n
        k = component_indices[i]
        X[:, i] = rand(MvNormal(gmm.means[:, k], gmm.covs[:, :, k]))
    end
    
    return X
end

Distributions.rand(gmm::GaussianMixture{T}, n::Int=1) where T<:Real = rand(Random.GLOBAL_RNG, gmm, n)

Distributions.mean(gmm::GaussianMixture{T}) where T<:Real = gmm.means * gmm.weights

# Compute conditional distribution of GMM
function conditional(gmm::GaussianMixture{T}, 
                    cond_dims::Vector{Int}, 
                    cond_values::Vector{T}) where T<:Real
    # Dimensions of the full and conditional distributions
    d = size(gmm.means, 1)
    d_cond = length(cond_dims)
    d_target = d - d_cond
    
    # Get the target (non-conditioned) dimensions
    target_dims = setdiff(1:d, cond_dims)
    
    # Prepare parameters for the conditional GMM
    cond_weights = zeros(T, gmm.n_components)
    cond_means = zeros(T, d_target, gmm.n_components)
    cond_covs = zeros(T, d_target, d_target, gmm.n_components)
    
    # For normalization of the weights
    evidence_probs = zeros(T, gmm.n_components)
    
    # For each component, compute the conditional distribution
    for k in 1:gmm.n_components
        # Extract mean and covariance for this component
        μ = gmm.means[:, k]
        Σ = gmm.covs[:,:, k]
        
        # Partition mean and covariance
        μ₁ = μ[target_dims]  # target mean
        μ₂ = μ[cond_dims]    # conditioning mean
        
        Σ₁₁ = Σ[target_dims, target_dims]    # target covariance
        Σ₁₂ = Σ[target_dims, cond_dims]      # cross-covariance
        Σ₂₁ = Σ[cond_dims, target_dims]      # cross-covariance transposed
        Σ₂₂ = Σ[cond_dims, cond_dims]        # conditioning covariance
        
        # Compute the conditional mean and covariance
        # μ₁|₂ = μ₁ + Σ₁₂ Σ₂₂⁻¹ (x₂ - μ₂)
        # Σ₁|₂ = Σ₁₁ - Σ₁₂ Σ₂₂⁻¹ Σ₂₁
        # Ensure numerical stability
        Σ₂₂_reg = Σ₂₂ + Matrix{T}(I, d_cond, d_cond) * 1e-8
        Σ₂₂_inv = inv(Σ₂₂_reg)
        
        # Compute conditional mean
        cond_means[:, k] = μ₁ + Σ₁₂ * Σ₂₂_inv * (cond_values - μ₂)
        
        # Compute conditional covariance
        cond_covs[:,:, k] = Σ₁₁ - Σ₁₂ * Σ₂₂_inv * Σ₂₁
        
        # Ensure the covariance is symmetric and positive definite
        cond_covs[:,:, k] = (cond_covs[:,:, k] + cond_covs[:,:, k]') / 2
        
        # Add small regularization for numerical stability
        cond_covs[:,:, k] += Matrix{T}(I, d_target, d_target) * 1e-8
        
        # Compute p(x₂|component k) for weight updates
        # This is the likelihood of the conditioning values under this component
        evidence_probs[k] = exp(log_gaussian_pdf(reshape(cond_values, :, 1), μ₂, Σ₂₂)[1])
    end
    
    # Update the weights: w'ₖ ∝ wₖ × p(x₂|component k)
    for k in 1:gmm.n_components
        cond_weights[k] = gmm.weights[k] * evidence_probs[k]
    end
    
    # Normalize weights
    if sum(cond_weights) > 0
        cond_weights ./= sum(cond_weights)
    else
        # Handle numerical issues - use original weights as fallback
        cond_weights .= gmm.weights
    end
    
    # Create and return the conditional GMM
    return GaussianMixture{T}(gmm.n_components, cond_weights, cond_means, cond_covs)
end