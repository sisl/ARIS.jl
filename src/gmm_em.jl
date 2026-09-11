# Fit instrumentation for EM, KMeans fallback, and covariance-collapse events.
# Callers can reset and read these counters around a fit or run.
const FIT_STATS = Dict{Symbol,Int}(:em => 0, :kmeans_fallback => 0, :em_trial_error => 0,
                                  :small_det => 0, :small_mix => 0,
                                  :degenerate_fit_guard => 0)
reset_fit_stats!() = (for k in keys(FIT_STATS); FIT_STATS[k] = 0; end; FIT_STATS)
get_fit_stats() = copy(FIT_STATS)

mutable struct EMGMM
    mixture::MixtureModel
    diag_covs::Bool
end

function EMGMM(n_components::Int, n_features::Int, diag_covs::Bool = false)
    mixture = MixtureModel(
        [MvNormal(zeros(n_features), I) for i in 1:n_components],
        ones(n_components) / n_components
    )
    return EMGMM(mixture, diag_covs)
end

function Distributions.logpdf(gmm::EMGMM, x::AbstractMatrix)
    return logpdf(gmm.mixture, x)
end

function Distributions.rand(gmm::EMGMM, n::Int)
    return rand(gmm.mixture, n)
end

function Base.rand(rng::AbstractRNG, gmm::EMGMM, n::Int)
    return rand(rng, gmm.mixture, n)
end

function Distributions.fit(gmm::EMGMM, X::AbstractMatrix, weights::AbstractVector)
    Ntrials = 5
    best_ll = -Inf
    best_mix = nothing
    n_components = length(gmm.mixture.components)
    mix_guess = nothing
    for trial in 1:Ntrials
        try
            kmeans_result = kmeans(X, n_components, weights=weights)
            means = kmeans_result.centers

            covs = [cov(X[:, kmeans_result.assignments .== i], dims=2) + 1e-4 * I for i in 1:n_components]
            
            if gmm.diag_covs
                covs = [diagm(0 => diag(covs[i])) for i in 1:n_components]
            end

            mix_guess = MixtureModel(
                [MvNormal(means[:, i], covs[i]) for i in 1:n_components],
                ones(n_components) / n_components
            )

            mix = nothing
        
            mix = fit_mle(mix_guess, X, weights; atol = 1e-3, robust = true, infos = false)
        
            # If any mixture weight is very small, COUNT it rather than printing.
            if any(probs(mix.prior) .< 1e-2)
                FIT_STATS[:small_mix] += count(probs(mix.prior) .< 1e-2)
            end

            # If any of the covariance matrices in the mixture have a very small determinant, count them
            mix_covs = [cov(mix.components[i]) for i in 1:n_components]
            mix_means = [mean(mix.components[i]) for i in 1:n_components]
            if any(det.(mix_covs) .< 1e-3)
                FIT_STATS[:small_det] += count(det.(mix_covs) .< 1e-3)   # count collapses
            end

            # If any of the mixture covariances are very small,  inflate the covariance
            for i in 1:n_components
                if det(mix_covs[i]) < 1e-3
                    mix_covs[i] += 1e-2 * I
                end
            end

            # Update the mixture model
            mix = MixtureModel(
                [MvNormal(mix_means[i], mix_covs[i]) for i in 1:n_components],
                probs(mix.prior)
            )

            ll = loglikelihood(mix, X)
            if ll > best_ll
                best_ll = ll
                best_mix = mix
            end

        catch e
            FIT_STATS[:em_trial_error] += 1                       # count failed EM trials
            @warn "EMGMM: EM trial $trial errored (will fall back to kmeans init only if ALL trials fail)" exception=(e, catch_backtrace())
            continue
        end
    end

    # If all else fails, return the KMeans guess
    if best_mix === nothing
        FIT_STATS[:kmeans_fallback] += 1                          # record the fallback
        @warn "EMGMM: ALL $(Ntrials) EM trials failed — FALLING BACK to the kmeans init (NO EM). \
               Results from this fit are kmeans-only (fallback-assisted)." n_components maxlog=50
    
        # If KMeans also failed to produce a valid mixture, retain the previous
        # mixture so this iteration becomes a no-op rather than raising.
        if mix_guess === nothing
            FIT_STATS[:degenerate_fit_guard] += 1
            @warn "EMGMM: kmeans init ALSO failed on every trial — retaining the previous mixture \
                   (no update this iteration; fallback-assisted)." n_components maxlog=50
            return EMGMM(gmm.mixture, gmm.diag_covs)
        end
        return EMGMM(mix_guess, gmm.diag_covs)
    end

    FIT_STATS[:em] += 1                                           # genuine EM fit
    return EMGMM(best_mix, gmm.diag_covs)
end

function conditional(gmm::EMGMM, condition_idx::Int)
    return conditional(gmm, [condition_idx])
end

function conditional(gmm::EMGMM, condition_idxs::Vector{Int})
    # This function returns a function that takes values for the conditioned dimensions
    # and returns the conditional distribution as a GMM
    
    # Extract components and weights from the mixture model
    components = gmm.mixture.components
    weights = probs(gmm.mixture.prior)
    n_components = length(components)
    
    # Get dimensionality of the distribution
    n_dim = length(components[1].μ)
    
    # Validate condition_idxs
    if any(condition_idxs .< 1) || any(condition_idxs .> n_dim)
        throw(ArgumentError("All condition indices must be between 1 and $n_dim"))
    end
    
    # Determine which dimensions to keep (all except condition_idxs)
    remaining_dims = setdiff(1:n_dim, condition_idxs)
    
    # Return a function that computes the conditional GMM given values
    return function(values::Vector{<:Real})
        if length(values) != length(condition_idxs)
            throw(ArgumentError("Number of values must match number of condition indices"))
        end
        
        # Create new conditional components
        conditional_components = Vector{MvNormal}(undef, n_components)
        
        # Compute posterior weights for each component given the conditioned values
        posterior_weights = similar(weights)
        
        # For each component, compute the conditional distribution
        for i in 1:n_components
            # Extract mean and covariance for this component
            μ = components[i].μ
            Σ = components[i].Σ
            
            # Partition mean: μ = [μ₁, μ₂] where μ₂ are the conditioned variables
            μ₁ = μ[remaining_dims]
            μ₂ = μ[condition_idxs]
            
            # Partition covariance matrix
            Σ₁₁ = Σ[remaining_dims, remaining_dims]
            Σ₁₂ = Σ[remaining_dims, condition_idxs]
            Σ₂₁ = Σ[condition_idxs, remaining_dims]
            Σ₂₂ = Σ[condition_idxs, condition_idxs]
            
            # Compute conditional mean: μ₁|₂ = μ₁ + Σ₁₂Σ₂₂⁻¹(values - μ₂)
            μ_cond = μ₁ + Σ₁₂ * (Σ₂₂ \ (values - μ₂))
            
            # Compute conditional covariance: Σ₁|₂ = Σ₁₁ - Σ₁₂Σ₂₂⁻¹Σ₂₁
            Σ_cond = Σ₁₁ - Σ₁₂ * (Σ₂₂ \ Σ₂₁)
            
            # Create the conditional distribution
            conditional_components[i] = MvNormal(μ_cond, Symmetric(Σ_cond))
            
            # Compute p(x₂) for this component (probability of the observed value)
            p_x2 = pdf(MvNormal(μ₂, Σ₂₂), values)
            
            # Update posterior weight
            posterior_weights[i] = weights[i] * p_x2
        end
        
        # Normalize posterior weights
        if sum(posterior_weights) > 0
            posterior_weights ./= sum(posterior_weights)
        else
            # If all weights are zero, revert to original weights to avoid NaN
            posterior_weights = weights
        end
        
        # Create a new mixture model with the conditional components and updated weights
        conditional_mixture = MixtureModel(conditional_components, posterior_weights)
        
        return conditional_mixture
    end
end


mutable struct ConditionalEMGMM
    gmm::EMGMM
    condition_idx::Int
end

function ConditionalEMGMM(n_components::Int, n_features::Int, condition_idx::Int; diag_covs::Bool = false)
    gmm = EMGMM(n_components, n_features, diag_covs)
    return ConditionalEMGMM(gmm, condition_idx)
end

function Distributions.fit(c_gmm::ConditionalEMGMM, X::AbstractMatrix, rs::AbstractVector, weights::AbstractVector)
    # Convert all to Float64
    X = Float64.(X)
    rs = Float64.(rs) .+ 1e-4 .*randn(size(rs)) # Add some noise to the rs
    weights = Float64.(weights)
    
    # Add rs to X as another dimension
    X_r = vcat(X, rs')

    joint_gmm = fit(c_gmm.gmm, X_r, weights)

    return ConditionalEMGMM(joint_gmm, c_gmm.condition_idx)
end

function Distributions.logpdf(c_gmm::ConditionalEMGMM, X::AbstractMatrix, rs::AbstractVector)
    # Compute the conditional distribution of the gmm given the condition_idx
    conditional_gmm = conditional(c_gmm.gmm, c_gmm.condition_idx)

    # Compute the logpdf of the conditional distribution
    return [logpdf(conditional_gmm([rs[i]]), X[:, i]) for i in 1:size(X, 2)]
end

function Distributions.rand(c_gmm::ConditionalEMGMM, r::AbstractVector)
    # Compute the conditional distribution of the gmm given the condition_idx
    conditional_gmm = conditional(c_gmm.gmm, c_gmm.condition_idx)

    samples = [rand(conditional_gmm([r[i]])) for i in 1:length(r)]
    return hcat(samples...)
end

function Base.rand(rng::AbstractRNG, c_gmm::ConditionalEMGMM, r::AbstractVector)
    # Compute the conditional distribution of the gmm given the condition_idx
    conditional_gmm = conditional(c_gmm.gmm, c_gmm.condition_idx)

    samples = [rand(rng, conditional_gmm([r[i]])) for i in 1:length(r)]
    return hcat(samples...)
end

function Base.rand(rng::AbstractRNG, c_gmm::ConditionalEMGMM, r::Real)
    # Compute the conditional distribution of the gmm given the condition_idx
    conditional_gmm = conditional(c_gmm.gmm, c_gmm.condition_idx)

    samples = rand(rng, conditional_gmm([r]))
    return vec(samples)
end
