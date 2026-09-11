
abstract type SamplingStrategy end

struct TargetSampling <: SamplingStrategy
    thresh_target::Float64
end

# Sample r values for TargetSampling
function sample(strategy::TargetSampling, n::Int; rng=Random.GLOBAL_RNG)
    return strategy.thresh_target .* ones(n)
end

function update_strategy(strategy::TargetSampling, ρs, rs, ws, logqs; smoothing=0.01, info=Dict())
    return strategy
end


mutable struct QuantileSampling <: SamplingStrategy
    quantile::AbstractFloat
    value::AbstractFloat
    target::AbstractFloat   # descent floor: stop descending at `target`
end

# Default target = 0 floors the descent at 0, i.e. clamp(·, 0, Inf).
QuantileSampling(quantile::AbstractFloat; target::Real=0.0) =
    QuantileSampling(quantile, 0.0f0, Float64(target))


# Sample r values for QuantileSampling
function sample(strategy::QuantileSampling, n::Int; rng=Random.GLOBAL_RNG)
    return strategy.value .* ones(n)
end

function update_strategy(strategy::QuantileSampling, ρs, rs, ws, logqs; smoothing=0.01, info=Dict())
    value = quantile(ρs, strategy.quantile)
    value = clamp(value, strategy.target, Inf)   # descent floors at `target` (default 0)
    info[:r_value] = value
    strategy.value = value
    return strategy
end

function kl_objective(ρs, rs, ws, logqs, smoothing=0.01)
    # Collect unique values of r
    unique_rs = unique(rs)
    obj_values = zeros(length(unique_rs))

    # Take self-normalized IS estimate of the objective at all unique r values
    for (i, r) in enumerate(unique_rs)
        # Get the indices of the samples with this r value
        idxs = findall(rs .== r)
        # Get the corresponding weights and logqs
        ws_r = ws[idxs]
        ρs_r = ρs[idxs]

        ws_r_smoothed = exp.(log.(ws_r) .+ log.(cdf(Logistic(0.0, smoothing), 0.0 .- ρs_r)))
        ws_r_smoothed = ws_r_smoothed ./ sum(ws_r_smoothed)
        
        logqs_r = logqs[idxs]
        
        # Compute the objective value
        obj_values[i] = -mean(ws_r_smoothed .* logqs_r)
    end

    return unique_rs, obj_values
end

# Buffer convenience method: extracts the fields and calls the method above
function kl_objective(buffer::Buffer, smoothing=0.01)
    ws = buffer[:w]
    ρs = buffer[:ρ]
    rs = buffer[:r]
    logqs = buffer[:logpdf]
    
    return kl_objective(ρs, rs, ws, logqs, smoothing)
end

# GaussianNES sampling strategy: maintains a Gaussian distribution and uses NES updates.
@with_kw mutable struct GaussianNES <: SamplingStrategy
    mu::Float64               # Mean
    sigma::Float64            # Standard deviation
    lr::Float64               # Learning rate for NES updates
    n::Int = 1                # Number of samples to evaluate expected objective
end

# Sample r values for GaussianNES
function sample(strategy::GaussianNES, n::Int; rng=Random.GLOBAL_RNG)
    n_samples = strategy.n
    n_rs = n ÷ n_samples
    
    dist = Normal(Float32(strategy.mu), Float32(strategy.sigma))
    rvals = rand(rng, dist, n_rs)
    return repeat(rvals, inner=n_samples)
end

function update_strategy(strategy::GaussianNES, ρs, rs, ws, logqs; smoothing=0.01, info=Dict())
    # Estimate KL objective
    unique_rs, obj_values = kl_objective(ρs, rs, ws, logqs, smoothing)

    # Gaussian log-likelihood gradient
    ∇logpr(r, μ, σ) = [(r - μ)/σ^2, -1/σ + (r - μ)^2/σ^3]

    # Compute the gradient of the objective with respect to mu and sigma
    grad = mean([obj_values[i] .* ∇logpr(unique_rs[i], strategy.mu, strategy.sigma) for i in 1:length(unique_rs)])
    grad_mu = grad[1]
    grad_sigma = grad[2]

    new_mu = strategy.mu - strategy.lr * grad_mu
    new_sigma = strategy.sigma - strategy.lr * grad_sigma

    strategy.mu = new_mu
    strategy.sigma = new_sigma
    return strategy
end

@with_kw mutable struct CrossEntropySampling <: SamplingStrategy
    distribution::Distribution
    quantile::Float64
    n::Int = 1
end

# Sample r values for CrossEntropySampling
function sample(strategy::CrossEntropySampling, n::Int; rng=Random.GLOBAL_RNG)
    dist = strategy.distribution
    n_samples = strategy.n
    n_rs = n ÷ n_samples
    
    rvals = rand(rng, dist, n_rs)
    return repeat(rvals, inner=n_samples)
end

function update_strategy(strategy::CrossEntropySampling, ρs, rs, ws, logqs; smoothing=0.01, info=Dict())
    unique_rs, obj_values = kl_objective(ρs, rs, ws, logqs, smoothing)
    
    # fit to the lower quantile of the objective values
    obj_quantile = quantile(obj_values, strategy.quantile)
    elite_idxs = findall(obj_values .<= obj_quantile)

    # fit to the corresponding r values
    elite_rs = unique_rs[elite_idxs]

    strategy.distribution = fit(typeof(strategy.distribution), elite_rs)

    if strategy.distribution isa Normal
        info[:mean] = mean(strategy.distribution)
        info[:std] = std(strategy.distribution)
    end

    return strategy
end






