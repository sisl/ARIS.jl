# Define a joint distribution p(x, r) = p(x|r)p(r)
mutable struct JointModel{R, T} 
    robustness_model::R
    trajectory_model::T
end

# Sample from from the joint model
function Base.rand(rng::AbstractRNG, jm::JointModel)
    r = rand(rng, jm.robustness_model)
    x = rand(rng, jm.trajectory_model, r)
    return x, r
end

# Sample from the joint model conditioned on robustness less than or equal to input gamma
function Base.rand(rng::AbstractRNG, jm::JointModel, γ::Real)
    truncated_robustness_model = truncated(jm.robustness_model; upper=γ)
    r = rand(rng, truncated_robustness_model)
    x = rand(rng, jm.trajectory_model, r)
    return x, r
end

function Base.rand(rng::AbstractRNG, jm::JointModel, γ::Vector)
    r_dists = [truncated(jm.robustness_model; upper=g) for g in γ]
    rs = [rand(rng, r_dist) for r_dist in r_dists]
    xs = rand(rng, jm.trajectory_model, rs)
    return xs
end

# For matrix of samples and vector of conditions
function Distributions.logpdf(jm::JointModel, xs::AbstractMatrix, γs::AbstractVector{<:Real})
    # For a batch sharing one threshold, evaluate the marginal density jointly to
    # reuse each conditional q_θ(·|r) evaluation across samples. Mixed-threshold
    # batches fall back to the per-sample path.
    isempty(γs) && return Float64[]
    if all(==(first(γs)), γs)
        q = marginal_pdf(jm, first(γs), xs)
        return [v > 0 ? log(v) : -Inf for v in q]
    end
    return [logpdf(jm, view(xs, :, i), γs[i]) for i in 1:size(xs, 2)]
end

# For single sample and condition
function Distributions.logpdf(jm::JointModel, x::AbstractVector, γ::Real)
    # Delegate to the canonical marginal-density evaluator so proposal-of-origin
    # and balance-MIS weights use the same numerical density calculation.
    return marginal_logpdf(jm, x, γ)

end

# Fit the joint model to data
function Distributions.fit(jm::JointModel, xs, rs, weights)
    # Fit the robustness model to the data
    if jm.robustness_model isa UnivariateDistribution
        rs = Float64.(rs)
        weights = Float64.(weights)
        robustness_model = Distributions.fit(typeof(jm.robustness_model), rs, weights)
    else
        error("Robustness model must be a univariate distribution")
    end

    # Fit the trajectory model to the data
    trajectory_model = Distributions.fit(jm.trajectory_model, xs, rs, weights)

    # Return a new joint model with the fitted models
    return JointModel(robustness_model, trajectory_model)
end


"""
    fit_trunc_normal_weighted(x, w; a=-Inf, b=Inf)

Fit a 1D Normal truncated to [a,b] to data x with weights w.
Returns (μ̂, σ̂).
"""
function fit_trunc_normal_weighted(x::AbstractVector, w::AbstractVector; a=-Inf, b=Inf)
    # ensure arrays
    x = collect(x)
    w = collect(w)
    # negative weighted log-likelihood, params = [μ, logσ]
    function neg_wll(params)
        μ, logσ = params
        σ = exp(logσ)
        # standardize
        z = (x .- μ) ./ σ
        # log pdf terms
        log_pdf = logpdf.(Normal(), z) .- logσ
        # log normalization constant
        logZ = logcdf(Normal(), (b-μ)/σ) - logcdf(Normal(), (a-μ)/σ)
        return -sum(w .* (log_pdf .- logZ))
    end

    # gradient via ForwardDiff
    grad!(g, p) = ForwardDiff.gradient!(g, neg_wll, p)

    # initial guess: weighted mean & σ
    μ0 = sum(w .* x) / sum(w)
    σ0 = sqrt(sum(w .* (x .- μ0).^2) / sum(w))
    p0 = [μ0, log(σ0)]

    # optimize with box constraint on σ > 0 (i.e. logσ ∈ ℝ)
    result = optimize(neg_wll, grad!, p0, BFGS(); autodiff = :forward)
    
    mu_est, logs_est   = Optim.minimizer(result)
    return mu_est, exp(logs_est)
end

# Define a joint distribution p(x, r) = p(x|r)p(r)
mutable struct TruncatedJointModel
    trajectory_model
    rs::Vector{Float64}
    ws::Vector{Float64}
end

function TruncatedJointModel(trajectory_model)
    return TruncatedJointModel(trajectory_model, Float64[], Float64[])
end

function truncated_robustness_model(jm::TruncatedJointModel, threshold::Real)
    rs = jm.rs[jm.rs .<= threshold]
    ws = jm.ws[jm.rs .<= threshold]
    m, s = fit_trunc_normal_weighted(rs, ws; a=-10.0, b=threshold)
    return truncated(Normal(m, s), -10.0, threshold)
end

# Sample from from the joint model
function Base.rand(rng::AbstractRNG, jm::TruncatedJointModel)
    robustness_model = truncated_robustness_model(jm, Inf)
    r = rand(rng, robustness_model)
    x = rand(rng, jm.trajectory_model, r)
    return x, r
end

# Sample from the joint model conditioned on robustness less than or equal to input gamma
function Base.rand(rng::AbstractRNG, jm::TruncatedJointModel, γ::Real)
    robustness_model = truncated_robustness_model(jm, γ)
    r = rand(rng, robustness_model)
    x = rand(rng, jm.trajectory_model, r)
    return x
end

function Base.rand(rng::AbstractRNG, jm::TruncatedJointModel, γ::Vector)
    samples = [rand(rng, jm, g) for g in γ]
    xs = hcat(samples...)
    return xs
end



# For matrix of samples and vector of conditions
function Distributions.logpdf(jm::TruncatedJointModel, xs::AbstractMatrix, γs::AbstractVector{<:Real})
    return [logpdf(jm, view(xs, :, i), γs[i]) for i in 1:size(xs, 2)]
end

# For single sample and condition
function Distributions.logpdf(jm::TruncatedJointModel, x::AbstractVector, γ::Real)
    robustness_model = truncated_robustness_model(jm, γ)
    trajectory_model = jm.trajectory_model

    function batched_integrand!(out, in_)
        # Create a matrix of xs on where each column is x repeated length(in) times
        xs = repeat(x, 1, length(in_))
        # Compute the logpdf of the trajectory model
        logpdf_traj = logpdf(trajectory_model, xs, in_)
        logpdf_r = logpdf(robustness_model, in_)
        # Compute the integrand
        out .= exp.(logpdf_traj) .* exp.(logpdf_r)
    end

    marginal, _ = quadgk(BatchIntegrand{Float64}(batched_integrand!; max_batch=500), -10.0, γ, rtol=1e-3)
    return log(marginal)

end

# Fit the joint model to data
function Distributions.fit(jm::TruncatedJointModel, xs, rs, weights)
    # Fit the robustness model to the data
    jm.rs = rs
    jm.ws = weights

    # Fit the trajectory model to the data
    trajectory_model = Distributions.fit(jm.trajectory_model, xs, rs, weights)

    # Return a new joint model with the fitted models
    return TruncatedJointModel(trajectory_model, rs, weights)
end
