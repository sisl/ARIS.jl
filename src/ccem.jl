@with_kw mutable struct ConditionalValidation
    model
    xdim::Int
    depth::Int
    sdim::Int
    n_iter::Int = 1
    n_samples::Int = 1000
    i::Int = 0
    ignore_keys::Vector{Symbol} = Symbol[]
    buffer_size::Int = n_samples*(n_iter)
    buffer::Buffer = Buffer(xdim, sdim, depth, buffer_size, ignore_keys)
    collect_states::Bool = false
    𝒫::NamedTuple = (;) # algorithm specific params?
    objective_smoothing::Float64 = 0.01
    sampling_strategy::SamplingStrategy = TargetSampling(0.0)
    log_callback::Function = (cv, D)->nothing
    log::Union{LoggerParams, Nothing} = nothing
    sample_batchsize::Union{Int, Nothing} = nothing  # sampling batch size (nothing = one batch)
    r_strategy = :recent
    max_weight::Float64 = 10.0
    γk::Float64 = Inf
end

n_samples(cv::ConditionalValidation) = cv.n_samples
n_iter(cv::ConditionalValidation) = cv.n_iter


# Helper function for batched sampling
function batch_sample(model, conditions, batchsize, xdim; rng=Random.GLOBAL_RNG)
    n = length(conditions)
    
    if isnothing(batchsize) || batchsize >= n
        # Sample all at once
        xs = rand(rng, model, conditions)
        logqs = logpdf(model, xs, conditions)
        return xs, logqs
    else
        # Sample in batches
        xs_batches = []
        logqs_batches = []
        
        for i in 1:batchsize:n
            end_idx = min(i + batchsize - 1, n)
            batch_conditions = conditions[i:end_idx]
            batch_size = length(batch_conditions)
            
            # Sample this batch
            xs_batch = rand(rng, model, batch_conditions)
            logqs_batch = logpdf(model, xs_batch, batch_conditions)
            
            push!(xs_batches, xs_batch)
            push!(logqs_batches, logqs_batch)
        end
        
        # Concatenate results
        xs = hcat(xs_batches...)
        logqs = vcat(logqs_batches...)
        
        return xs, logqs
    end
end

function ConditionalGaussianCEM(;xdim=1, depth=1, sdim=1, thresh_train=Inf, thresh_target=0.0, min_elite=1, kwargs...)
    model = ConditionalGaussian(xdim)
    𝒫 = (dim=xdim, thresh_train=thresh_train, thresh_target=thresh_target, min_elite=min_elite)
    return ConditionalValidation(model=model, xdim=xdim, depth=depth, sdim=sdim; 𝒫=𝒫, kwargs...)
end


function train!(
    cv::ConditionalValidation,
    system::System.SystemParameters;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    show_progress::Bool = false,
    )
    
    N = cv.n_samples
    max_iter = cv.n_iter
    xdim = System.get_xdim(system)
    depth = System.get_depth(system)
    p = System.prior(system)

    # Iteration 1 uses an untruncated prior batch.
    # The buffer therefore stores q_X^(1) = p with unit importance weights,
    # nominal log densities, and conditioning bound r = +∞.
    
    # Simulator budget: N draws, N evaluations, N buffer rows.
    xs = rand(rng, p, N)
    ss = System.simulate(system, xs)
    ρs = System.evaluate(system, ss)
    ws = ones(N)
    logqs = logpdf(p, xs)
    # batch 1 is untruncated, so the conditioning upper bound in effect is +∞
    rs = fill(Inf, N)
    push!(cv.buffer, xs, ρs, ss, ws, rs, logqs)

    info = Dict()

    # Update sampling strategy
    update_sampling_strategy!(cv, cv.buffer; info=info)

    # Train the model
    update(cv, cv.buffer; info=info)
    cv.log_callback(cv, cv.buffer)
    
    if !isnothing(cv.log)
        log(cv.log, 1, info; 𝒮=cv)
    end

    for iteration in 2:max_iter
        # Sample from the model; returns (input condition, xs, logqs)
        input_rs, xs, logqs = gen_samples(cv; rng=rng)

        # Evaluate the system
        ss = System.simulate(system, xs)
        ρs = System.evaluate(system, ss)
        ws = exp.(logpdf(p, xs) .- logqs)
        # Store raw proposal-of-origin weights for estimation; fitting applies
        # `max_weight` locally.

        # Save the data to the buffer, recording the mixture-component indicator at sample time
        # (ground truth); `defensive_mask` is `falses` for non-mixture proposals.
        defmask = defensive_mask(cv.model, N)
        D = Buffer(xdim, cv.sdim, cv.depth, N, cv.ignore_keys)
        push!(D, xs, ρs, ss, ws, input_rs, logqs, defmask)
        push!(cv.buffer, xs, ρs, ss, ws, input_rs, logqs, defmask)
        
        # Log the information
        cv.log_callback(cv, D)

        info = Dict()

        # Update the sampling strategy
        update_sampling_strategy!(cv, D; info=info)

        # Train the model
        update(cv, D; info=info)

        # Call the logger
        if !isnothing(cv.log)
            log(cv.log, iteration, info; 𝒮=cv)
        end

        cv.i += 1
        GC.gc()
    end
end

function update(cv::ConditionalValidation, buffer::Buffer; info=Dict())
    model = cv.model
    if model isa ConditionalGaussian
        return update_gaussian(cv, buffer, info=info)
    # `&& ENABLE_FLOW` short-circuits before touching PyConditionalNormalizingFlow,
    # which is undefined when the flow path is gated off (CGV_ENABLE_FLOW not "true").
    elseif ENABLE_FLOW && model isa PyConditionalNormalizingFlow
        return update_pyflow(cv, buffer, info=info)
    else
        xs = cv.buffer[:x]
        ws = cv.buffer[:w]
        ρs = cv.buffer[:ρ]

        # Fit only on the adaptive stratum when the model opts into stratified adaptation.
        let idx = adaptive_idx(cv, cv.buffer)
            if idx !== nothing
                xs = xs[:, idx]; ws = ws[idx]; ρs = ρs[idx]
            end
        end

        rvalue = maximum(buffer[:r])
        ws_fit = ws .* (ρs .<= rvalue)

        # Clamp weights for fitting only; the buffer retains raw weights for estimation.
        ws_clamped = clamp.(ws, 0.0, cv.max_weight)
        model =  Distributions.fit(model, xs, ρs, ws_clamped)
        cv.model = model
        return model
    end
end

function update_gaussian(cv::ConditionalValidation, D::Buffer; info=Dict())
    model = cv.model
    N = cv.n_samples
    xs = D[:x]
    # The buffer stores raw ws; clamp locally for fitting.
    ws = clamp.(D[:w], 0.0, cv.max_weight)
    ρs = D[:ρ]
    # The fit sees only the adaptive stratum (no-op unless the model opts in).
    let idx = adaptive_idx(cv, D)
        if idx !== nothing; xs = xs[:, idx]; ws = ws[idx]; ρs = ρs[idx]; end
    end

    order = sortperm(ρs)
    sorted_xs = xs[:, order]
    sorted_rs = ρs[order]
    sorted_ws = ws[order]

    elite_thresh = cv.𝒫[:thresh_train]
    min_elite = cv.𝒫[:min_elite]

    N_elite = sorted_rs[end] < elite_thresh ? N : findfirst(sorted_rs .>= elite_thresh) - 1

    N_elite = max(N_elite, min_elite)

    # Fit the model
    model = Distributions.fit(
        model,
        sorted_xs[:, 1:N_elite],
        sorted_rs[1:N_elite],
        sorted_ws[1:N_elite]
    )

    # record mean and covariance of the model
    info[:mean] = model.μ1
    info[:cov] = model.Σ11

    # Update the model
    cv.model = model
end

function update_pyflow(cv::ConditionalValidation, D::Buffer; info=Dict())
    model = cv.model
    N = cv.n_samples
    buffer = cv.buffer
    xs = buffer[:x]
    # The buffer stores raw ws; clamp locally for fitting (mirrors the GMM/Gaussian paths).
    ws = clamp.(buffer[:w], 0.0, cv.max_weight)
    ρs = buffer[:ρ]
    rs = buffer[:r]
    # The fit sees only the adaptive stratum (no-op unless the model opts in).
    let idx = adaptive_idx(cv, buffer)
        if idx !== nothing; xs = xs[:, idx]; ws = ws[idx]; ρs = ρs[idx]; rs = rs[idx]; end
    end

    # Fit on all of the data
    rs_in = ρs
    if cv.r_strategy == :recent
        rs_in = D[:ρ]
    end

    model = Distributions.fit(model, xs, ρs, rs_in, ws)
    GC.gc()

    # Update the model
    cv.model = model

end

# Draw conditions from the sampling strategy (dispatch on its type), then sample the model at them
function gen_samples(cv::ConditionalValidation; rng=Random.GLOBAL_RNG)
    model = cv.model
    N = cv.n_samples
    input_rs = sample(cv.sampling_strategy, N; rng=rng)
    xs, logqs = batch_sample(model, input_rs, cv.sample_batchsize, cv.xdim; rng=rng)
    return input_rs, xs, logqs
end

# STRATIFIED ADAPTATION.
# For defensive-mixture proposals, adaptation uses only adaptive draws,
# while estimation uses both adaptive and defensive draws.
defensive_mask(model, n::Int) = falses(n)   # ground-truth indicator, recorded at sample time
decouple_adaptation(model) = false          # opt-in: exclude the defensive stratum from adaptation

"""Indices of the adaptive stratum in `buffer`, or `nothing` when no filtering applies."""
function adaptive_idx(cv::ConditionalValidation, buffer::Buffer)
    decouple_adaptation(cv.model) || return nothing
    haskey(buffer, :defensive) || return nothing
    d = buffer[:defensive]
    (any(d) && !all(d)) || return nothing
    return findall(.!d)
end

# Update cv.sampling_strategy from the buffer contents
function update_sampling_strategy!(cv::ConditionalValidation, buffer::Buffer; info=Dict())
    ρs = buffer[:ρ]
    rs = buffer[:r]
    ws = buffer[:w]
    logqs = buffer[:logpdf]

    # Schedule sees only the adaptive stratum (no-op unless the model opts in).
    idx = adaptive_idx(cv, buffer)
    if idx !== nothing
        ρs = ρs[idx]; rs = rs[idx]; ws = ws[idx]; logqs = logqs[idx]
    end

    cv.sampling_strategy = update_strategy(
        cv.sampling_strategy, 
        ρs, rs, ws, logqs; 
        info=info
    )
end
