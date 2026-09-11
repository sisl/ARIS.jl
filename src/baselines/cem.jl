@with_kw mutable struct CrossEntropyMethod
    model
    xdim::Int
    depth::Int
    sdim::Int
    n_iter::Int = 1
    n_samples::Int = 1000
    n_elite::Int = 100
    i::Int = 0
    ignore_keys::Vector{Symbol} = Symbol[:r]
    buffer_size::Int = n_samples*(n_iter)
    buffer::Buffer = Buffer(xdim, sdim, depth, buffer_size, ignore_keys)
    collect_states::Bool = false
    log::Union{LoggerParams, Nothing} = nothing
    log_callback::Function = (cv, buffer) -> nothing
    sample_batchsize::Union{Int, Nothing} = nothing  # sampling batch size (nothing = one batch)
    max_weight::Float64 = 10.0
end

n_samples(cv::CrossEntropyMethod) = cv.n_samples
n_iter(cv::CrossEntropyMethod) = cv.n_iter

function batch_sample(model, n_samples; rng=Random.GLOBAL_RNG, batchsize=nothing)
    if isnothing(batchsize) || batchsize >= n_samples
        # Sample all at once
        xs = rand(rng, model, n_samples)
        logqs = Distributions.logpdf(model, xs)
        return xs, logqs
    else
        # Sample in batches
        xs_batches = []
        logqs_batches = []
        
        for i in 1:batchsize:n_samples
            end_idx = min(i + batchsize - 1, n_samples)
            batch_size = end_idx - i + 1
            
            # Sample this batch
            xs_batch = rand(rng, model, batch_size)
            logqs_batch = Distributions.logpdf(model, xs_batch)
            
            push!(xs_batches, xs_batch)
            push!(logqs_batches, logqs_batch)
        end
        
        # Concatenate results
        xs = hcat(xs_batches...)
        logqs = vcat(logqs_batches...)
        
        return xs, logqs
    end
end


function train!(
    cv::CrossEntropyMethod,
    system::System.SystemParameters;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    show_progress::Bool = false,
    )

    N = cv.n_samples
    max_iter = cv.n_iter
    xdim = System.get_xdim(system)
    depth = System.get_depth(system)
    p = System.prior(system)

    # Fit to samples from the prior, use a default condition (zero)
    xs = rand(rng, p, N)
    logqs = Distributions.logpdf(p, xs)

    ss = System.simulate(system, xs)
    ρs = System.evaluate(system, ss)
    ws = exp.(Distributions.logpdf(p, xs) .- logqs)
    rs = zeros(N)
    push!(cv.buffer, xs, ρs, ss, ws, rs, logqs)

    info = Dict()

    # Train the model
    order = sortperm(ρs)
    ρs_sorted = ρs[order]
    N_elite = ρs_sorted[end] < 0.0 ? N : findfirst(ρs_sorted .> 0.0) - 1

    N_elite = max(N_elite, cv.n_elite)

    # Update based on elite samples
    elite_xs = xs[:, order[1:N_elite]]
    elite_ws = ws[order[1:N_elite]]

    # clamp the weights to [0, max_weight]
    elite_ws = clamp.(elite_ws, 0.0, cv.max_weight)

    # Fit the prior-batch elites so that iteration 2 samples from the fitted proposal.
    cv.model = Distributions.fit(cv.model, elite_xs, elite_ws)

    info[:elite_thresh] = maximum(ρs_sorted[1:N_elite])
    cv.log_callback(cv, cv.buffer)
    
    if !isnothing(cv.log)
        log(cv.log, 0, info; 𝒮=cv)
    end

    for iteration in 2:max_iter
        # Sample from the model; returns (xs, logqs)
        xs, logqs = batch_sample(cv.model, N; rng=rng, batchsize=cv.sample_batchsize)

        # Evaluate the system
        ss = System.simulate(system, xs)
        ρs = System.evaluate(system, ss)
        ws = exp.(Distributions.logpdf(p, xs) .- logqs)
        input_rs = zeros(N)

        # Save the data to the buffer
        D = Buffer(xdim, cv.sdim, cv.depth, N, cv.ignore_keys)
        push!(D, xs, ρs, ss, ws, input_rs, logqs)
        push!(cv.buffer, xs, ρs, ss, ws, input_rs, logqs)
        
        # Log the information
        cv.log_callback(cv, D)

        info = Dict()

        # Train the model
        order = sortperm(ρs)
        ρs_sorted = ρs[order]
        N_elite = ρs_sorted[end] < 0.0 ? N : findfirst(ρs_sorted .> 0.0) - 1
    
        N_elite = max(N_elite, cv.n_elite)
    
        # Update based on elite samples
        elite_xs = xs[:, order[1:N_elite]]
        elite_ws = ws[order[1:N_elite]]

        elite_ws = clamp.(elite_ws, 0.0, cv.max_weight)
        cv.model = Distributions.fit(cv.model, elite_xs, elite_ws)

        info[:elite_thresh] = maximum(ρs_sorted[1:N_elite])

        # Call the logger
        if !isnothing(cv.log)
            log(cv.log, iteration, info; 𝒮=cv)
        end

        cv.i += 1
    end
end
