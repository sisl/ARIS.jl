elapsed(i::Int, N::Int) = (i % N) == 0
elapsed(i::UnitRange, N::Int) = any([i...] .% N .== 0)

@with_kw mutable struct LoggerParams
    dir::String = "log/"
    period::Int64 = 1
    use_wandb::Bool = false
    config::Union{Dict, Nothing} = nothing
    project::Union{AbstractString, Nothing} = nothing
    entity::Union{AbstractString, Nothing} = nothing
    notes::Union{AbstractString, Nothing} = nothing
    # TensorBoardLogger/WeightsAndBiasLogger are not dependencies of this package,
    # so the `logger` field is untyped and defaults to `nothing`; wandb/TB logging
    # is inert on the toy path (diagnostics pass `log=nothing`).
    logger::Any = nothing
    fns = Any[log_failure_probability, log_weights, log_failrate] # Functions to log
    writeout::Dict{Int, Any} = Dict() # Other things to write to disk. Period => Function
    verbose::Bool = true
end

#Note that i can be an int or a unitrange
function Base.log(p::LoggerParams, i::Union{Int, UnitRange}, data...; 𝒮=nothing)
    
    # Write things to disc
    for (period, fn) in p.writeout
        elapsed(i, period) && fn(i=i[end], dir=p.dir, logger=p.logger)
    end
    
    # Save other run information
    !elapsed(i, p.period) && return
    i = i[end]
    p.verbose && print("Step: $i")
    dicts = Any[p.fns..., data...]
    
    all_dicts = []
    for dict in dicts
        d = dict isa Function ? dict(solver=𝒮, i=i) : dict
        for (k,v) in d
            p.verbose && print(", ", k, ": ", v)
            log_value(p.logger, string(k), v, step = i)
        end
        push!(all_dicts, d)
    end

    p.verbose && println()
end

function log_failure_probability(;solver, i, kwargs...)
    pfail = is_estimate(solver.buffer)
    return Dict("pfail" => pfail)
end

function log_target_performance(system, n)
    (;solver, i, kwargs...) -> begin
        rs = zeros(n)
        x, _ = batch_sample(solver.model, rs, 500, solver.xdim)
        trajs = System.simulate(system, x)
        rhos = System.evaluate(system, trajs)
        fs = rhos .<= 0.0
        prior = System.prior(system)
        ws = exp.(logpdf(prior, x) .- logpdf(solver.model, x, rs))
        ws_smoothed = zeros(n)
        if solver.objective_smoothing > 0.0
            ws_smoothed = ws .* cdf(Logistic(0.0, solver.objective_smoothing), 0.0 .- rhos)
        else
            ws_smoothed = ws .* fs
        end
        pfail = mean(ws .* fs)
        return Dict("pfail_target" => pfail, "ess_target" =>  ess(ws_smoothed)/n)
    end
end

function get_ws(buffer::Buffer)
    return buffer[:w]
end
# Only a Buffer method is defined; log_weights' non-Buffer branch requires `solver.𝒟` to be a Buffer.


function log_weights(;solver, i, kwargs...)
    buffer = solver.buffer
    
    # get the last N weights
    ws = zeros(n_samples(solver))
    if solver isa ConditionalValidation || solver isa CrossEntropyMethod
        N = n_samples(solver)
        last = length(buffer)
        ws = solver.buffer[:w][last-N+1:last]
    else
        ws = get_ws(solver.𝒟)
    end

    # log min, max, mean, std of weights
    d = Dict(
        "min_w" => minimum(ws),
        "max_w" => maximum(ws),
        "mean_w" => mean(ws),
        "std_w" => std(ws),
    )
    return d
end

function get_ρs(buffer::Buffer)
    return buffer[:ρ]
end
# Only a Buffer method is defined; log_failrate's non-Buffer branch requires `solver.𝒟` to be a Buffer.

function log_failrate(;solver, i, kwargs...)
    buffer = solver.buffer
    
    # get the last N weights
    ρs = zeros(n_samples(solver))
    if solver isa ConditionalValidation || solver isa CrossEntropyMethod
        N = n_samples(solver)
        last = length(buffer)
        ρs = solver.buffer[:ρ][last-N+1:last]
    else
        ρs = get_ρs(solver.𝒟)
    end

    d = Dict(
        "fail_rate" => mean(ρs .<= 0.0)
    )
    return d
end