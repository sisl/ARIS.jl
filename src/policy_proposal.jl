@with_kw mutable struct ConditionalPolicySolver <: Solver
    agent::PolicyParams # Policy
    S::AbstractSpace # State space
    N::Int = 1000 # Number of episode samples
    ΔN::Int = 200 # Number of episode samples between updates
    max_steps::Int = 100 # Maximum number of steps per episode
    log::Union{Nothing, LoggerParams} = nothing # The logging parameters
    i::Int = 0 # The current number of episode interactions
    a_opt::Union{Nothing, TrainingParams} = nothing# Training parameters for the actor
    𝒫::NamedTuple = (;) # Parameters of the algorithm
    post_sample_callback = (𝒟; kwargs...) -> nothing # Callback that that happens after sampling experience
    pre_train_callback = (𝒮; kwargs...) -> nothing # callback that gets called once prior to training
    required_columns = Symbol[:traj_importance_weight, :return, :logprob]
    objective_smoothing::Float32 = 0.01f0
    sampling_strategy::SamplingStrategy = TargetSampling(0.0)
    
    # Stuff specific to estimation
    training_type = :none # Type of training loop, either :policy_gradient, :value, :none
    training_buffer_size = ΔN*max_steps # Whether or not to train on all prior data or just the recent batch
    weight_fn = safe_weight_fn
    agent_pretrain=nothing # Function to pre-train the agent before any rollouts

    # Create buffer to store all of the samples
    buffer_size = N*max_steps + ΔN*max_steps
    buffer = ExperienceBuffer(S, agent.space, buffer_size, required_columns)
    𝒟 = nothing

end

n_samples(cv::ConditionalPolicySolver) = cv.ΔN
n_iter(cv::ConditionalPolicySolver) = cv.N ÷ cv.ΔN


function ConditionalPolicy(;
    agent::PolicyParams,
    S::AbstractSpace,
    N::Int = 1000,
    ΔN::Int = 200,
    max_steps::Int = 100,
    log::Union{Nothing, LoggerParams} = nothing,
    a_opt::Union{Nothing, TrainingParams} = nothing,
    required_columns = Symbol[:traj_importance_weight, :return, :logprob],
    objective_smoothing::Float32 = 0.01f0,
    sampling_strategy::SamplingStrategy = TargetSampling(0.0),
    r_strategy=:recent,
    kwargs...,
)
    𝒫 = (;objective_smoothing=objective_smoothing, rs=nothing, r_strategy=r_strategy)
    
    ConditionalPolicySolver(;agent,
                     S=S,
                     𝒫=𝒫,
                     N=N,
                     ΔN=ΔN,
                     training_buffer_size=ΔN*max_steps,
                     required_columns,
                     training_type=:none,
                     log=log,
                     a_opt=a_opt,
                     max_steps=max_steps,
                     sampling_strategy=sampling_strategy,
                     kwargs...)
end


function safe_weight_fn(agent, data, ep)
	logp = trajectory_logpdf(agent.pa, data, ep)
	logq = trajectory_logpdf(agent.π, data, ep)
	if logq == -Inf
		return 0f0
	else
		return exp(logp - logq)
	end	
end


function to_buffer(
    eb::ExperienceBuffer,
    mdp,
    xdim::Int,
    sdim::Int,
    depth::Int,
    ignore_keys::Vector{Symbol} = Symbol[],
)
    Neps = length(episodes(eb))
    D = Buffer(xdim + sdim, sdim, depth, Neps, ignore_keys)

    eps = episodes(eb)
    for ep in eps
        # Extract trajectories of disturbances (s0 and actions from the buffer)
        s0 = get_s(mdp, eb[:s][:, ep[1]])
        x_ = eb[:a][:, ep[1]:ep[end]]  # Extract actions
        x = vcat(s0, x_[:])  # Concatenate initial state with actions

        # if length (x) < depth, pad with zeros
        if size(x, 2) < depth
            # Pad with zeros if the episode is shorter than depth
            pad_size = depth - length(x)
            x = vcat(x, zeros(pad_size))
        end
        
        s = eb[:s][:, ep[1]:ep[end]]  # Extract state trajectories

        if size(s, 2) < depth
            # Pad with zeros if the episode is shorter than depth
            pad_size = depth - size(s, 2)
            s = hcat(s, zeros(size(s, 1), pad_size))
        end
        s = s[:, 1:depth]  # Ensure we only take the first 'depth' states

        # convert state trajectories
        straj = mapreduce(si->get_s(mdp, si), hcat, eachcol(s))

        # Extract returns
        ρ = eb[:return][1, ep[1]]  # Extract returns for the episode

        # extract conditions
        r = eb[:s][1, ep[1]]

        # Extract importance weights
        w = eb[:traj_importance_weight][1, ep[1]]

        # Extract traj likelihood
        logq = sum(eb[:logprob][:, ep[1]:ep[end]])

        # Push to the buffer
        push!(D, x, ρ, straj, w, r, logq)
    end

    return D
end


function mle_loss(π, 𝒫, 𝒟; info = Dict())
    # Compute the log probability
    new_probs = logpdf(π, 𝒟[:s], 𝒟[:a])
    ρs = 𝒟[:return][1, :]
    rs = 𝒟[:s][1, :]
    δ = [cdf(Logistic(0.0f0, 𝒫[:objective_smoothing]), ri-ρi) for (ri, ρi) in zip(rs, ρs)]
    weight = 𝒟[:traj_importance_weight] .* δ 
    
    -mean(new_probs .* weight)
end


function vanilla_mle_loss(π, 𝒫, 𝒟; info = Dict())
    # Compute the log probability
    new_probs = logpdf(π, 𝒟[:s], 𝒟[:a])
    ρs = 𝒟[:return][1, :]
    rs = 𝒟[:s][1, :]
    weight = 𝒟[:traj_importance_weight] .* (ρs .<= rs)
    
    norm = Zygote.ignore_derivatives() do
        info[:kl] = mean(𝒟[:logprob] .- new_probs)
        norm = mean(weight)
        if norm == 0f0
            norm = 1f0
        end
        norm
    end 
    
    -mean(new_probs .* weight ./ norm)
end


function relabel_experience_buffer(experience_buffer)
    eps =episodes(experience_buffer)
    ρs = [experience_buffer[:return][1, ep[1]] for ep in eps]
    
    for (ρ, ep) in zip(ρs, eps)
        experience_buffer[:s][1, ep[1]:ep[2]] .= Float32(ρ)
        experience_buffer[:sp][1, ep[1]:ep[2]] .= Float32(ρ)
    end
    return experience_buffer
end


function sanitize_types!(buffer::ExperienceBuffer)
    for k in keys(buffer.data)
        # If the data is an array of float64, convert it to float32
        if isa(buffer.data[k], AbstractArray) && eltype(buffer.data[k]) == Float64
            buffer.data[k] = Float32.(buffer.data[k])
        end
    end
end


function conditional_batch_train!(π, p::TrainingParams, 𝒫, 𝒟::ExperienceBuffer...; info=Dict(), π_loss=π)
    infos = [] # stores the aggregated info for each epoch
    total_batches = 0

    for epoch in 1:p.epochs
        minibatch_infos = [] # stores the info from each minibatch
        
        # Relabel each episode's condition with a random achieved return ≥ its own
        for D in 𝒟
            r_in = D[:return][1, :]

            eps = episodes(D)
            r_eps = [D[:return][1, ep[1]] for ep in eps]
            
            for (i, ep) in enumerate(eps)
                valid_rs = r_eps[r_eps[i] .<= r_eps]
                D[:s][1, ep[1]:ep[2]] .= rand(valid_rs)
            end
        end
            
        # Call train for each minibatch
        partitions = [Base.Iterators.partition(1:length(D), p.batch_size) for D in 𝒟]
        for indices in zip(partitions...)
            mbs = [minibatch(D, i) for (D, i) in zip(𝒟, indices)] 
            push!(minibatch_infos, Crux.train!(π, (;kwargs...)->p.loss(π_loss, 𝒫, mbs...; kwargs...), p, info=info))
            total_batches += 1 
            total_batches >= p.max_batches && break
            p.early_stopping([infos...,  aggregate_info(minibatch_infos)]) && break
        end
        push!(infos, aggregate_info(minibatch_infos))
        p.early_stopping(infos) && break
        total_batches >= p.max_batches && break
    end
    info[string(p.name, "batches_trained")] = total_batches
    merge!(info, aggregate_info(infos))
end


function policy_training(cv::ConditionalPolicySolver, buffer::ExperienceBuffer; info=Dict())
    # info
    π = cv.agent.π
    𝒟 = cv.buffer

    if actor(π) isa DistributionPolicy
        @info "Training distribution policy"
        return info
    end

    sanitize_types!(𝒟)

    if cv.𝒫[:r_strategy] == :recent
        sample_rs = buffer[:return][1, :]
    else
        sample_rs = 𝒟[:return][1, :]
    end
    sample_rs = clamp.(sample_rs, 0.0f0, Inf)
    cv.𝒫 = merge(cv.𝒫, (;rs=sample_rs))

    conditional_batch_train!(actor(π), cv.a_opt, cv.𝒫, deepcopy(𝒟), info=info, π_loss=π)

    info
end


function reset_sampler!(sampler::Sampler, c)
    sampler.was_reset && return
    if sampler.agent.π isa LatentConditionedNetwork && hasproperty(sampler.mdp, :z)
        sampler.agent.π.z = sampler.mdp.z
    end

    Crux.new_ep_reset!(sampler.agent.π)

    sampler.s = rand(initialstate(sampler.mdp; c=c))
    sampler.svec = tovec(initial_observation(sampler.mdp, sampler.s), sampler.S)
    sampler.episode_length = 0
    sampler.was_reset=true
end


function terminate_episode!(sampler::Sampler, data, j, c)
    data[:episode_end][1,j] = true
    ep = j - sampler.episode_length + 1 : j
    haskey(data, :advantage) && fill_gae!(data, ep, sampler.agent.π, sampler.λ, sampler.γ)
    haskey(data, :return) && fill_returns!(data, ep, sampler.γ)
    haskey(data, :fwd_importance_weight) && fill_fwd_importance_weight!(data, ep)
    haskey(data, :cum_importance_weight) && fill_cum_importance_weight!(data, ep)
    haskey(data, :rev_importance_weight) && fill_rev_importance_weight!(data, ep)

    haskey(data, :traj_importance_weight) && (data[:traj_importance_weight][1,ep] .= sampler.traj_weight_fn(sampler.agent, data, ep))

    # Dealing with cost constraints
    haskey(data, :cost_advantage) && fill_gae!(data, ep, sampler.Vc, sampler.λ, sampler.γ, source=:cost, target=:cost_advantage)
    haskey(data, :cost_return) && fill_returns!(data, ep, sampler.γ, source=:cost, target=:cost_return)

    reset_sampler!(sampler, c)
end


function step!(data, c, j::Int, sampler::Sampler; explore=false, i=0)
    sampler.was_reset=false
    a, logprob = explore ? exploration(sampler.agent.π_explore, sampler.svec, π_on=sampler.agent.π, i=i) : (action(sampler.agent.π, sampler.svec), NaN)
    (a isa AbstractArray || a isa Tuple) && length(a) == 1 && (a = a[1])

    # This implements the ability to get cost information from safety gym
    info = Dict()
    kwargs = (haskey(data, :cost) || haskey(data, :z) || haskey(data, :grasp_success)) ? (info=info,) : ()

    args = (a,)
    if !isnothing(sampler.adversary)
        x, xlogprob = explore ? exploration(sampler.adversary.π_explore, sampler.svec, π_on=sampler.adversary.π, i=i) : (action(sampler.adversary.π, sampler.svec), NaN)
        (x isa AbstractArray || x isa Tuple) && length(x) == 1 && (x = x[1]) # disturbances always come out as an array
        data[:x][:, j:j] .= tovec(x, sampler.adversary.space)
        haskey(data, :xlogprob) && (data[:xlogprob][:, j] .= xlogprob)
        args = (a, x)
    end

    if sampler.mdp isa POMDP
        sp, o, r = @gen(:sp,:o,:r)(sampler.mdp, sampler.s, args...; kwargs...)
        spvec = convert_o(AbstractArray, o, sampler.mdp)
    else
        sp, r = @gen(:sp,:r)(sampler.mdp, sampler.s, args...; kwargs...)
        spvec = convert_s(AbstractArray, sp, sampler.mdp)
    end
    spvec = tovec(spvec, sampler.S)
    done = isterminal(sampler.mdp, sp)

    # Save the tuple
    bslice(data[:s], j:j) .= sampler.svec
    data[:a][:, j:j] .= tovec(a, sampler.agent.space)
    bslice(data[:sp], j:j) .= spvec
    data[:r][1, j] = r
    data[:done][1, j] = done

    # Handle optional data storage
    haskey(data, :logprob) && (data[:logprob][:, j] .= logprob)
    if haskey(data, :importance_weight)
        nom_logprob = logpdf(sampler.agent.pa, sampler.svec, tovec(a, sampler.agent.space))
        data[:importance_weight][:, j] .= exp.(nom_logprob .- logprob)
    end
    haskey(data, :t) && (data[:t][1, j] = sampler.episode_length + 1)
    haskey(data, :i) && (data[:i][1, j] = i+1)
    haskey(data, :cost) && (data[:cost][1,j] = info["cost"])
    haskey(data, :grasp_success) && (data[:grasp_success][1,j] = info["grasp_success"])
    if haskey(data, :z) && haskey(info, "z")
        z = info["z"]
        if sampler.agent.π isa LatentConditionedNetwork
            sampler.agent.π.z = z
        end

        if size(data[:z], 1) == 0
            data[:z] = fill(z[1], length(z), size(data[:s], 2))
        end
        data[:z][:, j] = z
    end
    haskey(data, :fail) && (data[:fail][1,j] = extra_functions["isfailure"](sampler.mdp, sp))

    # Cut the episode short if needed
    sampler.episode_length += 1
    if done || sampler.episode_length >= sampler.max_steps
        terminate_episode!(sampler, data, j, c)
    else
        sampler.s = sp
        sampler.svec = spvec
    end
end


function conditional_episodes!(sampler::Sampler, c, buffer=nothing; store=nothing, cb=(kwargs...)->nothing, Neps=1, explore=false, i=0, return_episodes=false)
    reset_sampler!(sampler, c)
    data = mdp_data(sampler.S, sampler.agent.space, Neps*sampler.max_steps, sampler.required_columns)
    episode_starts, episode_ends = zeros(Int, Neps), zeros(Int, Neps)

    j, k = 0, 1
    while k <= Neps
        episode_starts[k] = j+1
        while true
            j = j+1
            step!(data, c, j, sampler, explore=explore, i=i + (i-1))
            if sampler.episode_length == 0
                episode_ends[k] = j
                sampler.episode_checker(data, episode_starts[k], j) ? (k = k+1) : (j = episode_starts[k]-1)
                break
            end
        end
    end
    Crux.trim!(data, j)

    cb(data) # Run the callback on the dataset before adding it
    !isnothing(store) && push!(store, data) # add it to the storage array if provided
    !isnothing(buffer) && push!(buffer, data) # Push it to the provided buffer

    return_episodes ? (data, zip(episode_starts, episode_ends)) : data
end


function sample_policy(cv, mdp) 
    # 1. sample input conditions from strategy
    N = cv.ΔN
    input_rs = sample(cv.sampling_strategy, N)

    # 2. sample episodes from the MDP using the sampled input conditions
    sampler = Sampler(mdp, cv.agent, S=cv.S, required_columns=cv.required_columns, max_steps=cv.max_steps, traj_weight_fn=cv.weight_fn)
    for r in input_rs
        data = conditional_episodes!(sampler, r;  Neps=1, explore=true)
        push!(cv.𝒟, data)
    end

    return cv.𝒟
end


function train!(
    cv::ConditionalPolicySolver,
    mdp::RMDP;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    show_progress::Bool = false,
    )

    max_iter = cv.N ÷ cv.ΔN

    # Fit to samples from the prior
    cv.𝒟 = buffer_like(cv.buffer, capacity=cv.training_buffer_size, device=device(cv.agent.π))

    # Construct the training buffer, constants, and sampler
    nominal_policy = PolicyParams(;π=cv.agent.pa, pa=cv.agent.pa)
    sampler = Sampler(mdp, nominal_policy, S=cv.S, required_columns=cv.required_columns, max_steps=cv.max_steps, traj_weight_fn=cv.weight_fn)

    start_index=length(cv.buffer) + 1
    clear!(cv.𝒟)
    episodes!(sampler, cv.𝒟, store=cv.buffer, Neps=cv.ΔN, explore=true,)
    end_index=length(cv.buffer)


    info = Dict()
    update_sampling_strategy!(cv, cv.𝒟; info=info)

    # Train the model
    info = policy_training(cv, cv.𝒟; info=info)

    # Call the logger
    if !isnothing(cv.log)
        log(cv.log, 0, info; 𝒮=cv)
    end

    for iteration in 1:max_iter
        # Sample episodes from the policy
        clear!(cv.𝒟)
        cv.𝒟 = sample_policy(cv, mdp)
        push!(cv.buffer, cv.𝒟)

        # Update the sampling strategy
        info = Dict()
        update_sampling_strategy!(cv, cv.𝒟; info=info)

        # Train the model
        policy_training(cv, cv.𝒟; info=info)

        cv.i += 1

        # Call the logger
        if !isnothing(cv.log)
            log(cv.log, iteration, info; 𝒮=cv)
        end

    end

    # Close the logger
    if !isnothing(cv.log)
        close(cv.log.logger)
    end

end


function update_sampling_strategy!(cv::ConditionalPolicySolver, current_buffer::ExperienceBuffer; info=Dict())
    eps = episodes(current_buffer)
    ρs = [current_buffer[:return][1, ep[1]] for ep in eps]
    rs = [current_buffer[:s][1, ep[1]] for ep in eps]
    ws = [current_buffer[:traj_importance_weight][1, ep[1]] for ep in eps]
    logqs = [sum(current_buffer[:logprob][:, ep[1]:ep[end]]) for ep in eps]

    # Make sure they all have the same length
    if length(ρs) != length(rs) || length(ρs) != length(ws) || length(ρs) != length(logqs)
        error("All input arrays must have the same length")
    end
    
    cv.sampling_strategy = update_strategy(
        cv.sampling_strategy, 
        ρs, rs, ws, logqs; 
        info=info
    )
end
