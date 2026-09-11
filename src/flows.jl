@with_kw struct FlowTrainingParams
    batch_size::Int = 64
    lr::Float64 = 3e-4
    weight_decay::Float64 = 1e-5
    verbose::Bool = false
    epochs::Int = 10        # fixed number of training epochs
end


@with_kw mutable struct PyConditionalNormalizingFlow
    model::PyObject
    latent_size::Int
    context_size::Int
    device::String = "cpu"
    params::FlowTrainingParams = FlowTrainingParams()
end


function AutoregressiveSplineFlow(n_layers::Int, latent_size::Int, hidden_layers::Int, hidden_units::Int, context_size::Int; params=FlowTrainingParams(), tail_bound=8, num_bins=8, device="cpu", actnorm=false)
    # Create an empty Python list
    flows = pyimport("builtins").list()
    
    for i = 1:n_layers
        l1 = nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, num_context_channels=context_size, tail_bound=tail_bound, num_bins=num_bins)
        push!(flows, l1)
        l2 = nf.flows.LULinearPermute(latent_size)
        push!(flows, l2)
    end

    q0 = nf.distributions.DiagGaussian(latent_size, trainable=false)
    model = nf.ConditionalNormalizingFlow(q0=q0, flows=flows)

    return PyConditionalNormalizingFlow(model=model, latent_size=latent_size, context_size=context_size, device=device, params=params)
end


function Distributions.rand(rng::AbstractRNG, q::PyConditionalNormalizingFlow, r::Vector)
    model = q.model
    
    # Repeat the context context_size times for each sample
    r = repeat(r, 1, q.context_size)
    rnp = np.array(r')

    # Convert to torch tensor
    tm = torch.inference_mode()
    tm.__enter__()
    try
        xtorch, _ = model.sample(size(r, 1), torch.tensor(rnp', dtype=torch.float32))
        x = Array{Float32}(xtorch.detach().cpu().numpy())' # PyCall conversion
        tm.__exit__(nothing, nothing, nothing) # PyCall doesn't use pybuiltins.None
        return Array(x)
    catch e
        @warn "Error in sampling from the model"
        tm.__exit__(nothing, nothing, nothing)
        throw(e)
    end
end

# Unconditional sampling
function Distributions.rand(rng::AbstractRNG, q::PyConditionalNormalizingFlow, n::Int)
    model = q.model
    

    # Convert to torch tensor
    tm = torch.inference_mode()
    tm.__enter__()
    try
        xtorch, _ = model.sample(n)
        x = Array{Float32}(xtorch.detach().cpu().numpy())' # PyCall conversion
        tm.__exit__(nothing, nothing, nothing) # PyCall doesn't use pybuiltins.None
        return Array(x)
    catch e
        @warn "Error in sampling from the model"
        tm.__exit__(nothing, nothing, nothing)
        throw(e)
    end
end

function Distributions.rand(q::PyConditionalNormalizingFlow, r::Vector{Float64})
    rng = Random.GLOBAL_RNG
    return Distributions.rand(rng, q, r)
end
    

function Distributions.logpdf(q::PyConditionalNormalizingFlow, x::AbstractArray, r::AbstractVector)
    model = q.model

    # Repeat the context context_size times for each sample
    r = repeat(r, 1, q.context_size)

    tm = torch.inference_mode()
    tm.__enter__()

    try
        # Convert to torch tensor
        xtorch = torch.tensor(np.array(x'), dtype=torch.float32)
        rtorch = torch.tensor(np.array(r), dtype=torch.float32)

        # Compute the log likelihood
        log_prob_torch = model.log_prob(xtorch, rtorch)
        tm.__exit__(nothing, nothing, nothing)

        log_prob = Array{Float32}(log_prob_torch.detach().cpu().numpy()) # PyCall conversion
        return log_prob
    catch e
        @warn "Error in computing logpdf from the model"
        tm.__exit__(nothing, nothing, nothing)
        throw(e)
    end
end

# Unconditional logpdf
function Distributions.logpdf(q::PyConditionalNormalizingFlow, x::AbstractArray)
    model = q.model

    tm = torch.inference_mode()
    tm.__enter__()

    try
        # Convert to torch tensor
        xtorch = torch.tensor(np.array(x'), dtype=torch.float32)

        # Compute the log likelihood
        log_prob_torch = model.log_prob(xtorch)
        tm.__exit__(nothing, nothing, nothing)

        log_prob = Array{Float32}(log_prob_torch.detach().cpu().numpy()) # PyCall conversion
        return log_prob
    catch e
        @warn "Error in computing logpdf from the model"
        tm.__exit__(nothing, nothing, nothing)
        throw(e)
    end
end


function stack_buffers_relabel(buffers; limit=nothing)
    xdim = size(buffers[1][:x], 1)
    sdim = size(buffers[1][:s], 1)
    depth = size(buffers[1][:s], 2)
    n = isnothing(limit) ? sum(length(b) for b in buffers) : limit
    buffer = Buffer(xdim, sdim, depth, n, [])
    count = 0
    for b in buffers
        xs_ = b[:x]
        ss_  = b[:s]
        ρs_ = b[:ρ]
        ws_ = b[:w]
        rs_ = b[:r]
        logpdfs_ = b[:logpdf]

        nb = length(ρs_)

        rs = rand(ρs_, nb)
        for (i, r) in enumerate(rs)
            idxs = findall(ρs_ .<= r)
            idx = rand(idxs)
            push!(buffer, xs_[:, idx], ρs_[idx], ss_[:, :, idx], ws_[idx], r, logpdfs_[idx])
            count += 1
            if count >= n
                return buffer
            end
        end
    end
    return buffer
end

function relabel_buffer(buff)
    buffers = [buff]
    xdim = size(buffers[1][:x], 1)
    sdim = size(buffers[1][:s], 1)
    depth = size(buffers[1][:s], 2)
    n = sum(length(b) for b in buffers)
    buffer = Buffer(xdim, sdim, depth, n, [])
    count = 0
    for b in buffers
        xs_ = b[:x]
        ss_  = b[:s]
        ρs_ = b[:ρ]
        ws_ = b[:w]
        rs_ = b[:r]
        logpdfs_ = b[:logpdf]

        nb = length(ρs_)

        rs = rand(ρs_, nb)
        for (i, r) in enumerate(rs)
            idxs = findall(ρs_ .<= r)
            idx = rand(idxs)
            push!(buffer, xs_[:, idx], ρs_[idx], ss_[:, :, idx], ws_[idx], r, logpdfs_[idx])
            count += 1
        end
    end
    return buffer
end

function relabel_buffer_v2(buff)
    buffers = [buff]
    xdim = size(buffers[1][:x], 1)
    sdim = size(buffers[1][:s], 1)
    depth = size(buffers[1][:s], 2)
    n = sum(length(b) for b in buffers)
    buffer = Buffer(xdim, sdim, depth, n, [])
    count = 0
    for b in buffers
        xs_ = b[:x]
        ss_  = b[:s]
        ρs_ = b[:ρ]
        ws_ = b[:w]
        rs_ = b[:r]
        logpdfs_ = b[:logpdf]

        nb = length(ρs_)

        for (i, xi) in enumerate(eachcol(xs_))
           ρi = ρs_[i]
           
           valid_idxs = findall(rs_ .>= ρi)
           idx = rand(valid_idxs)
            push!(buffer, xi, ρi, ss_[:, :, i], ws_[i], rs_[idx], logpdfs_[i])
        end
    end

    return buffer
end

# Conditional fit
function Distributions.fit(q::PyConditionalNormalizingFlow, xs, rs, ws)
    model = q.model

    batch_size = q.params.batch_size
    lr = q.params.lr
    weight_decay = q.params.weight_decay
    verbose = q.params.verbose
    epochs = q.params.epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_hist = []

    n_samples = size(xs, 2)
    n_batches = ceil(Int, n_samples / batch_size)

    for epoch = 1:epochs
        # shuffle indices for a full pass through the dataset
        shuffled = randperm(n_samples)
        for i = 1:n_batches
            optimizer.zero_grad()
            
            batch_start = (i - 1) * batch_size + 1
            batch_end = min(i * batch_size, n_samples)
            batch_idxs = shuffled[batch_start:batch_end]
            current_batch_size = length(batch_idxs)

            xbatch = xs[:, batch_idxs]
            rbatch = repeat(rs[batch_idxs], 1, q.context_size)'
            wbatch = ws[batch_idxs]

            xnp = np.array(xbatch')
            x = torch.tensor(xnp).float()

            rnp = np.array(rbatch')
            r = torch.tensor(rnp).float()

            wnp = np.array(wbatch)
            w = torch.tensor(wnp).float()
            
            # Compute loss
            log_q = torch.zeros(current_batch_size)
            z = x
            for k = length(model.flows)-1:-1:-1
                z, log_det = model.flows[k].inverse(z, context=r)
                log_q += log_det
            end
            log_q += model.q0.log_prob(z, context=r)
            loss = -torch.sum(log_q * w) / torch.sum(w) # weighted loss
            
            # Backpropagation and optimizer step
            loss.backward()
            optimizer.step()

            if verbose
                println("Epoch: ", epoch, " Batch: ", i, " Loss: ", loss.item())
            end
            
            push!(loss_hist, Float64(loss.item())) # PyCall conversion
        end
    end

    return q
end

function Distributions.fit(q::PyConditionalNormalizingFlow, xs, ρs, rs, ws; smoothing=0.0)
    model = q.model

    batch_size = q.params.batch_size
    lr = q.params.lr
    weight_decay = q.params.weight_decay
    verbose = q.params.verbose
    epochs = q.params.epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_hist = []

    n_samples = size(xs, 2)
    n_batches = ceil(Int, n_samples / batch_size)

    if length(rs) != n_samples
        rs = rand(rs, n_samples)
    end

    for epoch = 1:epochs
        # shuffle indices for a full pass through the dataset
        shuffled = randperm(n_samples)

        # randomly shuffle the r values
        rs = shuffle(rs)

        # compute weights
        ws_fail = zeros(n_samples)
        if !isnothing(smoothing) && smoothing > 0.0
            δ = [cdf(Logistic(0.0, smoothing), ri-ρi) for (ri, ρi) in zip(rs, ρs)]
            ws_fail = ws .* δ
        else
            ws_fail = ws .* (ρs .<= rs)
        end

        for i = 1:n_batches
            optimizer.zero_grad()
            
            batch_start = (i - 1) * batch_size + 1
            batch_end = min(i * batch_size, n_samples)
            batch_idxs = shuffled[batch_start:batch_end]
            current_batch_size = length(batch_idxs)

            xbatch = xs[:, batch_idxs]
            rbatch = repeat(rs[batch_idxs], 1, q.context_size)'
            wbatch = ws_fail[batch_idxs]

            xnp = np.array(xbatch')
            x = torch.tensor(xnp).float()

            rnp = np.array(rbatch')
            r = torch.tensor(rnp).float()

            wnp = np.array(wbatch)
            w = torch.tensor(wnp).float()
            
            # Compute loss
            log_q = torch.zeros(current_batch_size)
            z = x
            for k = length(model.flows)-1:-1:-1
                z, log_det = model.flows[k].inverse(z, context=r)
                log_q += log_det
            end
            log_q += model.q0.log_prob(z, context=r)
            norm = torch.sum(w)
            
            if Float64(norm.item()) == 0.0
                norm = torch.tensor(1.0).float()
            end
            loss = -torch.sum(log_q * w) / norm
            
            # Backpropagation and optimizer step
            loss.backward()
            optimizer.step()

            if verbose
                println("Epoch: ", epoch, " Batch: ", i, " Loss: ", loss.item())
            end
            
            loss_i = Float64(loss.item())
            
            if isnan(loss_i)
                @warn "Loss is NaN"
                @show xbatch
                @show rbatch
                @show wbatch
                @show log_q
                error("Loss is NaN")
            end

            push!(loss_hist, loss_i)
        end
    end

    return q
end

function Distributions.fit(q::PyConditionalNormalizingFlow, xs, ws)
    model = q.model

    batch_size = q.params.batch_size
    lr = q.params.lr
    weight_decay = q.params.weight_decay
    verbose = q.params.verbose
    epochs = q.params.epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_hist = []

    n_samples = size(xs, 2)
    n_batches = ceil(Int, n_samples / batch_size)

    for epoch = 1:epochs
        # shuffle indices for a full pass through the dataset
        shuffled = randperm(n_samples)

        for i = 1:n_batches
            optimizer.zero_grad()
            
            batch_start = (i - 1) * batch_size + 1
            batch_end = min(i * batch_size, n_samples)
            batch_idxs = shuffled[batch_start:batch_end]
            current_batch_size = length(batch_idxs)

            xbatch = xs[:, batch_idxs]
            wbatch = ws[batch_idxs]

            xnp = np.array(xbatch')
            x = torch.tensor(xnp).float()

            wnp = np.array(wbatch)
            w = torch.tensor(wnp).float()
            
            # Compute loss
            log_q = torch.zeros(current_batch_size)
            z = x
            for k = length(model.flows)-1:-1:-1
                z, log_det = model.flows[k].inverse(z)
                log_q += log_det
            end
            log_q += model.q0.log_prob(z)
            norm = torch.sum(w)
            
            if Float64(norm.item()) == 0.0
                norm = torch.tensor(1.0).float()
            end
            loss = -torch.sum(log_q * w) / norm
            
            # Backpropagation and optimizer step
            loss.backward()
            optimizer.step()

            if verbose
                println("Epoch: ", epoch, " Batch: ", i, " Loss: ", loss.item())
            end
            
            loss_i = Float64(loss.item())
            
            if isnan(loss_i)
                @warn "Loss is NaN"
                @show xbatch
                @show wbatch
                @show log_q
                error("Loss is NaN")
            end

            push!(loss_hist, loss_i)
        end
    end

    return q
end

function Distributions.fit(q::PyConditionalNormalizingFlow, buffer::Buffer; augment=false, smoothing=0.1)
    model = q.model

    batch_size = q.params.batch_size
    lr = q.params.lr
    weight_decay = q.params.weight_decay
    verbose = q.params.verbose
    epochs = q.params.epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_hist = []

    n_samples = length(buffer)
    n_batches = ceil(Int, n_samples / batch_size)

    for epoch = 1:epochs
        buff = augment ? relabel_buffer_v2(buffer) : buffer

        rs = buff[:r]
        ρs = buff[:ρ]
        ws_ = buff[:w]
        xs = buff[:x]
        δ = [cdf(Logistic(0.0, smoothing), ri-ρi) for (ri, ρi) in zip(rs, ρs)]
        ws = exp.(log.(ws_) .+ log.(δ))
        

        # shuffle indices for a full pass through the dataset
        shuffled = randperm(n_samples)
        for i = 1:n_batches
            optimizer.zero_grad()
            
            batch_start = (i - 1) * batch_size + 1
            batch_end = min(i * batch_size, n_samples)
            batch_idxs = shuffled[batch_start:batch_end]
            current_batch_size = length(batch_idxs)

            xbatch = xs[:, batch_idxs]
            rbatch = repeat(rs[batch_idxs], 1, q.context_size)'
            wbatch = ws[batch_idxs]

            xnp = np.array(xbatch')
            x = torch.tensor(xnp).float()

            rnp = np.array(rbatch')
            r = torch.tensor(rnp).float()

            wnp = np.array(wbatch)
            w = torch.tensor(wnp).float()
            
            # Compute loss
            log_q = torch.zeros(current_batch_size)
            z = x
            for k = length(model.flows)-1:-1:-1
                z, log_det = model.flows[k].inverse(z, context=r)
                log_q += log_det
            end
            log_q += model.q0.log_prob(z, context=r)
            loss = -torch.sum(log_q * w) / torch.sum(w) # weighted loss
            
            # Backpropagation and optimizer step
            loss.backward()
            optimizer.step()

            if verbose
                println("Epoch: ", epoch, " Batch: ", i, " Loss: ", loss.item())
            end
            
            loss_i = Float64(loss.item())
            if isnan(loss_i)
                @warn "Loss is NaN"
                @show xbatch
                @show rbatch
                @show wbatch
                @show log_q
            end

            push!(loss_hist, loss_i)
        end
    end

    return q
end