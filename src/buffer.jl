mutable struct Buffer
    data::Dict{Symbol, AbstractArray}
    elements::Int64
    next_ind::Int64
    indices::Vector{Int}
    total_count::Int
    ignored_keys
end

function Buffer(xdim::Int, sdim::Int, depth::Int, capacity::Int, ignored_keys::Vector)
    # `:defensive` records whether each sample came from the defensive
    # component of a mixture proposal. It defaults to `false` for
    # non-mixture proposals.
    all_keys = [:x, :ρ, :s, :w, :r, :logpdf, :defensive]
    ignored_set = Set(ignored_keys)
    
    # Initialize data dict only with keys we want to store
    data = Dict{Symbol, AbstractArray}()
    if !(:x in ignored_keys)
        data[:x] = zeros(Float32, xdim, capacity)
    end
    if !(:ρ in ignored_keys)
        data[:ρ] = zeros(Float32, capacity)
    end
    if !(:s in ignored_keys)
        data[:s] = zeros(Float32, sdim, depth, capacity)
    end
    if !(:w in ignored_keys)
        data[:w] = zeros(Float32, capacity)
    end
    if !(:r in ignored_keys)
        data[:r] = zeros(Float32, capacity)
    end
    if !(:logpdf in ignored_keys)
        data[:logpdf] = zeros(Float32, capacity)
    end
    if !(:defensive in ignored_keys)
        data[:defensive] = falses(capacity)
    end

    indices = collect(1:capacity)
    return Buffer(data, 0, 1, indices, 0, ignored_set)
end

capacity(buffer::Buffer) = length(buffer.indices)

Base.getindex(buffer::Buffer, key::Symbol) = bslice(buffer.data[key], 1:buffer.elements)

Base.keys(buffer::Buffer) = keys(buffer.data)

Base.first(buffer::Buffer) = first(buffer.data)

Base.haskey(buffer::Buffer, key::Symbol) = haskey(buffer.data, key)

Base.length(buffer::Buffer) = buffer.elements

# Single-sample push!; keys not stored (ignored keys) are skipped
function Base.push!(buffer::Buffer, x::AbstractVector, ρ::Real, s::AbstractArray, w::Real, rs::Real, logpdf::Real, defensive::Bool=false)
    if haskey(buffer.data, :defensive)
        buffer.data[:defensive][buffer.next_ind] = defensive
    end
    if haskey(buffer.data, :x)
        buffer.data[:x][:, buffer.next_ind] = x
    end
    if haskey(buffer.data, :ρ)
        buffer.data[:ρ][buffer.next_ind] = ρ
    end
    if haskey(buffer.data, :s)
        buffer.data[:s][:, :, buffer.next_ind] .= s
    end
    if haskey(buffer.data, :w)
        buffer.data[:w][buffer.next_ind] = w
    end
    if haskey(buffer.data, :r)
        buffer.data[:r][buffer.next_ind] = rs
    end
    if haskey(buffer.data, :logpdf)
        buffer.data[:logpdf][buffer.next_ind] = logpdf
    end
    
    buffer.next_ind = mod1(buffer.next_ind + 1, length(buffer.indices))
    buffer.elements += 1
    buffer.total_count += 1
end

# Batch push!, including the input condition 'rs'
function Base.push!(buffer::Buffer, x::AbstractMatrix, ρ::AbstractVector, s::AbstractArray, w::AbstractVector, rs::AbstractVector, logpdf::AbstractVector, defensive::AbstractVector{Bool}=falses(length(ρ)))
    for i in 1:size(x, 2)
        push!(buffer, x[:, i], ρ[i], bslice(s, i), w[i], rs[i], logpdf[i], defensive[i])
    end
end
