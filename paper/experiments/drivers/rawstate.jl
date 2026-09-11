# Raw-state persistence for offline estimator re-evaluation.
#
# Each bundle stores per-sample inputs, robustness values, proposal-of-origin
# information, importance weights, and proposal checkpoints. This allows
# estimator aggregation and marginal-density calculations to be recomputed
# without rerunning the simulator.
#
# Sample arrays are stored in a compact binary format with a JSON manifest;
# proposal checkpoints are serialized separately.

module RawState

using JSON, Serialization

export RunRecorder, record_run!, write_bundle, rawstate_scope

mutable struct RunRecorder
    runs::Vector{Dict{String,Any}}
    blocks::Vector{Dict{String,Any}}
    payload::Vector{UInt8}
    models::Vector{Any}
end
RunRecorder() = RunRecorder(Dict{String,Any}[], Dict{String,Any}[], UInt8[], Any[])

_f64(v::AbstractArray) = Array{Float64}(v)

function _add_block!(rec::RunRecorder, name::AbstractString, A::AbstractArray)
    B = _f64(A)
    off = length(rec.payload)
    bytes = reinterpret(UInt8, vec(B))
    append!(rec.payload, bytes)
    push!(rec.blocks, Dict{String,Any}(
        "name" => name, "dtype" => "float64", "layout" => "column-major",
        "shape" => collect(size(B)), "offset" => off, "nbytes" => length(bytes)))
    return nothing
end

"""
    record_run!(rec, tag; X, rho, w, logq, iter, defmask=nothing, checkpoints=nothing, extra=Dict())

Record one adaptive run and its optional proposal checkpoints.
"""
function record_run!(rec::RunRecorder, tag::AbstractString;
                     X, rho, w, logq, iter,
                     defmask = nothing, checkpoints = nothing, extra = Dict{String,Any}())
    _add_block!(rec, "$(tag)/X",    X)
    _add_block!(rec, "$(tag)/rho",  rho)
    _add_block!(rec, "$(tag)/w",    w)
    _add_block!(rec, "$(tag)/logq", logq)
    _add_block!(rec, "$(tag)/iter", iter)
    defmask === nothing || _add_block!(rec, "$(tag)/defensive", Float64.(defmask))

    midx = -1
    if checkpoints !== nothing
        push!(rec.models, Dict("tag" => tag,
                               "checkpoints" => [(deepcopy(m), Float64(g)) for (m, g) in checkpoints]))
        midx = length(rec.models)
    end
    push!(rec.runs, merge(Dict{String,Any}(
        "tag" => tag, "n_samples_total" => length(rho),
        "d" => size(X, 1), "n_checkpoints" => checkpoints === nothing ? 0 : length(checkpoints),
        "models_index" => midx), Dict{String,Any}(string(k) => v for (k, v) in extra)))
    return rec
end

"""
    write_bundle(rec, outdir, key, meta) -> Dict

Write the raw-state bundle under `<outdir>/raw_state/`.
"""
function write_bundle(rec::RunRecorder, outdir::AbstractString, key::AbstractString,
                      meta::AbstractDict = Dict{String,Any}())
    dir = joinpath(outdir, "raw_state"); mkpath(dir)
    pb = joinpath(dir, "$(key).arrays.bin")
    pm = joinpath(dir, "$(key).manifest.json")
    pj = joinpath(dir, "$(key).models.jls")

    open(pb * ".tmp", "w") do io; write(io, rec.payload); end
    mv(pb * ".tmp", pb; force = true)

    if !isempty(rec.models)
        Serialization.serialize(pj * ".tmp", rec.models)
        mv(pj * ".tmp", pj; force = true)
    end

    man = Dict{String,Any}(
        "key" => key, "runs" => rec.runs, "blocks" => rec.blocks,
        "arrays_file" => basename(pb), "models_file" => isempty(rec.models) ? nothing : basename(pj),
        "total_bytes" => length(rec.payload),
        "read_note" => "blocks are Float64, column-major, at byte `offset` for `nbytes` in arrays_file",
        "scope" => rawstate_scope(),
        "meta" => Dict{String,Any}(string(k) => v for (k, v) in meta))
    open(pm * ".tmp", "w") do io; JSON.print(io, man, 2); end
    mv(pm * ".tmp", pm; force = true)

    return Dict{String,Any}("raw_state_key" => key, "raw_state_bytes" => length(rec.payload),
                            "raw_state_runs" => length(rec.runs),
                            "raw_state_models" => length(rec.models))
end

"""
    rawstate_scope() -> Dict

Describe the contents and supported uses of a raw-state bundle.
"""
rawstate_scope() = Dict{String,Any}(
    "persisted" => ["x", "rho(x)", "iteration/proposal-of-origin index", "raw p/q weight",
                    "stored log q", "defensive indicator (when the proposal is a mixture)",
                    "per-iteration proposal checkpoints with their gamma_k",
                    "seed", "git sha", "config (in the group's config.json/meta.json)"],
    "not_persisted" => ["simulator state trajectories (s)",
                        "AMS baseline internals (a splitting method: no proposal density exists)"],
    "sufficient_for" => ["recomputing importance weights under a different marginal-density " *
                         "evaluator or quadrature setting",
                         "re-running balance-MIS / per-threshold aggregation at any gamma grid",
                         "re-deriving ILE, discovery and ESS diagnostics"],
    "insufficient_for" => ["anything requiring new robustness evaluations"],
)

end # module
