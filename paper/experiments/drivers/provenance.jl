# Shared experiment-output utilities.
#
# Each experiment writes `config.json` and `meta.json` through this module.
# Reruns default to `paper/rerun/<group>/`; set `ARIS_OUTROOT` to override.
module RunProvenance

using JSON, Dates, Pkg, LinearAlgebra

const PAPER   = normpath(joinpath(@__DIR__, "..", ".."))
const REPO    = normpath(joinpath(PAPER, ".."))
const OUTROOT = get(ENV, "ARIS_OUTROOT", joinpath(PAPER, "rerun"))

function pkg_versions(names::Vector{String})
    out = Dict{String,Any}()
    try
        for (_, info) in Pkg.dependencies()
            info.name in names && (out[info.name] = string(info.version))
        end
    catch e
        out["_error"] = sprint(showerror, e)
    end
    return out
end

const TRACKED_PKGS = ["Distributions","StatsBase","Clustering","QuadGK","ExpectationMaximization",
    "Optim","ForwardDiff","JSON","StructuredGaussianMixtures","AdversarialDriving",
    "AutomotiveSimulator","POMDPs","POMDPTools","Crux","Arpack","GaussianMixtures"]

"""
    init_group(group, config) -> (dir, meta)

Creates `OUTROOT/<group>/raw` and writes config.json and meta.json (Julia version, threads,
package versions and the active environment, relative to the repository root).
"""
function init_group(group::AbstractString, config::AbstractDict)
    dir = joinpath(OUTROOT, group)
    mkpath(dir); mkpath(joinpath(dir, "raw"))
    meta = Dict{String,Any}(
        "experiment_group" => group,
        "timestamp_start"  => string(now()),
        "julia_version"    => string(VERSION),
        "nthreads"         => Threads.nthreads(),
        "blas_threads"     => (try LinearAlgebra.BLAS.get_num_threads() catch; nothing end),
        "pkg_versions"     => pkg_versions(TRACKED_PKGS),
        "environment"      => relpath(Base.active_project(), REPO),
    )
    _atomic_json(joinpath(dir, "config.json"), config)
    _atomic_json(joinpath(dir, "meta.json"), meta)
    return dir, meta
end

function _atomic_json(path, obj)
    tmp = path * ".tmp"
    open(tmp, "w") do io; JSON.print(io, _san(obj), 2); end
    mv(tmp, path; force=true)
    return path
end

# Convert non-finite values to JSON `null` recursively.
_san(x::Float64) = isfinite(x) ? x : nothing
_san(x::AbstractDict) = Dict(string(k) => _san(v) for (k, v) in x)
_san(x::AbstractVector) = [_san(v) for v in x]
_san(x) = x

"""Write one raw per-seed/per-cell record, atomically. Returns the path."""
write_raw(dir, key, row) = _atomic_json(joinpath(dir, "raw", string(key, ".json")), row)

"""Write the group's aggregate summary."""
write_summary(dir, summary) = _atomic_json(joinpath(dir, "summary.json"), summary)

export init_group, write_raw, write_summary, OUTROOT

end # module
