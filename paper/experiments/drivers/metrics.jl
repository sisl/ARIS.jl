# Shared metric utilities for experiment drivers.
#
# ILE is the mean absolute log error over grid points with finite positive
# estimates and positive reference values. The returned records include the
# contributing-point mask and grid counts explicitly.
module FinalMetrics

using Statistics

const ILE_FORMULA = "ILE = mean_{γ ∈ used} |log est(γ) − log truth(γ)|; " *
                    "used = {γ : est(γ) !== nothing && est(γ) > 0 && truth(γ) > 0}"

"""
    curve_metrics(curve, grid, truthf) -> Dict

Compute per-threshold metrics and ILE over valid grid points.
"""
function curve_metrics(curve::AbstractDict, grid::AbstractVector, truthf)
    per = Vector{Dict{String,Any}}()
    used_mask = Bool[]
    logerrs = Float64[]
    for γ in grid
        est = get(curve, γ, nothing)
        T = truthf(γ)
        ok = (est !== nothing) && (est isa Real) && isfinite(est) && est > 0 && T > 0
        push!(used_mask, ok)
        if ok; push!(logerrs, abs(log(est) - log(T))); end
        push!(per, Dict{String,Any}(
            "gamma"        => γ,
            "truth"        => T,
            "estimate"     => est,
            "signed_relerr"=> (est === nothing || T <= 0) ? nothing : (est - T) / T,
            "abs_relerr"   => (est === nothing || T <= 0) ? nothing : abs(est - T) / T,
            "contributes_to_ile" => ok,
        ))
    end
    return Dict{String,Any}(
        "per_threshold" => per,
        "ile"           => isempty(logerrs) ? nothing : mean(logerrs),
        "n_grid_total"  => length(grid),
        "n_grid_used"   => count(used_mask),
        "ile_mask"      => used_mask,
        "ile_formula"   => ILE_FORMULA,
        "n_cells_nonnull" => count(γ -> get(curve, γ, nothing) !== nothing, grid),
    )
end

"""ESS fraction: (Σw)² / (n · Σw²). Returns 0.0 for an empty or all-zero weight vector."""
function ess_frac(w::AbstractVector)
    (isempty(w) || sum(abs2, w) == 0) && return 0.0
    return sum(w)^2 / (length(w) * sum(abs2, w))
end

"""Raw (unnormalised) ESS: (Σw)² / Σw²."""
function ess_raw(w::AbstractVector)
    (isempty(w) || sum(abs2, w) == 0) && return 0.0
    return sum(w)^2 / sum(abs2, w)
end

"""ESS over samples satisfying ρ ≤ γ."""
function tail_ess(ρ::AbstractVector, w::AbstractVector, γ::Real)
    m = ρ .<= γ
    any(m) || return 0.0
    return ess_raw(w[m])
end

"""Median condition number of the fitted mixture's component covariances; NaN if unavailable."""
function cov_cond_median(model)
    try
        comps = model.trajectory_model.gmm.mixture.components
        κ = Float64[cond(Matrix(cov(c))) for c in comps]
        κ = filter(isfinite, κ)
        return isempty(κ) ? NaN : median(κ)
    catch
        return NaN
    end
end

"""
    discovery(γmin, grid; target) -> Dict

Record whether the adaptive schedule reaches the evaluation grid and target.
"""
function discovery(γmin::Real, grid::AbstractVector; target::Real=0.0)
    gmax = maximum(grid)
    return Dict{String,Any}(
        "gamma_min"        => γmin,
        "gamma_target"     => target,
        "gamma_min_gap"    => γmin - target,
        "grid_gmax"        => gmax,
        "grid_gmin"        => minimum(grid),
        "discovered"       => γmin <= gmax,
        "reached_target"   => γmin <= target,
    )
end

"""Aggregate finite numeric values from a field and report the contributing count."""
function agg(rows, field; by=identity)
    v = Any[]
    for r in rows
        x = get(r, field, nothing)
        (x isa Real && isfinite(x)) && push!(v, by(x))
    end
    isempty(v) && return Dict{String,Any}("n"=>0, "median"=>nothing, "mean"=>nothing,
                                          "min"=>nothing, "max"=>nothing, "std"=>nothing)
    return Dict{String,Any}("n"=>length(v), "median"=>median(v), "mean"=>mean(v),
        "min"=>minimum(v), "max"=>maximum(v),
        "std"=>length(v) > 1 ? std(v) : nothing)
end

"""Seed accounting for a method: requested, completed, failed, and per-metric contribution."""
function seed_accounting(requested::AbstractVector, rows; metric_fields=("ile",))
    completed = [r for r in rows if get(r, "ok", false) === true]
    failed    = [r for r in rows if get(r, "ok", false) !== true]
    d = Dict{String,Any}(
        "seeds_requested"  => collect(requested),
        "n_requested"      => length(requested),
        "n_completed"      => length(completed),
        "n_failed"         => length(failed),
        "failed_seeds"     => [get(r, "seed", nothing) for r in failed],
        "failure_records"  => [Dict("seed"=>get(r,"seed",nothing), "error"=>get(r,"error",nothing))
                               for r in failed],
    )
    for f in metric_fields
        d["n_contributing_$(f)"] = count(r -> (get(r, f, nothing) isa Real) &&
                                              isfinite(get(r, f, NaN)), completed)
    end
    return d
end

export curve_metrics, ess_frac, ess_raw, tail_ess, cov_cond_median, discovery, agg,
       seed_accounting, ILE_FORMULA

end # module
