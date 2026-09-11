# Experiment 2 offline diagnostics.
#
# Recomputes schedule-reach, sample-support, and emitted-threshold diagnostics
# from the stored Exp2 raw-state bundle without rerunning simulation or fitting.
#
# Usage:
#   RAW_STATE_DIR=<dir> julia --project=. paper/scripts/exp2_diagnostics.jl

using JSON, Serialization, Statistics, Printf, LinearAlgebra
using ARIS, Distributions

const PAPER = normpath(joinpath(@__DIR__, ".."))
const ROOT = get(ENV, "EXP2_RESULTS_DIR", joinpath(PAPER, "results", "exp2"))
const RAW_STATE = get(ENV, "RAW_STATE_DIR", joinpath(ROOT, "raw_state"))
const OUT  = get(ENV, "EXP2_DIAGNOSTICS_OUT", joinpath(ROOT, "diagnostics"))

# grid, reconstructed independently (same construction as problem() in paper/experiments/drivers/exp2_shell.jl)
function problem_grid(d::Int)
    PGRID = exp10.(range(log10(1e-3), log10(1e-6), length=16))
    TVEC  = Float64[sqrt(quantile(Chisq(d), 1 - p)) for p in PGRID]
    RCRIT = maximum(TVEC) + 0.5
    return (grid=RCRIT .- TVEC, rcrit=RCRIT, pgrid=PGRID)
end

function read_block(man, raw, name)
    for b in man["blocks"]
        if b["name"] == name
            off = b["offset"]; nb = b["nbytes"]
            v = reinterpret(Float64, raw[off+1 : off+nb])
            sh = Int.(b["shape"])
            return length(sh) == 1 ? Vector(v) : reshape(Vector(v), sh...)
        end
    end
    return nothing
end

rows = Dict{String,Any}()
for d in (2, 20), B in (25_000, 50_000)
    pr = problem_grid(d); G = pr.grid; GMAX = maximum(G); GMIN = minimum(G)
    cellkey = "d$(d)_B$(B)"
    cellrows = Any[]
    for seed in 1:10
        key = "$(cellkey)_seed$(seed)"
        manp = joinpath(RAW_STATE, key * ".manifest.json")
        binp = joinpath(RAW_STATE, key * ".arrays.bin")
        mdlp = joinpath(RAW_STATE, key * ".models.jls")
        (isfile(manp) && isfile(binp)) || (println("MISSING $key"); continue)
        man = JSON.parsefile(manp); raw = read(binp)
        ρ    = read_block(man, raw, "a_amortized/rho")
        w    = read_block(man, raw, "a_amortized/w")
        iter = read_block(man, raw, "a_amortized/iter")

        # γ^k trajectory from the serialized per-iteration checkpoints
        γtraj = Float64[]
        if isfile(mdlp)
            for e in Serialization.deserialize(mdlp)
                e["tag"] == "a_amortized" || continue
                γtraj = Float64[g for (_, g) in e["checkpoints"]]
            end
        end
        # as in exp2_shell.jl, γmin excludes checkpoint 1 (the untruncated prior batch)
        γmin      = length(γtraj) > 1 ? minimum(γtraj[2:end]) : NaN
        k_at_γmin = length(γtraj) > 1 ? argmin(γtraj[2:end]) + 1 : -1

        ρmin      = minimum(ρ)
        k_at_ρmin = Int(iter[argmin(ρ)])
        n_below_gmax = count(<=(GMAX), ρ)          # samples inside ANY evaluation threshold
        n_below_gmin = count(<=(GMIN), ρ)          # samples inside the TIGHTEST threshold (0.5)
        n_below_0    = count(<=(0.0), ρ)           # samples at/below the nominal failure target
        n_emitted  = count(γ -> γ >= γmin, G)
        n_positive = count(γ -> (γ >= γmin) && any(ρ .<= γ), G)

        push!(cellrows, Dict(
            "seed"=>seed, "rho_min"=>ρmin, "iter_at_rho_min"=>k_at_ρmin,
            "gamma_min"=>γmin, "iter_at_gamma_min"=>k_at_γmin,
            "n_iters"=>length(γtraj), "n_samples"=>length(ρ),
            "reached_target_0"=>(γmin <= 0.0), "discovered_gmin_le_gmax"=>(γmin <= GMAX),
            "n_below_gmax"=>n_below_gmax, "n_below_gridgmin_0p5"=>n_below_gmin,
            "n_below_zero"=>n_below_0,
            "n_thresholds_emitted"=>n_emitted, "n_thresholds_positive_support"=>n_positive,
            "gamma_traj_last5"=>round.(γtraj[max(1,end-4):end], digits=5)))
    end
    rows[cellkey] = Dict("d"=>d, "B"=>B, "grid_gmax"=>GMAX, "grid_gmin"=>GMIN,
                         "r_crit"=>pr.rcrit, "seeds"=>cellrows)
end

open(joinpath(OUT, "offline_exp2_diagnostics.json"), "w") do io
    JSON.print(io, rows, 2)
end

med(v) = isempty(v) ? NaN : median(v)
for cellkey in ("d2_B25000","d2_B50000","d20_B25000","d20_B50000")
    c = rows[cellkey]; S = c["seeds"]
    @printf("\n================ %s   (grid gmax=%.4f, tightest grid gmin=%.4f, r_crit=%.6f)\n",
            cellkey, c["grid_gmax"], c["grid_gmin"], c["r_crit"])
    @printf("%-5s %-10s %-8s %-10s %-8s %-7s %-7s %-9s %-8s %-7s %-7s\n",
            "seed","rho_min","it(rho)","gamma_min","it(g)","disc","targ","n<=gmax","n<=0.5","n_emit","n_pos")
    for r in S
        @printf("%-5d %-10.5f %-8d %-10.5f %-8d %-7s %-7s %-9d %-8d %-7d %-7d\n",
                r["seed"], r["rho_min"], r["iter_at_rho_min"], r["gamma_min"], r["iter_at_gamma_min"],
                r["discovered_gmin_le_gmax"] ? "yes" : "no", r["reached_target_0"] ? "yes" : "no",
                r["n_below_gmax"], r["n_below_gridgmin_0p5"],
                r["n_thresholds_emitted"], r["n_thresholds_positive_support"])
    end
    @printf("  MEDIAN  rho_min=%.5f  gamma_min=%.5f  n<=gmax=%.1f  n<=0.5=%.1f  n_pos=%.1f\n",
            med([r["rho_min"] for r in S]), med([r["gamma_min"] for r in S]),
            med([Float64(r["n_below_gmax"]) for r in S]), med([Float64(r["n_below_gridgmin_0p5"]) for r in S]),
            med([Float64(r["n_thresholds_positive_support"]) for r in S]))
    @printf("  RANGE   rho_min=[%.5f, %.5f]  gamma_min=[%.5f, %.5f]\n",
            minimum(r["rho_min"] for r in S), maximum(r["rho_min"] for r in S),
            minimum(r["gamma_min"] for r in S), maximum(r["gamma_min"] for r in S))
    @printf("  COUNTS  discovered=%d/%d  reached_target=%d/%d  any_sample<=gmax=%d/%d  any_sample<=0.5=%d/%d\n",
            count(r->r["discovered_gmin_le_gmax"], S), length(S),
            count(r->r["reached_target_0"], S), length(S),
            count(r->r["n_below_gmax"]>0, S), length(S),
            count(r->r["n_below_gridgmin_0p5"]>0, S), length(S))
end
