# EXPERIMENT 1 — K-scaling (grid-density amortization).
#
# Measured quantity: estimation error as the number of requested thresholds K increases at FIXED
# total simulator budget B.
#
# Arms (identical total budget B at every K):
#   amortized     — exactly ONE adaptive run per seed; the SAME run is re-indexed at every K grid.
#   perthreshold  — K independent fits, budget B/K each; total evaluations constrained to B.
#
# Quadrature: nodes = 1000 for EVERY K and every amortized result. Recorded in config.json.
#
# The last grid point is constructed as the target γ = 0 but evaluates to -5.77e-15 in floating
# point, so the amortized arm's `γ >= γmin` gate stores null there; exp1_endpoint_curves.jl
# rebuilds that endpoint from the raw-state bundle.
#
# Usage: julia --project=. paper/experiments/drivers/exp1_kscaling.jl [nseeds] [budget] [nodes]
# Reruns write to paper/rerun/exp1_kscaling/ (ARIS_OUTROOT overrides the output root).
using Distributed
const NSEEDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
const BUDGET = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 25_000
const NODES  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1000
const KS     = [10, 25, 50, 100]
const SEEDS  = collect(1:NSEEDS)
addprocs(max(0, min(8, Sys.CPU_THREADS - 2) - nworkers() + 1))

@everywhere include(joinpath(@__DIR__, "metrics.jl"))
@everywhere using .FinalMetrics
@everywhere include(joinpath(@__DIR__, "rawstate.jl"))
@everywhere using .RawState

@everywhere begin
    using ARIS
    using Distributions, Random, Statistics, LinearAlgebra, JSON
    LinearAlgebra.BLAS.set_num_threads(1)
    include(joinpath(@__DIR__, "..", "..", "..", "examples", "problems", "toy.jl"))
    include(joinpath(@__DIR__, "..", "balance.jl"))

    const SYS   = UnimodalToy(3.0)
    const PRIOR = System.prior(SYS)
    truthf(γ) = cdf(Normal(), -(3.0 - γ))^2
    # identical endpoints for every K; density is the only thing that changes
    const P_LO, P_HI = 1e-3, cdf(Normal(), -3.0)^2
    grid(K) = [3.0 + quantile(Normal(), sqrt(p))
               for p in exp10.(range(log10(P_LO), log10(P_HI), length=K))]


    # streaming persistence: each worker writes its own row the moment it completes, so a teardown
    # costs only in-flight runs and a relaunch resumes from what is already banked.
    _san_w(x::Float64) = isfinite(x) ? x : nothing
    _san_w(x::AbstractDict) = Dict(string(k) => _san_w(v) for (k, v) in x)
    _san_w(x::AbstractVector) = [_san_w(v) for v in x]
    _san_w(x) = x
    function persist_row(key, row)
        isdefined(Main, :OUTDIR) || return row
        p = joinpath(Main.OUTDIR, "raw", string(key, ".json")); tmp = p * ".tmp"
        open(tmp, "w") do io; JSON.print(io, _san_w(row), 2); end
        mv(tmp, p; force=true); return row
    end
    banked(key) = isdefined(Main, :OUTDIR) &&
        (p = joinpath(Main.OUTDIR, "raw", string(key, ".json")); isfile(p) && filesize(p) > 0)

    const BUDGET_W = $BUDGET
    const NODES_W  = $NODES
    const KS_W     = $KS

    "One adaptive run. Returns buffer, checkpoints, and the ACTUAL robustness-evaluation count."
    function one_run(seed, budget; target=0.0, N=1000)
        Random.seed!(seed); reset_fit_stats!()
        iters = max(2, budget ÷ N)
        CHK = Any[]; t0 = time()
        cb = (cv, D) -> push!(CHK, (deepcopy(cv.model), Float64(cv.sampling_strategy.value)))
        model = JointModel(Normal(), ConditionalEMGMM(2, 3, 3))
        cv = ConditionalValidation(model=model, xdim=2, depth=1, sdim=2, n_samples=N, n_iter=iters,
            sampling_strategy=QuantileSampling(0.2; target=target), max_weight=10.0, log_callback=cb)
        train!(cv, SYS)
        ρ = Float64.(cv.buffer[:ρ])
        return (CHK=CHK, X=Float64.(cv.buffer[:x]), ρ=ρ, w=Float64.(cv.buffer[:w]),
                logq=Float64.(cv.buffer[:logpdf]), def=Vector{Bool}(cv.buffer[:defensive]),
                iter=repeat(1:iters, inner=N)[1:length(ρ)],
                N=N, iters=iters, n_evals=length(ρ), secs=time() - t0)
    end

    function run_seed(seed)
        key = "seed$(seed)"
        banked(key) && return JSON.parsefile(joinpath(Main.OUTDIR,"raw",key*".json"))
        t_all = time()
        out = Dict{String,Any}("seed"=>seed, "ok"=>false)
        REC = RunRecorder()      # raw state for offline re-evaluation of the estimators
        try
            # ---------- amortized: ONE run, ONE shared re-index, re-thresholded at every K ----------
            r = one_run(seed, BUDGET_W)
            record_run!(REC, "amortized"; X=r.X, rho=r.ρ, w=r.w, logq=r.logq, iter=r.iter,
                        defmask=r.def, checkpoints=r.CHK,
                        extra=Dict("budget"=>BUDGET_W, "N"=>r.N, "iters"=>r.iters))
            K_ = length(r.ρ) ÷ r.N
            γmin = minimum(γk for (_, γk) in r.CHK[2:end])
            GMAX = maximum(grid(KS_W[1]))     # identical across K by construction
            t_ri = time()
            idx = findall(r.ρ .<= GMAX); Xi = r.X[:, idx]; ρi = r.ρ[idx]
            p_i = pdf(PRIOR, Xi); qsum = copy(p_i)
            for k in 2:K_
                mk, γk = r.CHK[k]
                qsum .+= proposal_pdf(mk, γk, Xi, PRIOR; nodes=NODES_W)
            end
            wb = p_i ./ (qsum ./ K_); secs_reindex = time() - t_ri
            Ntot = length(r.ρ)

            amort = Dict{String,Any}()
            for K in KS_W
                G = grid(K)
                curve = Dict(γ => (γ >= γmin ? sum(wb[ρi .<= γ]) / Ntot : nothing) for γ in G)
                m = curve_metrics(curve, G, truthf)
                m["gamma_grid"] = G
                merge!(m, discovery(γmin, G))
                m["n_evals_actual"] = r.n_evals
                m["n_runs"] = 1
                m["secs_sim"] = r.secs
                m["secs_reindex_shared"] = secs_reindex
                m["secs_total"] = r.secs + secs_reindex
                m["fit_ess_reindex"] = ess_frac(wb)
                m["tail_ess_at_target"] = tail_ess(ρi, wb, 0.0)
                amort[string(K)] = m
            end

            # ---------- per-threshold: K independent fits, B/K each ----------
            per = Dict{String,Any}()
            for K in KS_W
                G = grid(K); perb = BUDGET_W ÷ K; t2 = time()
                curve = Dict{Float64,Any}(); evals = 0
                for γ in G
                    N = min(500, max(50, perb ÷ 10))
                    rb = one_run(seed * 100_000 + hash(γ) % 1000, perb; target=γ, N=N)
                    record_run!(REC, string("perthreshold/K", K, "/g", round(γ, digits=6));
                                X=rb.X, rho=rb.ρ, w=rb.w, logq=rb.logq, iter=rb.iter,
                                defmask=rb.def, checkpoints=rb.CHK,
                                extra=Dict("K"=>K, "gamma"=>γ, "budget"=>perb, "N"=>N))
                    curve[γ] = sum((rb.ρ .<= γ) .* rb.w) / length(rb.ρ)
                    evals += rb.n_evals
                end
                m = curve_metrics(curve, G, truthf)
                m["gamma_grid"] = G
                m["n_evals_actual"] = evals
                m["n_runs"] = K
                m["per_threshold_budget"] = perb
                m["secs_total"] = time() - t2
                m["secs_sim"] = time() - t2
                per[string(K)] = m
            end
            out["amortized"] = amort; out["perthreshold"] = per
            out["secs_seed_total"] = time() - t_all
            out["ok"] = true
        catch e
            out["error"] = sprint(showerror, e)
            out["backtrace"] = string.(stacktrace(catch_backtrace())[1:min(8,end)])
        end
        # Write the raw-state bundle before the JSON row, so a row on disk always has its
        # bundle beside it. Bundle failures are recorded, never fatal to the run.
        if isdefined(Main, :OUTDIR)
            try
                merge!(out, write_bundle(REC, Main.OUTDIR, "seed$(seed)",
                        Dict("seed"=>seed, "git_sha"=>(isdefined(Main, :GIT_SHA) ? Main.GIT_SHA : "unknown"),
                             "marginal"=>marginal_settings())))
            catch e
                out["raw_state_error"] = sprint(showerror, e)
            end
        end
        return persist_row("seed$(seed)", out)
    end
end

using Printf, JSON, Statistics
include(joinpath(@__DIR__, "provenance.jl")); using .RunProvenance
include(joinpath(@__DIR__, "metrics.jl")); using .FinalMetrics

config = Dict{String,Any}(
    "experiment" => "exp1_kscaling",
    "measured_quantity" => "estimation error vs number of requested thresholds K at fixed total budget",
    "problem" => "unimodal_toy", "problem_def" => "rho = 3 - min(x1,x2), x ~ N(0,I2)",
    "truth" => "P(rho<=g) = Phi(-(3-g))^2 (closed form)",
    "dimension" => 2, "budget_total" => BUDGET, "batch_size_amortized" => 1000,
    "n_iter_amortized" => BUDGET ÷ 1000, "Ks" => KS, "seeds" => SEEDS, "n_seeds" => NSEEDS,
    "grid_endpoints_P" => [1e-3, cdf(Normal(), -3.0)^2],
    "grid_note" => "identical endpoints for every K; density is the only difference",
    "quadrature_nodes" => NODES,
    "quadrature_note" => "nodes=1000 (default quadrature resolution)",
    "proposal_family" => "JointModel(Normal(), ConditionalEMGMM(k=2))",
    "n_mixture_components" => 2, "adaptive_quantile" => 0.2, "target_threshold" => 0.0,
    "max_weight_fit" => 10.0, "sigma_smoothing" => 0.0, "defensive_eta" => 0.0,
    "arms" => ["amortized", "perthreshold"],
    "budget_accounting" => "n_evals_actual recorded per arm per K; amortized = 1 run of B; perthreshold = K runs of B/K",
    "ile_formula" => FinalMetrics.ILE_FORMULA,
)
dir, meta = init_group("exp1_kscaling", config)
@printf("exp1_kscaling → %s\n  %d seeds × (1 amortized + %d perthreshold grids), B=%d, nodes=%d, %d workers\n",
        dir, NSEEDS, length(KS), BUDGET, NODES, nworkers())

@everywhere const OUTDIR = $dir
rows = pmap(Main.run_seed, SEEDS)

# ---------------- aggregate (recomputed from rows only) ----------------
summary = Dict{String,Any}("experiment"=>"exp1_kscaling", "config"=>config,
                           "ile_formula"=>FinalMetrics.ILE_FORMULA, "cells"=>Dict{String,Any}())
ok = [r for r in rows if r["ok"] === true]
summary["seed_accounting"] = Dict("n_requested"=>NSEEDS, "n_completed"=>length(ok),
    "n_failed"=>NSEEDS - length(ok), "seeds_requested"=>SEEDS,
    "failed_seeds"=>[r["seed"] for r in rows if r["ok"] !== true],
    "failure_records"=>[Dict("seed"=>r["seed"], "error"=>get(r,"error",nothing))
                        for r in rows if r["ok"] !== true])
for arm in ("amortized", "perthreshold"), K in KS
    cells = [r[arm][string(K)] for r in ok if haskey(r, arm)]
    if isempty(cells)
        summary["cells"]["$(arm)_K$(K)"] = Dict{String,Any}("arm"=>arm, "K"=>K, "n_seeds"=>0,
            "n_contributing_ile"=>0, "ile_median"=>nothing, "note"=>"no completed seeds")
        continue
    end
    iles  = [c["ile"] for c in cells if c["ile"] !== nothing]
    summary["cells"]["$(arm)_K$(K)"] = Dict{String,Any}(
        "arm"=>arm, "K"=>K, "n_seeds"=>length(cells),
        "n_contributing_ile"=>length(iles),
        "ile_median"=>isempty(iles) ? nothing : median(iles),
        "ile_mean"=>isempty(iles) ? nothing : mean(iles),
        "ile_std"=>length(iles)>1 ? std(iles) : nothing,
        "ile_per_seed"=>[c["ile"] for c in cells],
        "n_grid_total"=>cells[1]["n_grid_total"],
        "n_grid_used_per_seed"=>[c["n_grid_used"] for c in cells],
        "n_evals_actual_per_seed"=>[c["n_evals_actual"] for c in cells],
        "n_evals_actual_median"=>median([c["n_evals_actual"] for c in cells]),
        "n_runs"=>cells[1]["n_runs"],
        "secs_total_median"=>median([c["secs_total"] for c in cells]),
    )
end
write_summary(dir, summary)

@printf("\n%-14s %-5s %-4s %-12s %-10s %-12s %s\n", "arm","K","n","ILE median","ILE std","evals(med)","n_runs")
for arm in ("amortized","perthreshold"), K in KS
    c = summary["cells"]["$(arm)_K$(K)"]
    @printf("%-14s %-5d %-4d %-12s %-10s %-12s %d\n", arm, K, c["n_contributing_ile"],
            c["ile_median"] === nothing ? "—" : string(round(c["ile_median"], sigdigits=6)),
            c["ile_std"] === nothing ? "—" : string(round(c["ile_std"], sigdigits=4)),
            string(round(Int, c["n_evals_actual_median"])), c["n_runs"])
end
println("EXP1_DONE")
