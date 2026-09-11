# Experiment 2: radial-shell benchmark.
#
# Evaluates curve-estimation accuracy and adaptive schedule reach under
# isotropic rare-event geometry.
#
# The amortized method reports a pooled balance-MIS estimate at threshold γ
# only when its adaptive schedule has reached γ. Its coverage therefore
# measures schedule reach rather than estimator availability.
#
# The per-threshold baseline reports its selected estimate at every threshold,
# while AMS reports estimates only within the range of levels it reaches.
#
# Usage:
#   julia --project=. paper/experiments/drivers/exp2_shell.jl [nseeds]
#
# Reruns default to `paper/rerun/exp2_shell/`.

using Distributed
const NSEEDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
const SEEDS  = collect(1:NSEEDS)
const CELLS  = [(d=2, B=25_000), (d=2, B=50_000), (d=20, B=25_000), (d=20, B=50_000)]
const NODES  = 1000
addprocs(max(0, min(8, Sys.CPU_THREADS - 2) - nworkers() + 1))

@everywhere include(joinpath(@__DIR__, "metrics.jl"))
@everywhere using .FinalMetrics
@everywhere include(joinpath(@__DIR__, "rawstate.jl"))
@everywhere using .RawState

@everywhere begin
    using ARIS
    using Distributions, Random, Statistics, LinearAlgebra, JSON
    LinearAlgebra.BLAS.set_num_threads(1)
    include(joinpath(@__DIR__, "..", "..", "..", "examples", "problems", "radial_shell.jl"))
    include(joinpath(@__DIR__, "..", "balance.jl"))
    include(joinpath(@__DIR__, "..", "ams_curve.jl"))

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

    const NODES_W = $NODES

    "Problem handle for a cell: system, prior, closed-form truth and γ grid."
    function problem(d::Int)
        PGRID = exp10.(range(log10(1e-3), log10(1e-6), length=16))
        TVEC  = Float64[sqrt(quantile(Chisq(d), 1 - p)) for p in PGRID]
        RCRIT = maximum(TVEC) + 0.5
        sys   = RadialShell(; r_crit=RCRIT, d=d)
        return (sys=sys, prior=System.prior(sys), grid=RCRIT .- TVEC, rcrit=RCRIT, pgrid=PGRID,
                truth=(γ -> shell_truth(sys, γ)))
    end

    function one_run(pr, seed, budget; target=0.0, N=1000)
        Random.seed!(seed); reset_fit_stats!()
        iters = max(2, budget ÷ N); CHK = Any[]; t0 = time()
        cb = (cv, D) -> push!(CHK, (deepcopy(cv.model), Float64(cv.sampling_strategy.value)))
        d = pr.sys.d
        model = JointModel(Normal(), ConditionalEMGMM(2, d + 1, d + 1))
        cv = ConditionalValidation(model=model, xdim=d, depth=1, sdim=d, n_samples=N, n_iter=iters,
            sampling_strategy=QuantileSampling(0.2; target=target), max_weight=10.0, log_callback=cb)
        train!(cv, pr.sys)
        ρ = Float64.(cv.buffer[:ρ])
        return (CHK=CHK, X=Float64.(cv.buffer[:x]), ρ=ρ, w=Float64.(cv.buffer[:w]),
                logq=Float64.(cv.buffer[:logpdf]), def=Vector{Bool}(cv.buffer[:defensive]),
                iter=repeat(1:iters, inner=N)[1:length(ρ)],
                N=N, iters=iters, n_evals=length(ρ), secs=time() - t0)
    end

    function run_cell_seed(d::Int, B::Int, seed::Int)
        key = "d$(d)_B$(B)_seed$(seed)"
        banked(key) && return JSON.parsefile(joinpath(Main.OUTDIR,"raw",key*".json"))
        pr = problem(d); G = pr.grid; GMAX = maximum(G)
        row = Dict{String,Any}("d"=>d, "budget"=>B, "seed"=>seed, "ok"=>false,
                               "r_crit"=>pr.rcrit, "gamma_grid"=>G)
        REC = RunRecorder()      # raw state for offline re-evaluation of the estimators
        try
            # ---------------- arm (a) amortized ----------------
            ta = time(); ra = one_run(pr, seed, B); secs_a = time() - ta
            record_run!(REC, "a_amortized"; X=ra.X, rho=ra.ρ, w=ra.w, logq=ra.logq,
                        iter=ra.iter, defmask=ra.def, checkpoints=ra.CHK,
                        extra=Dict("d"=>d, "budget"=>B, "N"=>ra.N, "iters"=>ra.iters))
            K = length(ra.ρ) ÷ ra.N
            γmin = minimum(γk for (_, γk) in ra.CHK[2:end])
            idx = findall(ra.ρ .<= GMAX); Xi = ra.X[:, idx]; ρi = ra.ρ[idx]
            p_i = pdf(pr.prior, Xi); qsum = copy(p_i)
            for k in 2:K
                mk, γk = ra.CHK[k]; qsum .+= proposal_pdf(mk, γk, Xi, pr.prior; nodes=NODES_W)
            end
            wb = p_i ./ (qsum ./ K)
            # emission rule (see header): estimate only where the adaptive schedule reached γ
            curve_a = Dict(γ => (γ >= γmin ? sum(wb[ρi .<= γ]) / length(ra.ρ) : nothing) for γ in G)
            ma = curve_metrics(curve_a, G, pr.truth)
            merge!(ma, discovery(γmin, G))
            adapted = sort([γ for γ in G if γ >= γmin])
            γtight = isempty(adapted) ? γmin : first(adapted)
            ma["fit_ess_all"]   = ess_frac(wb)
            ma["fit_ess_tight"] = ess_frac(wb[ρi .<= γtight])
            ma["tail_ess_gmax"] = tail_ess(ρi, wb, GMAX)
            ma["cov_cond_med"]  = cov_cond_median(ra.CHK[end][1])
            ma["n_evals_actual"] = ra.n_evals
            ma["secs"] = secs_a
            ma["n_runs"] = 1
            row["arm_a"] = ma

            # ---------------- per-threshold, oracle-tuned over N ----------------
            tb = time(); perb = B ÷ length(G); curve_b = Dict{Float64,Any}()
            bN = Dict{String,Any}(); evals_b = 0

            for (gi, γ) in enumerate(G)
                best = nothing; bestre = Inf; bestN = 0; T = pr.truth(γ)
                for (j, N) in enumerate((100, 250, 500))
                    sx = seed * 100_000 + (gi - 1) * 10 + j
                    rb = one_run(pr, sx, perb; target=γ, N=N)
                    record_run!(REC, string("b_perthreshold/g", round(γ, digits=8), "/N", N);
                                X=rb.X, rho=rb.ρ, w=rb.w, logq=rb.logq, iter=rb.iter,
                                defmask=rb.def, checkpoints=rb.CHK,
                                extra=Dict("gamma"=>γ, "gamma_index"=>gi, "budget"=>perb, "N"=>N,
                                           "candidate"=>j, "rng_seed"=>sx))
                    evals_b += rb.n_evals
                    est = sum((rb.ρ .<= γ) .* rb.w) / length(rb.ρ)
                    re = isfinite(est) && T > 0 ? abs(est - T) / T : Inf
                    if re < bestre; bestre = re; best = est; bestN = N; end
                end
                curve_b[γ] = best; bN[string(γ)] = bestN
            end
            mb = curve_metrics(curve_b, G, pr.truth)
            mb["selected_N"] = bN; mb["per_threshold_budget"] = perb
            mb["n_evals_actual"] = evals_b
            mb["n_evals_note"] = "oracle tuning evaluates 3 candidate N per threshold; ALL are counted"
            mb["secs"] = time() - tb; mb["n_runs"] = 3 * length(G)
            row["arm_b"] = mb

            # ---------------- AMS, oracle-tuned over (p0, nmcmc) ----------------
            tc = time(); best_c = nothing; best_err = Inf; best_meta = nothing; evals_c_all = 0
            for p0 in (0.1, 0.2, 0.3), nmcmc in (5, 10)
                nlev = max(3, ceil(Int, log(minimum(pr.pgrid)) / log(p0)))
                m = max(40, B ÷ (nlev * (1 + 2 * nmcmc)))
                Random.seed!(seed * 999 + Int(round(100p0)) * 17 + nmcmc)
                levels, nev = ams_curve(pr.sys; m=m, m_elite=max(3, round(Int, m * p0)),
                                        nmcmc=nmcmc, σ=0.5)
                evals_c_all += nev
                cc = Dict{Float64,Any}(); errs = Float64[]
                for γ in G
                    est, _ = ams_at(levels, γ); cc[γ] = est
                    if est !== nothing; T = pr.truth(γ); T > 0 && push!(errs, abs(est - T) / T); end
                end
                err = isempty(errs) ? Inf : median(errs)
                if err < best_err
                    best_err = err; best_c = cc
                    best_meta = Dict("p0"=>p0, "nmcmc"=>nmcmc, "nevals"=>nev, "m"=>m, "nlevels"=>nlev)
                end
            end
            mc = curve_metrics(best_c === nothing ? Dict{Float64,Any}() : best_c, G, pr.truth)
            mc["selected"] = best_meta
            mc["n_evals_actual"] = best_meta === nothing ? 0 : best_meta["nevals"]
            mc["n_evals_all_configs"] = evals_c_all
            mc["n_evals_note"] = "n_evals_actual = the SELECTED config's evaluations incl. MCMC moves; " *
                                 "n_evals_all_configs = sum over all 6 oracle-tuning configs"
            mc["secs"] = time() - tc; mc["n_runs"] = 6
            row["arm_c"] = mc
            row["ok"] = true
        catch e
            row["error"] = sprint(showerror, e)
            row["backtrace"] = string.(stacktrace(catch_backtrace())[1:min(8,end)])
        end
        # Write the raw-state bundle before the JSON row, so a row on disk always has its
        # bundle beside it. Bundle failures are recorded, never fatal to the run.
        if isdefined(Main, :OUTDIR)
            try
                merge!(row, write_bundle(REC, Main.OUTDIR, "d$(d)_B$(B)_seed$(seed)",
                        Dict("seed"=>seed, "git_sha"=>(isdefined(Main, :GIT_SHA) ? Main.GIT_SHA : "unknown"),
                             "marginal"=>marginal_settings())))
            catch e
                row["raw_state_error"] = sprint(showerror, e)
            end
        end
        return persist_row("d$(d)_B$(B)_seed$(seed)", row)
    end
end

using Printf, JSON, Statistics
include(joinpath(@__DIR__, "provenance.jl")); using .RunProvenance
include(joinpath(@__DIR__, "metrics.jl")); using .FinalMetrics

config = Dict{String,Any}(
    "experiment"=>"exp2_shell",
    "measured_quantity"=>"curve accuracy AND (separately) discovery behaviour on an isotropic failure set",
    "problem"=>"radial_shell", "problem_def"=>"rho = r_crit - ||x||, x ~ N(0,I_d)",
    "truth"=>"P(rho<=g) = ccdf(Chisq(d), (r_crit-g)^2) (closed form)",
    "cells"=>[Dict("d"=>c.d, "budget"=>c.B) for c in CELLS],
    "seeds"=>SEEDS, "n_seeds"=>NSEEDS,
    "grid_construction"=>"PGRID = logspace(1e-3, 1e-6, 16); TVEC = sqrt(quantile(Chisq(d),1-p)); r_crit = max(TVEC)+0.5; grid = r_crit - TVEC",
    "n_grid_points"=>16, "quadrature_nodes"=>NODES,
    "proposal_family"=>"JointModel(Normal(), ConditionalEMGMM(k=2))", "n_mixture_components"=>2,
    "batch_size_amortized"=>1000, "adaptive_quantile"=>0.2, "target_threshold"=>0.0,
    "max_weight_fit"=>10.0, "sigma_smoothing"=>0.0, "defensive_eta"=>0.0,
    "arms"=>["a_amortized","b_perthreshold_oracle","c_ams_oracle"],
    "arm_b_protocol"=>"K runs at B/K; oracle-selects best N in {100,250,500} by ground truth (all 3 counted in budget)",
    "arm_c_protocol"=>"ams_curve oracle-selected over p0 in {0.1,0.2,0.3} x nmcmc in {5,10} by median relative error; evaluations include MCMC moves",
    "ile_formula"=>FinalMetrics.ILE_FORMULA,
    "discovery_policy"=>"discovery recorded separately from accuracy; every ILE carries n_grid_used and an explicit mask",
)
dir, meta = init_group("exp2_shell", config)
tasks = [(c.d, c.B, s) for c in CELLS for s in SEEDS]
@printf("exp2_shell → %s\n  %d cells × %d seeds = %d runs, nodes=%d, %d workers\n",
        dir, length(CELLS), NSEEDS, length(tasks), NODES, nworkers())

@everywhere const OUTDIR = $dir
rows = pmap(t -> Main.run_cell_seed(t...), tasks)

summary = Dict{String,Any}("experiment"=>"exp2_shell", "config"=>config,
    "ile_formula"=>FinalMetrics.ILE_FORMULA, "cells"=>Dict{String,Any}())
for c in CELLS
    key = "d$(c.d)_B$(c.B)"
    cr = [r for r in rows if r["d"] == c.d && r["budget"] == c.B]
    ok = [r for r in cr if r["ok"] === true]
    cell = Dict{String,Any}("d"=>c.d, "budget"=>c.B,
        "seed_accounting"=>Dict("n_requested"=>NSEEDS, "n_completed"=>length(ok),
            "n_failed"=>length(cr)-length(ok), "seeds_requested"=>SEEDS,
            "failure_records"=>[Dict("seed"=>r["seed"],"error"=>get(r,"error",nothing))
                                for r in cr if r["ok"] !== true]))
    for (arm, label) in (("arm_a","a_amortized"), ("arm_b","b_perthreshold_oracle"), ("arm_c","c_ams_oracle"))
        ms = [r[arm] for r in ok]
        iles = [m["ile"] for m in ms if m["ile"] !== nothing]
        a = Dict{String,Any}(
            "n_seeds"=>length(ms), "n_contributing_ile"=>length(iles),
            "ile_median_conditional_on_contribution"=>isempty(iles) ? nothing : median(iles),
            "ile_per_seed"=>[m["ile"] for m in ms],
            "n_grid_total"=>isempty(ms) ? nothing : ms[1]["n_grid_total"],
            "n_grid_used_per_seed"=>[m["n_grid_used"] for m in ms],
            "n_cells_nonnull_per_seed"=>[m["n_cells_nonnull"] for m in ms],
            "n_evals_actual_median"=>isempty(ms) ? nothing : median([m["n_evals_actual"] for m in ms]),
            "n_evals_actual_per_seed"=>[m["n_evals_actual"] for m in ms],
            "secs_median"=>isempty(ms) ? nothing : median([m["secs"] for m in ms]))
        if arm == "arm_a"
            a["discovery_count"] = count(m -> m["discovered"], ms)
            a["reached_target_count"] = count(m -> m["reached_target"], ms)
            a["gamma_min_per_seed"] = [m["gamma_min"] for m in ms]
            a["gamma_min_median"] = isempty(ms) ? nothing : median([m["gamma_min"] for m in ms])
            a["grid_gmax"] = isempty(ms) ? nothing : ms[1]["grid_gmax"]
            for f in ("fit_ess_all","fit_ess_tight","tail_ess_gmax","cov_cond_med")
                a[f * "_median"] = isempty(ms) ? nothing :
                    (v = [m[f] for m in ms if m[f] isa Real && isfinite(m[f])]; isempty(v) ? nothing : median(v))
            end
        end
        cell[label] = a
    end
    summary["cells"][key] = cell
end
write_summary(dir, summary)

@printf("\n%-12s %-22s %-4s %-6s %-10s %-22s %s\n","cell","arm","n","disc","ILE(med)","n_used/seed(first 4)","evals(med)")
for c in CELLS
    k = "d$(c.d)_B$(c.B)"; cell = summary["cells"][k]
    for label in ("a_amortized","b_perthreshold_oracle","c_ams_oracle")
        a = cell[label]
        @printf("%-12s %-22s %-4d %-6s %-10s %-22s %s\n", k, label, a["n_contributing_ile"],
            label == "a_amortized" ? "$(a["discovery_count"])/$(a["n_seeds"])" : "—",
            a["ile_median_conditional_on_contribution"] === nothing ? "—" :
              string(round(a["ile_median_conditional_on_contribution"], sigdigits=6)),
            string(a["n_grid_used_per_seed"][1:min(4,end)]),
            a["n_evals_actual_median"] === nothing ? "—" : string(round(Int, a["n_evals_actual_median"])))
    end
end
println("EXP2_DONE")
