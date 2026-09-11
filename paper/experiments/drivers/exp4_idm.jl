# Experiment 4: 150-dimensional IDM crosswalk benchmark.
#
# The primary method uses a low-rank conditional proposal with a defensive
# mixture and a total simulator budget of 25,000 evaluations per seed.
# Optional baselines use the same nominal budget.
#
# Usage:
#   julia --project=paper/envs/idm paper/experiments/drivers/exp4_idm.jl [nseeds] [arms] [nodes]
#
# Reruns default to `paper/rerun/exp4_idm/`.

using Distributed
const NSEEDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
const ARMS   = length(ARGS) >= 2 ? ARGS[2] : "a"
const NODES  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1000
const SEEDS  = collect(1:NSEEDS)
const BUDGET = 25_000
const NSAMP  = 1000
const NITER  = 25
const RANK   = 10
const ETA    = 0.1
const ANCHOR_QUANTILES = [0.02, 0.05, 0.10, 0.20, 0.30]   # PREDEFINED, fixed before any fitting
addprocs(max(0, min(8, Sys.CPU_THREADS - 2) - nworkers() + 1); exeflags="--project=paper/envs/idm")

@everywhere include(joinpath(@__DIR__, "metrics.jl"))
@everywhere using .FinalMetrics
@everywhere include(joinpath(@__DIR__, "rawstate.jl"))
@everywhere using .RawState

@everywhere begin
    using ARIS
    using Distributions, Random, Statistics, LinearAlgebra, JSON
    LinearAlgebra.BLAS.set_num_threads(1)
    include(joinpath(@__DIR__, "..", "..", "..", "examples", "problems", "crosswalk.jl"))
    include(joinpath(@__DIR__, "..", "conditional_sgm.jl"))
    include(joinpath(@__DIR__, "..", "eta_mixture.jl"))
    include(joinpath(@__DIR__, "..", "balance.jl"))
    include(joinpath(@__DIR__, "..", "ams_curve.jl"))
    include(joinpath(@__DIR__, "..", "evt_tail.jl"))

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

    const NODES_W = $NODES; const BUDGET_W = $BUDGET; const NSAMP_W = $NSAMP
    const NITER_W = $NITER; const RANK_W = $RANK; const ETA_W = $ETA
    const ANCH_W  = $ANCHOR_QUANTILES

    const SYS   = AdversarialCrosswalk(; dt=0.05)
    const D     = System.get_xdim(SYS)
    const PRIOR = System.prior(SYS)
    const GGRID = Float64[0.434, 0.38, 0.32, 0.26, 0.20, 0.14, 0.08, 0.0]
    const GMAX  = maximum(GGRID)
    sep_of(γ) = γ < 0.99 ? sqrt(max(20 / (0.99 - γ) - 20, 0.0)) : 0.0

    const H   = JSON.parsefile(joinpath(@__DIR__, "..", "..", "results", "reference", "idm_hist.json"))
    const HC  = Int.(H["counts"]); const HN = H["ndone"]
    const HBLO = H["blo"]; const HBW = H["bw"]
    _bin(γ) = clamp(Int(floor((γ - HBLO) / HBW)) + 1, 1, length(HC))
    # Each grid threshold is a histogram-bin lower edge, so exclude the containing
    # bin when computing P(ρ ≤ γ).
    truth_ref(γ) = sum(HC[1:_bin(γ)-1]) / HN
    function wilson(γ)
        k = sum(HC[1:_bin(γ)-1]); p = k / HN; z = 1.96; den = 1 + z^2 / HN
        c = (p + z^2 / (2HN)) / den
        h = z * sqrt(p * (1 - p) / HN + z^2 / (4HN^2)) / den
        return (k=k, lo=max(0, c - h), hi=c + h)
    end
    # SigmaSmoothed-style delegation: EtaMixture/SGM sample from the base, so the re-indexer must
    # evaluate the base marginal. Without these methods the balance weights would be wrong.
    proposal_pdf(m::EtaMixture, γ, X, prior; kw...) =
        (1 - m.η) .* proposal_pdf(m.base, γ, X, prior; kw...) .+ m.η .* pdf(prior, X)
    # Extend the package's per-r conditional evaluator to ConditionalSGM (marginal_pdf calls
    # conditional_logpdf_at directly, so a method on a local alias would not be seen).
    ARIS.conditional_logpdf_at(
        traj::ConditionalSGM, r::Real, X::AbstractMatrix) =
        Distributions.logpdf(_cond(traj, Float64(r)), X)

    "Primary arm: one adaptive run."
    function primary_run(seed)
        Random.seed!(seed); reset_fit_stats!()
        CHK = Any[]; t0 = time()
        cb = (cv, DD) -> push!(CHK, (deepcopy(cv.model), Float64(cv.sampling_strategy.value)))
        base  = JointModel(Normal(), ConditionalSGM(2, D + 1, D + 1; rank=RANK_W))
        model = EtaMixture(base, PRIOR, ETA_W)
        cv = ConditionalValidation(model=model, xdim=D, depth=System.get_depth(SYS),
            sdim=System.get_sdim(SYS), n_samples=NSAMP_W, n_iter=NITER_W,
            sampling_strategy=QuantileSampling(0.2; target=0.0), max_weight=Inf, log_callback=cb)
        train!(cv, SYS)
        ρ = Float64.(cv.buffer[:ρ])
        return (CHK=CHK, X=Float64.(cv.buffer[:x]), ρ=ρ, w=Float64.(cv.buffer[:w]),
                logq=Float64.(cv.buffer[:logpdf]), def=Vector{Bool}(cv.buffer[:defensive]),
                iter=repeat(1:NITER_W, inner=NSAMP_W)[1:length(ρ)],
                N=NSAMP_W, n_evals=length(ρ), secs_sim_fit=time() - t0)
    end

    function run_seed(seed::Int, arms::String)
        row = Dict{String,Any}("seed"=>seed, "ok"=>false, "d"=>D, "budget"=>BUDGET_W,
                               "gamma_grid"=>GGRID, "arms"=>arms)
        REC = RunRecorder()      # raw state for offline re-evaluation of the estimators
        try
            # ---------------- primary (a) ----------------
            r = primary_run(seed)
            record_run!(REC, "a_primary_sgm"; X=r.X, rho=r.ρ, w=r.w, logq=r.logq, iter=r.iter,
                        defmask=r.def, checkpoints=r.CHK,
                        extra=Dict("d"=>D, "budget"=>BUDGET_W, "N"=>r.N, "iters"=>NITER_W,
                                   "rank"=>RANK_W, "eta"=>ETA_W))
            K = length(r.ρ) ÷ r.N
            γmin = minimum(γk for (_, γk) in r.CHK[2:end])
            t_ri = time()
            idx = findall(r.ρ .<= GMAX); Xi = r.X[:, idx]; ρi = r.ρ[idx]
            p_i = pdf(PRIOR, Xi); qsum = copy(p_i)
            for k in 2:K
                mk, γk = r.CHK[k]; qsum .+= proposal_pdf(mk, γk, Xi, PRIOR; nodes=NODES_W)
            end
            wb = p_i ./ (qsum ./ K); secs_reindex = time() - t_ri
            curve = Dict(γ => sum(wb[ρi .<= γ]) / length(r.ρ) for γ in GGRID)
            m = curve_metrics(curve, GGRID, truth_ref)
            merge!(m, discovery(γmin, GGRID))
            for pt in m["per_threshold"]
                pt["sep_m"] = sep_of(pt["gamma"])
                wl = wilson(pt["gamma"])
                pt["reference_wilson_lo"] = wl.lo; pt["reference_wilson_hi"] = wl.hi
                pt["reference_count"] = wl.k
            end
            m["sep_at_gamma_min"]        = sep_of(γmin)
            m["reference_P_at_gamma_min"] = truth_ref(γmin)
            m["fit_ess_reindex"] = ess_frac(wb)
            m["fit_ess_raw"]     = ess_frac(r.w)
            m["tail_ess_gmax"]   = tail_ess(ρi, wb, GMAX)
            m["tail_ess_at_gamma_min"] = tail_ess(ρi, wb, γmin)
            m["cov_cond_med"]    = cov_cond_median(r.CHK[end][1])
            m["n_evals_actual"]  = r.n_evals
            m["secs_sim_and_fit"] = r.secs_sim_fit
            m["secs_reindex"]     = secs_reindex
            m["secs_total"]       = r.secs_sim_fit + secs_reindex
            m["n_runs"] = 1
            fs = get_fit_stats()
            m["numerical"] = Dict("small_det"=>fs[:small_det], "small_mix"=>fs[:small_mix],
                                  "kmeans_fallback"=>fs[:kmeans_fallback],
                                  "fit_method"=>fs[:kmeans_fallback] > 0 ? "fallback_assisted" : "em")
            row["arm_a_primary"] = m

            # -------- GPD tail extrapolation (kept SEPARATE from the pooled balance-MIS curve) --------
            gpd = Dict{String,Any}("anchor_quantiles"=>ANCH_W, "target_gamma"=>0.0,
                "note"=>"unconstrained and xi>=0 fitted to the SAME buffer at PREDEFINED anchors; no anchor selected using ground truth",
                "reference_at_target"=>truth_ref(0.0), "anchors"=>Any[])
            for q in ANCH_W
                u = quantile(r.ρ, q); P_u = sum(r.w[r.ρ .<= u]) / length(r.ρ)
                un = evt_estimate(r.ρ, r.w, u, P_u, 0.0)
                co = evt_estimate(r.ρ, r.w, u, P_u, 0.0; xi_nonneg=true)
                push!(gpd["anchors"], Dict("quantile"=>q, "u"=>u, "sep_u_m"=>sep_of(u), "P_u"=>P_u,
                    "n_exceedances_raw"=>count(r.ρ .<= u),
                    "unconstrained"=>Dict("estimate"=>un.est, "xi"=>un.ξ, "sigma"=>un.σ, "n_eff"=>un.n_eff),
                    "xi_nonneg"=>Dict("estimate"=>co.est, "xi"=>co.ξ, "sigma"=>co.σ, "n_eff"=>co.n_eff)))
            end
            for k in ("unconstrained", "xi_nonneg")
                v = [a[k]["estimate"] for a in gpd["anchors"] if a[k]["estimate"] isa Real && isfinite(a[k]["estimate"]) && a[k]["estimate"] > 0]
                gpd["$(k)_anchor_spread_ratio"] = isempty(v) ? nothing : maximum(v) / minimum(v)
                gpd["$(k)_estimate_range"] = isempty(v) ? nothing : [minimum(v), maximum(v)]
            end
            row["gpd_tail"] = gpd

            # ---------------- baselines at the SAME budget ----------------
            if occursin("b", arms)
                tb = time(); perb = BUDGET_W ÷ length(GGRID)
                cb_curve = Dict{Float64,Any}(); evals_b = 0; sel = Dict{String,Any}()
                # Each threshold–candidate pair has its own deterministic RNG stream
                # seed*100_000 + (gi-1)*10 + j, where gi indexes GGRID and j the candidate N.
                # (gi-1)*10+j is injective over gi<=8, j<=2 and < 100_000, so streams never
                # collide across thresholds, candidates or seeds. All evaluations are charged.
                for (gi, γ) in enumerate(GGRID)
                    best = nothing; bestre = Inf; bestN = 0; T = truth_ref(γ)
                    for (j, N) in enumerate((500, 1000))
                        # per-(threshold, candidate) RNG stream
                        Random.seed!(seed * 100_000 + (gi - 1) * 10 + j); reset_fit_stats!()
                        it = max(2, perb ÷ N)
                        mdl = JointModel(Normal(), ConditionalEMGMM(2, D + 1, D + 1))
                        cvb = ConditionalValidation(model=mdl, xdim=D, depth=System.get_depth(SYS),
                            sdim=System.get_sdim(SYS), n_samples=N, n_iter=it,
                            sampling_strategy=QuantileSampling(0.2; target=γ), max_weight=10.0)
                        train!(cvb, SYS)
                        ρb = Float64.(cvb.buffer[:ρ]); wbb = Float64.(cvb.buffer[:w])
                        evals_b += length(ρb)
                        est = sum((ρb .<= γ) .* wbb) / length(ρb)
                        re = isfinite(est) && T > 0 ? abs(est - T) / T : Inf
                        if re < bestre; bestre = re; best = est; bestN = N; end
                    end
                    cb_curve[γ] = best; sel[string(γ)] = bestN
                end
                mb = curve_metrics(cb_curve, GGRID, truth_ref)
                mb["selected_N"] = sel; mb["per_threshold_budget"] = perb
                mb["n_evals_actual"] = evals_b
                mb["n_evals_note"] = "oracle tuning evaluates 2 candidate N per threshold; ALL counted"
                mb["secs_total"] = time() - tb; mb["n_runs"] = 2 * length(GGRID)
                row["arm_b_perthreshold"] = mb
            end
            if occursin("c", arms)
                # Oracle-tuned AMS over six configurations using the dimension-scaled
                # random-walk-Metropolis variant. Exact adaptive levels are used when
                # available; otherwise estimates use tagged log-linear interpolation.
                # Charged evaluation counts include MCMC moves.
                
                tc = time(); best_c = nothing; best_err = Inf; best_meta = nothing; ev_all = 0
                cand_rec = Any[]
                for p0 in (0.1, 0.2, 0.3), nmcmc in (5, 10)
                    nlev = 8   # capped level count
                    m_ = max(40, BUDGET_W ÷ (nlev * (1 + 2 * nmcmc)))
                    sx = seed * 999 + Int(round(100p0)) * 17 + nmcmc
                    Random.seed!(sx)
                    levels, nev, diag = ams_curve_rwm(SYS; m=m_, m_elite=max(3, round(Int, m_ * p0)),
                                                   nmcmc=nmcmc, k_max=nlev)
                    ev_all += nev
                    cc = Dict{Float64,Any}(); prov = Dict{String,Any}(); errs = Float64[]
                    for γ in GGRID
                        est, pv = ams_at_tagged(levels, γ)
                        cc[γ] = est; prov[string(γ)] = string(pv)
                        if est !== nothing; T = truth_ref(γ); T > 0 && push!(errs, abs(est - T) / T); end
                    end
                    nnf = count(x -> !isfinite(x), errs)
                    valid = !isempty(errs) && nnf == 0
                    e = valid ? median(errs) : nothing
                    push!(cand_rec, Dict("p0"=>p0, "nmcmc"=>nmcmc, "m"=>m_, "k_max"=>nlev,
                        "rng_seed"=>sx, "nevals"=>nev, "valid"=>valid, "oracle_error"=>e,
                        "n_nonfinite_errors"=>nnf,
                        "n_positive_estimates"=>count(g -> cc[g] isa Real && isfinite(cc[g]) && cc[g] > 0, GGRID),
                        "estimates"=>Dict(string(g)=>cc[g] for g in GGRID), "provenance"=>prov,
                        "diag"=>diag, "levels"=>[[l[1], l[2]] for l in levels]))
                    if valid && e < best_err
                        best_err = e; best_c = cc
                        best_meta = Dict("p0"=>p0, "nmcmc"=>nmcmc, "nevals"=>nev, "m"=>m_,
                                         "k_max"=>nlev, "rng_seed"=>sx, "oracle_error"=>e,
                                         "sigma_used"=>diag["sigma_used"],
                                         "acceptance_rate"=>diag["acceptance_rate"])
                    end
                end
                mc = curve_metrics(best_c === nothing ? Dict{Float64,Any}() : best_c, GGRID, truth_ref)
                mc["selected"] = best_meta
                mc["n_evals_actual"] = best_meta === nothing ? 0 : best_meta["nevals"]
                mc["n_evals_all_configs"] = ev_all
                mc["n_evals_note"] = "evaluations include MCMC moves"
                mc["n_valid_candidates"] = count(c -> c["valid"], cand_rec)
                mc["candidates"] = cand_rec
                mc["secs_total"] = time() - tc; mc["n_runs"] = 6
                row["arm_c_ams"] = mc
            end
            row["ok"] = true
        catch e
            row["error"] = sprint(showerror, e)
            row["backtrace"] = string.(stacktrace(catch_backtrace())[1:min(10,end)])
        end
        # Write the raw-state bundle before the JSON row, so a row on disk always has its
        # bundle beside it. Bundle failures are recorded, never fatal to the run.
        if isdefined(Main, :OUTDIR)
            try
                merge!(row, write_bundle(REC, Main.OUTDIR, "seed$(seed)",
                        Dict("seed"=>seed, "git_sha"=>(isdefined(Main, :GIT_SHA) ? Main.GIT_SHA : "unknown"),
                             "marginal"=>marginal_settings())))
            catch e
                row["raw_state_error"] = sprint(showerror, e)
            end
        end
        return persist_row("seed$(seed)", row)
    end
end

using Printf, JSON, Statistics
include(joinpath(@__DIR__, "provenance.jl")); using .RunProvenance
include(joinpath(@__DIR__, "metrics.jl")); using .FinalMetrics

# ---- reference configuration verification ----
_H = JSON.parsefile(joinpath(@__DIR__, "..", "..", "results", "reference", "idm_hist.json"))
ref_check = Dict{String,Any}(
    "hist_json_fields" => Dict(k => v for (k, v) in _H if k != "counts"),
    "run_ascale" => 0.60, "reference_ascale" => _H["ascale"],
    "run_d" => 150, "reference_d" => _H["d"],
    "run_nsub" => 4, "reference_nsub" => _H["nsub"],
    "run_dt_phys" => 0.05,
    "reference_dt_phys" => 0.05,
    "reference_dt_phys_source" => "NOT stored in hist.json; recovered from paper/experiments/idm_reference.jl " *
                                  "AdversarialCrosswalk(; dt=DT_DIST/NSUB) with DT_DIST=0.2, NSUB=4 => 0.05",
    "robustness_definition" => "rho = 0.99 - running-max PoCA reward; shared crosswalk_rollout used by " *
                              "both the reference generator and this run",
    "reference_n" => _H["ndone"],
)
ref_check["ascale_match"] = ref_check["run_ascale"] == ref_check["reference_ascale"]
ref_check["d_match"]      = ref_check["run_d"] == ref_check["reference_d"]
ref_check["nsub_match"]   = ref_check["run_nsub"] == ref_check["reference_nsub"]
ref_check["dt_phys_match"]= ref_check["run_dt_phys"] == ref_check["reference_dt_phys"]
ref_check["all_match"]    = all(ref_check[k] for k in ("ascale_match","d_match","nsub_match","dt_phys_match"))
ref_check["all_match"] || error("REFERENCE CONFIG MISMATCH — refusing to compare:\n" * string(ref_check))

config = Dict{String,Any}(
    "experiment"=>"exp4_idm",
    "measured_quantity"=>"failure-probability curve at real trajectory dimension against a 26M-rollout reference",
    "problem"=>"idm_crosswalk", "dimension"=>150,
    "problem_def"=>"AV crosswalk, 30 disturbance steps x 5 channels, dt_phys=0.05, ascale=0.60; rho = 0.99 - PoCA",
    "budget_total"=>BUDGET, "batch_size"=>NSAMP, "n_iter"=>NITER, "seeds"=>SEEDS, "n_seeds"=>NSEEDS,
    "threshold_grid"=>[0.434,0.38,0.32,0.26,0.20,0.14,0.08,0.0],
    "separation_mapping"=>"sep(gamma) = sqrt(20/(0.99-gamma) - 20)",
    "proposal_family"=>"EtaMixture(JointModel(Normal(), ConditionalSGM(k=2, rank=10)), prior, eta)",
    "conditional"=>"low-rank-plus-diagonal (FactorEM)", "rank"=>RANK, "n_mixture_components"=>2,
    "regularization"=>"MAP D-prior covariance floor (D .= max.(D, 1e-4)) in the FactorEM inner M-step, vendored SGM",
    "defensive_eta"=>ETA, "defensive_stratified"=>true,
    "fit_ess_guard"=>"EtaMixture floors fit-ESS at eta*N",
    "quadrature_nodes"=>NODES, "adaptive_quantile"=>0.2, "target_threshold"=>0.0,
    "max_weight_fit"=>"Inf (eta-mixture bounds weights at 1/eta)", "sigma_smoothing"=>0.0,
    "arms_requested"=>ARMS,
    "budget_accounting"=>"nominal budget B=$(BUDGET) per arm; the charged evaluations of each arm are in metrics.json",
    "gpd_anchor_quantiles"=>ANCHOR_QUANTILES,
    "gpd_policy"=>"unconstrained and xi>=0 on the SAME buffer at predefined anchors; anchor sensitivity reported; no ground-truth anchor selection",
    "reference_check"=>ref_check,
    "ile_formula"=>FinalMetrics.ILE_FORMULA,
)
dir, meta = init_group("exp4_idm", config)
@printf("exp4_idm → %s\n  %d seeds, arms=%s, B=%d, nodes=%d, rank=%d, eta=%.2f, %d workers\n",
        dir, NSEEDS, ARMS, BUDGET, NODES, RANK, ETA, nworkers())
@printf("  reference check: ascale=%s d=%s nsub=%s dt_phys=%s → ALL MATCH\n",
        ref_check["ascale_match"], ref_check["d_match"], ref_check["nsub_match"], ref_check["dt_phys_match"])

# resumable: skip seeds whose raw record already exists
todo = [s for s in SEEDS if !isfile(joinpath(dir, "raw", "seed$(s).json"))]
@printf("  %d/%d seeds to run (%d already banked)\n", length(todo), NSEEDS, NSEEDS - length(todo))
@everywhere const OUTDIR = $dir
rows_new = isempty(todo) ? Any[] : pmap(s -> Main.run_seed(s, ARMS), todo)
rows = [JSON.parsefile(joinpath(dir, "raw", "seed$(s).json")) for s in SEEDS
        if isfile(joinpath(dir, "raw", "seed$(s).json"))]

ok = [r for r in rows if r["ok"] === true]
summary = Dict{String,Any}("experiment"=>"exp4_idm", "config"=>config,
    "ile_formula"=>FinalMetrics.ILE_FORMULA,
    "seed_accounting"=>Dict("n_requested"=>NSEEDS, "n_completed"=>length(ok),
        "n_failed"=>length(rows)-length(ok), "seeds_requested"=>SEEDS,
        "failure_records"=>[Dict("seed"=>r["seed"],"error"=>get(r,"error",nothing))
                            for r in rows if r["ok"] !== true]),
    "arms"=>Dict{String,Any}())
for (akey, label) in (("arm_a_primary","a_primary_sgm"), ("arm_b_perthreshold","b_perthreshold"),
                      ("arm_c_ams","c_ams"))
    ms = [r[akey] for r in ok if haskey(r, akey)]
    isempty(ms) && continue
    iles = [m["ile"] for m in ms if m["ile"] !== nothing]
    a = Dict{String,Any}("n_seeds"=>length(ms), "n_contributing_ile"=>length(iles),
        "ile_median"=>isempty(iles) ? nothing : median(iles),
        "ile_per_seed"=>[m["ile"] for m in ms],
        "n_grid_total"=>ms[1]["n_grid_total"],
        "n_grid_used_per_seed"=>[m["n_grid_used"] for m in ms],
        "n_cells_nonnull_per_seed"=>[m["n_cells_nonnull"] for m in ms],
        "n_evals_actual_median"=>median([m["n_evals_actual"] for m in ms]),
        "secs_total_median"=>median([m["secs_total"] for m in ms]))
    if akey == "arm_a_primary"
        a["discovery_count"] = count(m -> m["discovered"], ms)
        a["reached_target_count"] = count(m -> m["reached_target"], ms)
        a["gamma_min_per_seed"] = [m["gamma_min"] for m in ms]
        a["gamma_min_median"] = median([m["gamma_min"] for m in ms])
        a["sep_at_gamma_min_median"] = median([m["sep_at_gamma_min"] for m in ms])
        for f in ("fit_ess_reindex","fit_ess_raw","tail_ess_gmax","cov_cond_med",
                  "secs_sim_and_fit","secs_reindex")
            v = [m[f] for m in ms if m[f] isa Real && isfinite(m[f])]
            a[f * "_median"] = isempty(v) ? nothing : median(v)
        end
        a["numerical_any_fallback_assisted"] = count(m -> m["numerical"]["fit_method"] == "fallback_assisted", ms)
    end
    # per-threshold aggregate across seeds
    a["per_threshold"] = [begin
        vals = [pt["estimate"] for m in ms for pt in m["per_threshold"] if pt["gamma"] == γ && pt["estimate"] !== nothing]
        rel  = [pt["abs_relerr"] for m in ms for pt in m["per_threshold"] if pt["gamma"] == γ && pt["abs_relerr"] !== nothing]
        Dict("gamma"=>γ, "sep_m"=>(γ < 0.99 ? sqrt(max(20/(0.99-γ)-20,0.0)) : 0.0),
             "n_nonnull"=>length(vals),
             "estimate_median"=>isempty(vals) ? nothing : median(vals),
             "abs_relerr_median"=>isempty(rel) ? nothing : median(rel))
    end for γ in [0.434,0.38,0.32,0.26,0.20,0.14,0.08,0.0]]
    summary["arms"][label] = a
end
gs = [r["gpd_tail"] for r in ok if haskey(r, "gpd_tail")]
if !isempty(gs)
    summary["gpd_tail"] = Dict{String,Any}("n_seeds"=>length(gs),
        "anchor_quantiles"=>ANCHOR_QUANTILES, "reference_at_target"=>gs[1]["reference_at_target"],
        "per_anchor"=>[Dict("quantile"=>q,
            "unconstrained_estimate_median"=>(v=[a["unconstrained"]["estimate"] for g in gs for a in g["anchors"] if a["quantile"]==q && a["unconstrained"]["estimate"] isa Real]; isempty(v) ? nothing : median(v)),
            "xi_nonneg_estimate_median"=>(v=[a["xi_nonneg"]["estimate"] for g in gs for a in g["anchors"] if a["quantile"]==q && a["xi_nonneg"]["estimate"] isa Real]; isempty(v) ? nothing : median(v)),
            "unconstrained_xi_median"=>(v=[a["unconstrained"]["xi"] for g in gs for a in g["anchors"] if a["quantile"]==q]; isempty(v) ? nothing : median(v)),
            "n_eff_median"=>(v=[a["unconstrained"]["n_eff"] for g in gs for a in g["anchors"] if a["quantile"]==q]; isempty(v) ? nothing : median(v)))
            for q in ANCHOR_QUANTILES],
        "unconstrained_anchor_spread_per_seed"=>[g["unconstrained_anchor_spread_ratio"] for g in gs],
        "xi_nonneg_anchor_spread_per_seed"=>[g["xi_nonneg_anchor_spread_ratio"] for g in gs])
end
write_summary(dir, summary)

@printf("\ncompleted %d/%d seeds\n", length(ok), NSEEDS)
for (label, a) in summary["arms"]
    @printf("%-18s n=%d  ILE(med)=%s  n_used/seed=%s  evals(med)=%s\n", label, a["n_seeds"],
        a["ile_median"] === nothing ? "—" : string(round(a["ile_median"], sigdigits=6)),
        string(a["n_grid_used_per_seed"]), string(round(Int, a["n_evals_actual_median"])))
end
if haskey(summary, "arms") && haskey(summary["arms"], "a_primary_sgm")
    a = summary["arms"]["a_primary_sgm"]
    @printf("primary: discovery %d/%d  γmin(med)=%.4f (sep %.2f m)  fit-ESS(reindex,med)=%s\n",
        a["discovery_count"], a["n_seeds"], a["gamma_min_median"], a["sep_at_gamma_min_median"],
        string(a["fit_ess_reindex_median"]))
end
println("EXP4_DONE")
