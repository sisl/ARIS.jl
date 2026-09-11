# Experiment 1 unconditional CEM baseline.
#
# Uses an unconditional two-component EMGMM cross-entropy proposal and
# evaluates the resulting proposal sequence with pooled balance MIS on the
# Exp1 threshold grids. Metrics are computed offline from the stored raw state
# by `cem_metrics.jl`.
#
# Usage:
#   julia --project=. paper/experiments/cem_baseline/run_cem_arm.jl \
#       [--seeds 1,2,...] [--out DIR]

using ARIS
using Distributions, Random, Statistics, LinearAlgebra, Printf, JSON, Serialization
LinearAlgebra.BLAS.set_num_threads(1)

const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(REPO, "examples", "problems", "toy.jl"))
include(joinpath(REPO, "paper", "experiments", "balance.jl"))
include(joinpath(REPO, "paper", "experiments", "drivers", "rawstate.jl"))
using .RawState
include(joinpath(@__DIR__, "uncond_emgmm.jl"))

function argval(flag, default=nothing)
    i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1]
end

const COND_DIR = get(ENV, "EXP1_RESULTS_DIR", joinpath(REPO, "paper", "results", "exp1"))
const OUTDIR   = argval("--out", joinpath(REPO, "paper", "rerun", "exp1_cem"))
const SEEDS    = parse.(Int, split(argval("--seeds", "1,2,3,4,5,6,7,8,9,10"), ","))

# ---- arm constants ----------------------------------------------------------
const ARM     = "unconditional_cem"
const COMP    = 2          # mixture components
const NSAMP   = 1000       # evaluations per iteration
const NITER   = 25         # iterations
const ALPHA   = 0.2        # nominal elite fraction
const NELITE  = round(Int, ALPHA * NSAMP)   # = 200, same elite rule as Exp3's CEM arm
const MAXW    = 10.0       # fit-weight cap
const BUDGET  = NSAMP * NITER               # = 25_000
const KS      = [10, 25, 50, 100]

const SYS   = UnimodalToy(3.0)
const PRIOR = System.prior(SYS)
truthf(γ)   = cdf(Normal(), -(3.0 - γ))^2
const SNAP_TOL = 1e-10
snap(γ) = abs(γ) < SNAP_TOL ? 0.0 : γ

# ---- evaluation grids: LOADED from the stored conditioned artifacts ---------
function load_stored_grids()
    grids = Dict{Int,Vector{Float64}}()
    ref = JSON.parsefile(joinpath(COND_DIR, "raw", "seed1.json"))
    for K in KS
        grids[K] = Float64.(ref["amortized"][string(K)]["gamma_grid"])
    end
    # assert identical across all 10 conditioned seeds (paired comparison requirement)
    for s in 1:10
        r = JSON.parsefile(joinpath(COND_DIR, "raw", "seed$(s).json"))
        for K in KS
            @assert Float64.(r["amortized"][string(K)]["gamma_grid"]) == grids[K] "grid mismatch seed $s K=$K"
        end
    end
    return grids
end
const GRIDS = load_stored_grids()
println("grids loaded from stored conditioned artifacts and verified identical across 10 seeds")

# ---- one CEM run ------------------------------------------------------------
function one_run(seed)
    Random.seed!(seed); reset_fit_stats!()
    CHK = Any[]; QDIAG = Float64[]; t0 = time()
    # Checkpoint capture. CHK[k] (k >= 2) is the model that generated batch k; CHK[1] is
    # recorded after the prior-batch fit and generates no batch (batch 1 is drawn from the
    # prior, which the evaluator uses directly). The second element is a
    # DIAGNOSTIC only -- CrossEntropyMethod has no conditioning schedule, so there is
    # no gamma_k. We record the batch's own 0.2-quantile so the effective-alpha
    # trajectory is inspectable. The evaluator does NOT gate on it.
    cb = (cv, D) -> begin
        push!(CHK, (deepcopy(cv.model), NaN))
        ρb = Float64.(D[:ρ])
        push!(QDIAG, isempty(ρb) ? NaN : quantile(ρb, ALPHA))
    end
    cem = CrossEntropyMethod(model=EMGMM(COMP, 2), xdim=2, depth=1, sdim=2,
                             n_iter=NITER, n_samples=NSAMP,
                             n_elite=NELITE, max_weight=MAXW, log_callback=cb)
    train!(cem, SYS)
    ρ = Float64.(cem.buffer[:ρ])
    return (CHK=CHK, QDIAG=QDIAG,
            X=Float64.(cem.buffer[:x]), ρ=ρ, w=Float64.(cem.buffer[:w]),
            logq=Float64.(cem.buffer[:logpdf]),
            def=Vector{Bool}(cem.buffer[:defensive]),
            iter=repeat(1:NITER, inner=NSAMP)[1:length(ρ)],
            n_evals=length(ρ), secs=time()-t0,
            fitstats=copy(get_fit_stats()))
end

# ---- per-seed driver --------------------------------------------------------
function run_seed(seed)
    key = "seed$(seed)"
    out = Dict{String,Any}("seed"=>seed, "ok"=>false)
    REC = RunRecorder()
    t_all = time()
    try
        r = one_run(seed)
        record_run!(REC, ARM; X=r.X, rho=r.ρ, w=r.w, logq=r.logq, iter=r.iter,
                    defmask=r.def, checkpoints=r.CHK,
                    extra=Dict("budget"=>BUDGET, "N"=>NSAMP, "iters"=>NITER,
                               "n_elite"=>NELITE, "alpha_nominal"=>ALPHA,
                               "arm"=>ARM,
                               "checkpoint_gamma_note"=>
                                 "NaN: CrossEntropyMethod has no conditioning schedule; " *
                                 "no gamma_k exists. See quantile_diagnostic.",
                               "quantile_diagnostic"=>r.QDIAG))

        # ---- pooled balance-MIS over all 25 proposal batches ----------------
        K_ = length(r.ρ) ÷ NSAMP                       # = 25
        GMAX = maximum(GRIDS[KS[1]])
        t_ri = time()
        idx = findall(r.ρ .<= GMAX); Xi = r.X[:, idx]; ρi = r.ρ[idx]
        p_i = pdf(PRIOR, Xi); qsum = copy(p_i)         # batch-1 proposal = prior
        for k in 2:K_
            mk, _ = r.CHK[k]
            qsum .+= proposal_pdf(mk, NaN, Xi, PRIOR)  # DIRECT density, no quadrature
        end
        wb = p_i ./ (qsum ./ K_); secs_reindex = time() - t_ri
        Ntot = length(r.ρ)

        cells = Dict{String,Any}()
        for K in KS
            G = snap.(GRIDS[K])
            pts = Any[]
            for γ in G
                est = sum(wb[ρi .<= γ]) / Ntot
                push!(pts, Dict("gamma"=>γ, "estimate"=>est, "truth"=>truthf(γ)))
            end
            cells[string(K)] = Dict{String,Any}(
                "gamma_grid"=>G, "per_threshold"=>pts,
                "n_evals_actual"=>r.n_evals, "n_runs"=>1,
                "n_grid_total"=>length(G),
                "secs_sim"=>r.secs, "secs_reindex_shared"=>secs_reindex,
                "secs_total"=>r.secs + secs_reindex,
                "fit_ess_reindex"=>ess_frac(wb))
        end
        out[ARM] = cells
        out["ok"] = true
        out["n_failures_found"] = count(r.ρ .<= 0.0)
        out["min_rho"] = minimum(r.ρ)
        out["fit_stats"] = r.fitstats
        out["quantile_diagnostic"] = r.QDIAG
        out["secs_seed_total"] = time() - t_all
        merge!(out, write_bundle(REC, OUTDIR, key,
                   Dict("seed"=>seed, "arm"=>ARM)))
    catch e
        out["error"] = sprint(showerror, e)
        @warn "seed $seed FAILED" exception=(e, catch_backtrace())
    end
    mkpath(joinpath(OUTDIR, "raw"))
    open(joinpath(OUTDIR, "raw", key*".json"), "w") do io; JSON.print(io, out, 2); end
    return out
end

ess_frac(w) = (s1 = sum(w); s2 = sum(abs2, w); s2 > 0 ? (s1^2 / s2) / length(w) : 0.0)

# ---- main -------------------------------------------------------------------
mkpath(OUTDIR)
@printf("\nEXP1 UNCONDITIONAL-CEM BASELINE\n")
@printf("  arm=%s  comp=%d  n_iter=%d  N=%d  n_elite=%d  max_weight=%.1f  budget=%d\n",
        ARM, COMP, NITER, NSAMP, NELITE, MAXW, BUDGET)
@printf("  seeds=%s\n  out=%s\n\n", string(SEEDS), OUTDIR)

rows = Any[]
for s in SEEDS
    t = time()
    r = run_seed(s)
    @printf("seed %2d  ok=%-5s  failures=%-6s min_rho=%-9s  %.1fs\n", s, r["ok"],
            get(r, "n_failures_found", "-"),
            haskey(r,"min_rho") ? string(round(r["min_rho"], digits=4)) : "-", time()-t)
    push!(rows, r)
end

# ---- config + meta ----------------------------------------------------------
open(joinpath(OUTDIR, "config.json"), "w") do io
    JSON.print(io, Dict{String,Any}(
        "experiment"=>"exp1_unconditional_cem_control",
        "arm"=>ARM,
        "control_definition"=>"CrossEntropyMethod with an unconditional EMGMM proposal",
        "problem"=>"unimodal_toy", "problem_def"=>"rho = 3 - min(x1,x2), x ~ N(0,I2)",
        "truth"=>"P(rho<=g) = Phi(-(3-g))^2 (closed form)",
        "dimension"=>2, "n_mixture_components"=>COMP,
        "n_iter"=>NITER, "batch_size"=>NSAMP, "budget_total"=>BUDGET,
        "n_elite"=>NELITE, "alpha_nominal"=>ALPHA,
        "elite_rule"=>"N_elite = (all rho<=0) ? N : #{rho<=0}; then max(N_elite, n_elite). " *
                      "src/baselines/cem.jl, train!(::CrossEntropyMethod, ::System.SystemParameters): " *
                      "applied to every batch, including the prior batch; the model is refit on " *
                      "each batch's clamped elites and the fitted model generates the next batch",
        "max_weight_fit"=>MAXW, "defensive_eta"=>0.0, "sigma_smoothing"=>0.0,
        "proposal_family"=>"CrossEntropyMethod(EMGMM(k=2)) - unconditional",
        "initial_proposal"=>"EMGMM(2,2) init = 2 x MvNormal(0,I) = nominal p(x); generates no batch: " *
                            "batch 1 is drawn from p(x), batch 2 from the fit to batch 1's elites",
        "seeds"=>SEEDS, "n_seeds"=>length(SEEDS), "Ks"=>KS,
        "grid_source"=>"paper/results/exp1/raw/seed*.json amortized.{K}.gamma_grid " *
                       "(identical across all 10 seeds)",
        "estimator"=>"pooled balance-MIS over all 25 proposal batches; " *
                     "q_mix = (1/25)(prior + sum_{k=2..25} q_k)",
        "density_evaluation"=>"DIRECT pdf(mixture, X); no quadrature (proposal_pdf(::EMGMM,...))",
        "curve_gating"=>"none: an unconditional proposal has no conditioning support limit. " *
                        "The conditioned arm's gamma_min=0.0 excludes nothing either.",
        "metric"=>"paper/experiments/cem_baseline/cem_metrics.jl",
    ), 2)
end
open(joinpath(OUTDIR, "meta.json"), "w") do io
    JSON.print(io, Dict{String,Any}(
        "julia_version"=>string(VERSION),
        "nthreads"=>Threads.nthreads(), "blas_threads"=>1,
        "pkg_versions"=>Dict("Distributions"=>string(pkgversion(Distributions)),
                             "JSON"=>string(pkgversion(JSON))),
        "environment"=>relpath(Base.active_project(), REPO),
    ), 2)
end
@printf("\nwrote %s\n", OUTDIR)
println("EXP1_CEM_DONE")
