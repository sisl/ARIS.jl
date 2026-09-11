# Experiment 3: multimodal tail-race conditioning ablation.
#
# Compares a robustness-conditioned adaptive proposal against an unconditional
# GMM baseline under a matched simulation budget. The experiment isolates the
# effect of robustness conditioning when dominant failure modes change across
# thresholds.
#
# Both methods use three-component GMMs, the same seed set, target threshold,
# and matched elite fraction. The unconditional baseline does not define an
# adaptive robustness schedule, so schedule-reach diagnostics apply only to the
# conditioned method.
#
# Usage:
#   julia --project=. paper/experiments/drivers/exp3_tailrace.jl [nseeds] [with_sigma_arm(0|1)]
#
# Reruns default to `paper/rerun/exp3_tailrace/`.

using Distributed
const NSEEDS   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
const WITHSIG  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) == 1 : false
const SEEDS    = collect(1:NSEEDS)
const C2LIST   = [16.0, 18.0, 20.0, 23.0]
const C1       = 5.5
const COMP     = 3
const NSAMP    = 2000
const NITER    = 40
const QUANT    = 0.15
const REGIME_RULE = "regime = (l2u+l2l == 0) ? \"locked\" : " *
                    "(Pm2/truth > 0.1) ? \"amplified\" : " *
                    "(small_mix > 50 && l2u+l2l > 1000) ? \"collapsed\" : \"trace\""
addprocs(max(0, min(8, Sys.CPU_THREADS - 2) - nworkers() + 1))

@everywhere include(joinpath(@__DIR__, "metrics.jl"))
@everywhere using .FinalMetrics
@everywhere include(joinpath(@__DIR__, "rawstate.jl"))
@everywhere using .RawState

@everywhere begin
    using ARIS
    using Distributions, Random, Statistics, LinearAlgebra, JSON
    LinearAlgebra.BLAS.set_num_threads(1)
    include(joinpath(@__DIR__, "..", "..", "..", "examples", "problems", "tailrace.jl"))
    include(joinpath(@__DIR__, "..", "sigma_smoothed.jl"))

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

    const C1_W = $C1; const COMP_W = $COMP; const NSAMP_W = $NSAMP
    const NITER_W = $NITER; const QUANT_W = $QUANT

    ess_r(w) = isempty(w) ? NaN : sum(w)^2 / sum(w .^ 2)
    classify(Xf, c1, c2) = ((c1 .- Xf[1, :]) .<= (c2 .- Xf[2, :] .^ 2), Xf[2, :] .> 0)
    # closed-form truth at gamma = 0: P1 + P2 - P1*P2
    truth_at0(c2) = (P1 = cdf(Normal(), -C1_W); P2 = 2 * cdf(Normal(), -sqrt(c2)); P1 + P2 - P1 * P2)

    function run_one(c2::Float64, arm::String, σ::Float64, seed::Int)
        key = "c2$(c2)_$(arm)_seed$(seed)"
        banked(key) && return JSON.parsefile(joinpath(Main.OUTDIR,"raw",key*".json"))
        t0 = time(); truth = truth_at0(c2)
        row = Dict{String,Any}("c2"=>c2, "arm"=>arm, "sigma"=>σ, "seed"=>seed, "comp"=>COMP_W,
                               "truth"=>truth, "ok"=>false,
                               "n_samples"=>NSAMP_W, "n_iter"=>NITER_W,
                               "budget_requested"=>NSAMP_W * NITER_W)
        REC = RunRecorder()                  # raw state for offline re-evaluation
        try
            SYS = TailRaceToy(; c1=C1_W, c2=c2)
            Random.seed!(seed); reset_fit_stats!()
            γtraj = Float64[]; fh = Dict(:m1=>0, :m2u=>0, :m2l=>0)
            CHK = Any[]                      # per-iteration proposal checkpoints
            local x, ρ, w
            if arm == "unconditional"
                cem = CrossEntropyMethod(model=EMGMM(COMP_W, 2), xdim=2, depth=1, sdim=2,
                    n_iter=NITER_W, n_samples=NSAMP_W, n_elite=round(Int, QUANT_W * NSAMP_W))
                train!(cem, SYS)
                x = Float64.(cem.buffer[:x]); ρ = Float64.(cem.buffer[:ρ]); w = Float64.(cem.buffer[:w])
                record_run!(REC, "unconditional"; X=x, rho=ρ, w=w,
                            logq=Float64.(cem.buffer[:logpdf]),
                            iter=repeat(1:NITER_W, inner=NSAMP_W)[1:length(ρ)],
                            defmask=Vector{Bool}(cem.buffer[:defensive]), checkpoints=nothing,
                            extra=Dict("c2"=>c2, "arm"=>arm,
                                       "note"=>"CrossEntropyMethod: no JointModel proposal, so no marginal-density checkpoints exist"))
                row["n_elite"] = round(Int, QUANT_W * NSAMP_W)
                row["schedule_note"] = "CrossEntropyMethod has no sampling_strategy: gamma_min and first-hit are undefined for this arm"
            else
                cb = (cv, D) -> begin
                    push!(γtraj, Float64(cv.sampling_strategy.value)); it = length(γtraj)
                    push!(CHK, (deepcopy(cv.model), Float64(cv.sampling_strategy.value)))
                    xb = Float64.(D[:x]); ρb = Float64.(D[:ρ]); fb = ρb .<= 0.0
                    any(fb) || return
                    m1, up = classify(xb[:, fb], C1_W, c2)
                    fh[:m1]  == 0 && any(m1) && (fh[:m1] = it)
                    fh[:m2u] == 0 && any(.!m1 .& up) && (fh[:m2u] = it)
                    fh[:m2l] == 0 && any(.!m1 .& .!up) && (fh[:m2l] = it)
                end
                base = JointModel(Normal(), ConditionalEMGMM(COMP_W, 3, 3))
                model = σ > 0 ? SigmaSmoothed(base, σ; q=QUANT_W) : base
                cv = ConditionalValidation(model=model, xdim=2, depth=1, sdim=2,
                    n_samples=NSAMP_W, n_iter=NITER_W,
                    sampling_strategy=QuantileSampling(QUANT_W), max_weight=10.0, log_callback=cb)
                train!(cv, SYS)
                x = Float64.(cv.buffer[:x]); ρ = Float64.(cv.buffer[:ρ]); w = Float64.(cv.buffer[:w])
                record_run!(REC, arm; X=x, rho=ρ, w=w, logq=Float64.(cv.buffer[:logpdf]),
                            iter=repeat(1:NITER_W, inner=NSAMP_W)[1:length(ρ)],
                            defmask=Vector{Bool}(cv.buffer[:defensive]), checkpoints=CHK,
                            extra=Dict("c2"=>c2, "arm"=>arm, "sigma"=>σ,
                                       "N"=>NSAMP_W, "iters"=>NITER_W))
                row["adaptive_quantile"] = QUANT_W
            end

            fs = ρ .<= 0.0; M = length(ρ); Xf = x[:, fs]; wf = w[fs]
            m1, up = classify(Xf, C1_W, c2)
            i2u = findall(.!m1 .& up); i2l = findall(.!m1 .& .!up); i2 = findall(.!m1)
            pm(idx) = isempty(idx) ? 0.0 : sum(wf[idx]) / M
            l1, l2u, l2l = count(m1), length(i2u), length(i2l)
            P = sum(fs .* w) / M; Pm2 = pm(i2); Pm1 = pm(findall(m1))
            st = get_fit_stats()
            regime = (l2u + l2l == 0) ? "locked" :
                     (Pm2 / truth > 0.1) ? "amplified" :
                     (st[:small_mix] > 50 && l2u + l2l > 1000) ? "collapsed" : "trace"

            merge!(row, Dict{String,Any}(
                "ok"=>true, "P_hat"=>P,
                "P_over_truth"=>P / truth,
                "signed_relerr"=>(P - truth) / truth, "abs_relerr"=>abs(P - truth) / truth,
                "P_mode1"=>Pm1, "P_mode2"=>Pm2,
                "n_fail"=>count(fs), "n_mode1"=>l1, "n_mode2_upper"=>l2u, "n_mode2_lower"=>l2l,
                "found_all_three_lobes"=>(l1 > 0 && l2u > 0 && l2l > 0),
                "regime"=>regime, "regime_rule"=>$REGIME_RULE,
                "gamma_min"=>(length(γtraj) > 1 ? minimum(γtraj[2:end]) : nothing),
                "adapted"=>(length(γtraj) > 1 ? minimum(γtraj[2:end]) <= 0.0 : nothing),
                "firsthit_mode1"=>(arm == "unconditional" ? nothing : fh[:m1]),
                "firsthit_mode2_upper"=>(arm == "unconditional" ? nothing : fh[:m2u]),
                "firsthit_mode2_lower"=>(arm == "unconditional" ? nothing : fh[:m2l]),
                "small_mix"=>st[:small_mix], "small_det"=>st[:small_det],
                "tail_ess_mode2"=>(length(i2) < 2 ? nothing : ess_r(wf[i2])),
                "tail_ess_all"=>(count(fs) < 2 ? nothing : ess_r(wf)),
                "fit_ess"=>ess_frac(w),
                "fit_method"=>(st[:kmeans_fallback] > 0 ? "fallback_assisted" : "em"),
                "n_evals_actual"=>M, "secs"=>time() - t0))
        catch e
            row["error"] = sprint(showerror, e)
            row["backtrace"] = string.(stacktrace(catch_backtrace())[1:min(8,end)])
        end
        if isdefined(Main, :OUTDIR)
            try
                merge!(row, write_bundle(REC, Main.OUTDIR, "c2$(c2)_$(arm)_seed$(seed)",
                        Dict("seed"=>seed, "c2"=>c2, "arm"=>arm,
                             "marginal"=>marginal_settings())))
            catch e
                row["raw_state_error"] = sprint(showerror, e)
            end
        end
        return persist_row("c2$(c2)_$(arm)_seed$(seed)", row)
    end
end

using Printf, JSON, Statistics
include(joinpath(@__DIR__, "provenance.jl")); using .RunProvenance
include(joinpath(@__DIR__, "metrics.jl")); using .FinalMetrics

ARMS = [("conditioned", 0.0), ("unconditional", 0.0)]
WITHSIG && append!(ARMS, [("conditioned_sigma0.05", 0.05), ("conditioned_sigma0.1", 0.1)])

config = Dict{String,Any}(
    "experiment"=>"exp3_tailrace",
    "measured_quantity"=>"effect of robustness-conditioning when the dominant failure mode changes across thresholds",
    "problem"=>"tailrace", "problem_def"=>"rho(x) = min(c1 - x1, c2 - x2^2), x ~ N(0,I2)",
    "c1"=>C1, "c2_values"=>C2LIST,
    "truth"=>"P(rho<=0) = P1 + P2 - P1*P2, P1 = Phi(-c1), P2 = 2*Phi(-sqrt(c2)) (closed form)",
    "truth_per_c2"=>Dict(string(c2) => (P1 = cdf(Normal(), -C1); P2 = 2 * cdf(Normal(), -sqrt(c2)); P1 + P2 - P1 * P2) for c2 in C2LIST),
    "primary_arms"=>["conditioned", "unconditional"],
    "secondary_sigma_arms_included"=>WITHSIG,
    "sigma_smoothing_primary"=>0.0, "sigma_note"=>"primary comparison uses the HARD indicator (sigma=0)",
    "seeds"=>SEEDS, "n_seeds"=>NSEEDS, "dimension"=>2,
    "n_mixture_components"=>COMP, "batch_size"=>NSAMP, "n_iter"=>NITER,
    "budget_total_per_run"=>NSAMP * NITER,
    "budget_matching"=>"both primary arms consume n_samples x n_iter = $(NSAMP*NITER) robustness evaluations",
    "adaptive_quantile"=>QUANT, "cem_n_elite"=>round(Int, QUANT * NSAMP),
    "elite_fraction_matched"=>QUANT, "target_threshold"=>0.0, "max_weight_fit"=>10.0,
    "defensive_eta"=>0.0, "proposal_family_A"=>"JointModel(Normal(), ConditionalEMGMM(k=3))",
    "proposal_family_B"=>"CrossEntropyMethod(EMGMM(k=3)) — unconditional",
    "regime_rule"=>REGIME_RULE,
    "regime_note"=>"regime is generated by this run's code; the rule above is the exact deterministic classifier",
    "unmatched_structural_difference"=>"CrossEntropyMethod exposes no sampling_strategy, so gamma_min / first-hit are undefined for the unconditional arm",
)
dir, meta = init_group("exp3_tailrace", config)
tasks = [(c2, a, σ, s) for c2 in C2LIST for (a, σ) in ARMS for s in SEEDS]
@printf("exp3_tailrace → %s\n  %d c2 × %d arms × %d seeds = %d runs, %d workers\n",
        dir, length(C2LIST), length(ARMS), NSEEDS, length(tasks), nworkers())

@everywhere const OUTDIR = $dir
rows = pmap(t -> Main.run_one(t...), tasks)

summary = Dict{String,Any}("experiment"=>"exp3_tailrace", "config"=>config, "cells"=>Dict{String,Any}())
for c2 in C2LIST, (a, σ) in ARMS
    key = "c2$(c2)_$(a)"
    cr = [r for r in rows if r["c2"] == c2 && r["arm"] == a]
    ok = [r for r in cr if r["ok"] === true]
    pv = [r["P_over_truth"] for r in ok]
    regs = [r["regime"] for r in ok]
    summary["cells"][key] = Dict{String,Any}(
        "c2"=>c2, "arm"=>a, "sigma"=>σ,
        "seed_accounting"=>Dict("n_requested"=>NSEEDS, "n_completed"=>length(ok),
            "n_failed"=>length(cr)-length(ok), "seeds_requested"=>SEEDS,
            "failure_records"=>[Dict("seed"=>r["seed"],"error"=>get(r,"error",nothing))
                                for r in cr if r["ok"] !== true]),
        "truth"=>isempty(ok) ? nothing : ok[1]["truth"],
        "P_over_truth_per_seed"=>pv,
        "P_over_truth_median"=>isempty(pv) ? nothing : median(pv),
        "P_over_truth_mean"=>isempty(pv) ? nothing : mean(pv),
        "P_over_truth_min"=>isempty(pv) ? nothing : minimum(pv),
        "P_over_truth_max"=>isempty(pv) ? nothing : maximum(pv),
        "signed_relerr_median"=>isempty(ok) ? nothing : median([r["signed_relerr"] for r in ok]),
        "abs_relerr_median"=>isempty(ok) ? nothing : median([r["abs_relerr"] for r in ok]),
        "regime_counts"=>Dict(g => count(==(g), regs) for g in unique(regs)),
        "found_all_three_lobes_count"=>count(r -> r["found_all_three_lobes"], ok),
        "n_fail_median"=>isempty(ok) ? nothing : median([r["n_fail"] for r in ok]),
        "small_mix_mean"=>isempty(ok) ? nothing : mean([r["small_mix"] for r in ok]),
        "n_evals_actual_median"=>isempty(ok) ? nothing : median([r["n_evals_actual"] for r in ok]),
        "gamma_min_median"=>(v = [r["gamma_min"] for r in ok if r["gamma_min"] isa Real];
                             isempty(v) ? nothing : median(v)),
        "adapted_count"=>count(r -> r["adapted"] === true, ok),
        "tail_ess_mode2_median"=>(v = [r["tail_ess_mode2"] for r in ok if r["tail_ess_mode2"] isa Real];
                                  isempty(v) ? nothing : median(v)),
        "fit_method_fallback_assisted_count"=>count(r -> r["fit_method"] == "fallback_assisted", ok),
    )
end
write_summary(dir, summary)

@printf("\n%-8s %-24s %-4s %-12s %-10s %-26s %s\n","c2","arm","n","P/truth med","lobes3","regimes","evals(med)")
for c2 in C2LIST, (a, _) in ARMS
    c = summary["cells"]["c2$(c2)_$(a)"]
    @printf("%-8s %-24s %-4d %-12s %-10s %-26s %s\n", string(c2), a,
        c["seed_accounting"]["n_completed"],
        c["P_over_truth_median"] === nothing ? "—" : string(round(c["P_over_truth_median"], sigdigits=5)),
        "$(c["found_all_three_lobes_count"])/$(c["seed_accounting"]["n_completed"])",
        string(c["regime_counts"]),
        c["n_evals_actual_median"] === nothing ? "—" : string(round(Int, c["n_evals_actual_median"])))
end
println("EXP3_DONE")
