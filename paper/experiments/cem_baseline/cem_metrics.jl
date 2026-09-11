# Experiment 1 metric reconstruction from stored raw state.
#
# Rebuilds each seed's pooled balance-MIS curve on the Exp1 threshold grid
# without rerunning the simulator, then computes ILE and coverage using the
# same conventions as `paper/scripts/metrics.py`.
#
# The evaluator also checks that the reconstructed curve matches the stored
# per-threshold estimates.
#
# Usage:
#   julia --project=. paper/experiments/cem_baseline/cem_metrics.jl \
#         --results DIR --out FILE [--arm KEY] [--validate] [--no-gate]

using ARIS
using Distributions, Random, Statistics, LinearAlgebra, Printf, JSON, Serialization
LinearAlgebra.BLAS.set_num_threads(1)

const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(REPO, "examples", "problems", "toy.jl"))
include(joinpath(REPO, "paper", "experiments", "balance.jl"))
# proposal_pdf methods for unconditional EMGMM checkpoints; must load after balance.jl
include(joinpath(@__DIR__, "uncond_emgmm.jl"))

# ---- argument parsing -------------------------------------------------------
function argval(flag, default=nothing)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    return ARGS[i+1]
end
const R        = argval("--results")
const ARM      = argval("--arm", "amortized")
const OUTPATH  = argval("--out")
const VALIDATE = "--validate" in ARGS
const NOGATE   = "--no-gate" in ARGS
R === nothing && error("--results is required")

# ---- problem definition (identical to exp1_kscaling.jl) ---------------------
const SYS   = UnimodalToy(3.0)
const PRIOR = System.prior(SYS)
truthf(γ)   = cdf(Normal(), -(3.0 - γ))^2
const P_LO, P_HI = 1e-3, cdf(Normal(), -3.0)^2
rawgrid(K)  = [3.0 + quantile(Normal(), sqrt(p))
               for p in exp10.(range(log10(P_LO), log10(P_HI), length=K))]

# ---- endpoint snapping -----------------------------------------------------
const SNAP_TOL = 1e-10
snap(γ) = abs(γ) < SNAP_TOL ? 0.0 : γ
grid(K) = snap.(rawgrid(K))

const KS = [10, 25, 50, 100]

# ---- metrics ---------------------------------------------------------------
"existing convention: mean over contributing points only (diagnostic / guard only)"
function ile_existing(curve, G)
    e = Float64[]
    for γ in G
        v = get(curve, γ, nothing); T = truthf(γ)
        (v isa Real && isfinite(v) && v > 0 && T > 0) && push!(e, abs(log(v) - log(T)))
    end
    return isempty(e) ? nothing : mean(e), length(e)
end

"""
Reported convention. Every grid point with truth > 0 contributes.
Missing / zero / non-finite estimates are SUBSTITUTED by `fl`.
Valid finite positive estimates are used AS IS — no upper clamp.
"""
function ile_full_noclamp(curve, G, fl)
    e = Float64[]; ncov = 0
    for γ in G
        v = get(curve, γ, nothing); T = truthf(γ)
        T > 0 || continue
        good = v isa Real && isfinite(v) && v > 0
        good && (ncov += 1)
        push!(e, abs(log(good ? v : fl) - log(T)))   # no upper clamp
    end
    return mean(e), ncov, length(e)
end

"clamped variant min(., 1.0), computed ONLY to measure whether a clamp would bind"
function ile_full_clamped(curve, G, fl)
    e = Float64[]
    for γ in G
        v = get(curve, γ, nothing); T = truthf(γ)
        T > 0 || continue
        good = v isa Real && isfinite(v) && v > 0
        push!(e, abs(log(min(good ? v : fl, 1.0)) - log(T)))
    end
    return mean(e)
end

# ---- raw-state bundle reader -----------------------------------------------
function bundle(seed, armtag)
    man = JSON.parsefile(joinpath(R, "raw_state", "seed$(seed).manifest.json"))
    raw = read(joinpath(R, "raw_state", "seed$(seed).arrays.bin"))
    chk = Serialization.deserialize(joinpath(R, "raw_state", "seed$(seed).models.jls"))
    function blk(name)
        b = first(filter(x -> x["name"] == name, man["blocks"]))
        v = collect(reinterpret(Float64, raw[b["offset"]+1 : b["offset"]+b["nbytes"]]))
        sh = Int.(b["shape"])
        return length(sh) == 1 ? v : reshape(v, sh...)
    end
    cp = first(filter(m -> m["tag"] == armtag, chk))["checkpoints"]
    return (X=blk("$(armtag)/X"), ρ=blk("$(armtag)/rho"),
            iter=Int.(blk("$(armtag)/iter")), CHK=cp)
end

# ---- main -------------------------------------------------------------------
seeds = sort([parse(Int, match(r"seed(\d+)\.json", basename(f))[1])
              for f in filter(f -> occursin(r"^seed\d+\.json$", basename(f)),
                              readdir(joinpath(R, "raw"), join=true))])
@printf("Exp1 evaluator | results=%s | arm=%s | seeds=%s\n", R, ARM, string(seeds))

cells = Dict{String,Any}()
recon_maxrel = Ref(0.0); recon_n = Ref(0)

for seed in seeds
    row = JSON.parsefile(joinpath(R, "raw", "seed$(seed).json"))
    b   = bundle(seed, ARM)
    K_  = maximum(b.iter)
    Ntot = length(b.ρ)
    GMAX = maximum(grid(KS[1]))
    γmin = NOGATE ? -Inf : minimum(γk for (_, γk) in b.CHK[2:end])

    # pooled balance-MIS rebuilt offline. NO SIMULATION.
    idx = findall(b.ρ .<= GMAX); Xi = b.X[:, idx]; ρi = b.ρ[idx]
    p_i = pdf(PRIOR, Xi); qsum = copy(p_i)
    for k in 2:K_
        mk, γk = b.CHK[k]
        qsum .+= proposal_pdf(mk, γk, Xi, PRIOR)
    end
    wb = p_i ./ (qsum ./ K_)

    for K in KS
        G   = grid(K)
        nev = row[ARM][string(K)]["n_evals_actual"]
        curve = Dict(γ => (γ >= γmin ? sum(wb[ρi .<= γ]) / Ntot : nothing) for γ in G)

        # self-check: the offline rebuild must reproduce the STORED estimates
        for pt in row[ARM][string(K)]["per_threshold"]
            st = pt["estimate"]; st === nothing && continue
            mine = curve[snap(pt["gamma"])]
            (mine isa Real && mine > 0 && st > 0) || continue
            recon_maxrel[] = max(recon_maxrel[], abs(mine - st) / st); recon_n[] += 1
        end

        fl = 1.0 / nev
        ile, ncov, ntot = ile_full_noclamp(curve, G, fl)
        ile_c           = ile_full_clamped(curve, G, fl)
        old, nold       = ile_existing(curve, G)
        if ncov == ntot && old !== nothing
            @assert isapprox(ile, old; rtol=1e-12) "ILE_full != ILE_existing at 100% coverage"
        end

        key = "$(ARM)_K$(K)"
        d = get!(cells, key, Dict{String,Any}(
            "arm"=>ARM, "K"=>K, "seeds"=>Int[],
            "ile"=>Float64[], "ile_clamped_for_comparison"=>Float64[],
            "coverage"=>Float64[], "n_pos"=>Int[], "n_grid"=>ntot,
            "floor"=>Float64[], "n_evals_actual"=>Int[],
            "curves"=>Dict{String,Any}()))
        push!(d["seeds"], seed)
        push!(d["ile"], ile); push!(d["ile_clamped_for_comparison"], ile_c)
        push!(d["coverage"], ncov/ntot); push!(d["n_pos"], ncov)
        push!(d["floor"], fl); push!(d["n_evals_actual"], nev)
        d["curves"][string(seed)] = [get(curve, γ, nothing) for γ in G]
    end
    @printf("  seed %2d rebuilt (iters=%d, gamma_min=%.4g, samples<=GMAX=%d)\n",
            seed, K_, γmin, length(idx))
end

med(v) = (w = filter(isfinite, v); isempty(w) ? NaN : median(w))

println()
@printf("offline rebuild vs STORED estimates: max rel diff = %.3e over %d points\n",
        recon_maxrel[], recon_n[])
println()
println("="^92)
@printf("%-12s %5s | %-18s %-18s | %8s %8s\n",
        "arm", "K", "median ILE", "median ILE (clamped)", "cov med", "cov min")
println("="^92)
for K in KS
    d = cells["$(ARM)_K$(K)"]
    @printf("%-12s %5d | %-18.10f %-18.10f | %7.1f%% %7.1f%%\n",
            ARM, K, med(d["ile"]), med(d["ile_clamped_for_comparison"]),
            100*median(d["coverage"]), 100*minimum(d["coverage"]))
end
println("="^92)

# ---- validation against the amortized arm's Exp1 median ILEs ----------------
const AMORTIZED_MEDIANS = Dict(10=>0.0146378911, 25=>0.0150585530, 50=>0.0152046734, 100=>0.0148790676)
validation = Dict{String,Any}()
if VALIDATE
    println()
    println("VALIDATION vs amortized-arm Exp1 median ILEs")
    println("-"^92)
    @printf("%5s %-20s %-20s %-14s %s\n", "K", "reference median", "evaluator", "abs diff", "verdict")
    local allok = true
    for K in KS
        got = med(cells["$(ARM)_K$(K)"]["ile"])
        want = AMORTIZED_MEDIANS[K]
        diff = abs(got - want)
        # reference medians are quoted to 10 decimals; agreement "up to numerical
        # precision" is therefore |diff| <= 5e-11 (half a unit in the last quoted place).
        ok = diff <= 5e-11
        allok &= ok
        @printf("%5d %-20.10f %-20.10f %-14.3e %s\n", K, want, got, diff, ok ? "PASS" : "FAIL")
        validation[string(K)] = Dict("reference"=>want, "evaluator"=>got,
                                     "abs_diff"=>diff, "pass"=>ok)
    end
    println("-"^92)
    validation["all_pass"] = allok
    validation["tolerance_abs"] = 5e-11
    if allok
        println("VALIDATION: PASS — evaluator reproduces all four amortized-arm medians.")
    else
        println("VALIDATION: FAIL — STOP. Do not interpret any CEM result.")
    end
end

if OUTPATH !== nothing
    out = Dict{String,Any}(
        "arm" => ARM,
        "ile_formula" => "mean over ALL grid points with truth>0 of |log(est_or_floor) - log(truth)|",
        "floor_rule" => "P~ = P^ if finite and > 0, else 1/n_evals_actual; valid estimates never modified",
        "coverage_rule" => "#{finite positive estimate} / #{truth > 0}",
        "endpoint_snap" => Dict("tolerance"=>SNAP_TOL,
                                "raw_endpoint"=>rawgrid(10)[end], "snapped_to"=>0.0),
        "offline_rebuild_check" => Dict("max_rel_diff"=>recon_maxrel[], "n_points"=>recon_n[]),
        "validation" => validation,
        "cells" => cells)
    open(OUTPATH, "w") do io; JSON.print(io, out, 2); end
    println("wrote ", OUTPATH)
end
