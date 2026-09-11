# Experiment 1 endpoint-curve reconstruction.
#
# Rebuilds the amortized curve from stored Exp1 raw state and proposal
# checkpoints, including the target endpoint at γ = 0, without rerunning the
# simulator. Per-threshold curves are read from the stored experiment rows.
#
# Output curves can be passed to `paper/scripts/metrics.py exp1` to compute
# the reported ILE and coverage metrics.
#
# Usage:
#   julia --project=. paper/experiments/drivers/exp1_endpoint_curves.jl

using ARIS
using Distributions, Random, Statistics, LinearAlgebra, Printf, JSON, Serialization
LinearAlgebra.BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "..", "..", "..", "examples", "problems", "toy.jl"))
include(joinpath(@__DIR__, "..", "balance.jl"))

const R         = get(ENV, "EXP1_RESULTS_DIR", normpath(joinpath(@__DIR__, "..", "..", "results", "exp1")))
const RAW_STATE = get(ENV, "RAW_STATE_DIR", joinpath(R, "raw_state"))
const OUTFILE   = get(ENV, "EXP1_CURVES_OUT", normpath(joinpath(@__DIR__, "..", "..", "rerun", "exp1_endpoint_curves.json")))
const SYS    = UnimodalToy(3.0)
const PRIOR  = System.prior(SYS)
truthf(γ)    = cdf(Normal(), -(3.0 - γ))^2
const P_LO, P_HI = 1e-3, cdf(Normal(), -3.0)^2
rawgrid(K)   = [3.0 + quantile(Normal(), sqrt(p))
                for p in exp10.(range(log10(P_LO), log10(P_HI), length=K))]

# --- endpoint snapping -------------------------------------------------------
# The construction makes the final grid point exactly the target γ = 0; floating point lands it at
# -5.77e-15. Snap anything within 1e-10 of the target so the `γ >= γ_min` gate behaves as intended.
const SNAP_TOL = 1e-10
snap(γ) = abs(γ) < SNAP_TOL ? 0.0 : γ
grid(K) = snap.(rawgrid(K))

# --- metrics -----------------------------------------------------------------
"in-driver convention (metrics.jl): mean over contributing points only"
function ile_existing(curve, G)
    e = Float64[]
    for γ in G
        v = get(curve, γ, nothing); T = truthf(γ)
        (v isa Real && isfinite(v) && v > 0 && T > 0) && push!(e, abs(log(v) - log(T)))
    end
    return isempty(e) ? nothing : mean(e), length(e)
end

"floor convention: every grid point contributes; zero/null SUBSTITUTED by `fl` (valid estimates untouched)"
function ile_full(curve, G, fl)
    e = Float64[]; ncov = 0
    for γ in G
        v = get(curve, γ, nothing); T = truthf(γ)
        T > 0 || continue
        good = v isa Real && isfinite(v) && v > 0
        good && (ncov += 1)
        push!(e, abs(log(min(good ? v : fl, 1.0)) - log(T)))
    end
    return mean(e), ncov, length(e)
end

# --- bundle reader -----------------------------------------------------------
function bundle(seed)
    man = JSON.parsefile(joinpath(RAW_STATE, "seed$(seed).manifest.json"))
    raw = read(joinpath(RAW_STATE, "seed$(seed).arrays.bin"))
    chk = Serialization.deserialize(joinpath(RAW_STATE, "seed$(seed).models.jls"))
    function blk(name)
        b = first(filter(x -> x["name"] == name, man["blocks"]))
        v = collect(reinterpret(Float64, raw[b["offset"]+1 : b["offset"]+b["nbytes"]]))
        sh = Int.(b["shape"])
        return length(sh) == 1 ? v : reshape(v, sh...)
    end
    cp = first(filter(m -> m["tag"] == "amortized", chk))["checkpoints"]
    return (X=blk("amortized/X"), ρ=blk("amortized/rho"), iter=Int.(blk("amortized/iter")), CHK=cp)
end

# --- main --------------------------------------------------------------------
seeds = sort([parse(Int, match(r"seed(\d+)\.json", basename(f))[1])
              for f in filter(f -> occursin(r"^seed\d+\.json$", basename(f)),
                              readdir(joinpath(R, "raw"), join=true))])
const KS = [10, 25, 50, 100]
results = Dict{String,Any}()
recon_maxrel = Ref(0.0); recon_n = Ref(0)

for seed in seeds
    row = JSON.parsefile(joinpath(R, "raw", "seed$(seed).json"))
    b   = bundle(seed)
    K_  = maximum(b.iter)
    Ntot = length(b.ρ)
    GMAX = maximum(grid(KS[1]))
    γmin = minimum(γk for (_, γk) in b.CHK[2:end])

    # rebuild balance-MIS weights offline (no simulation)
    idx = findall(b.ρ .<= GMAX); Xi = b.X[:, idx]; ρi = b.ρ[idx]
    p_i = pdf(PRIOR, Xi); qsum = copy(p_i)
    for k in 2:K_
        mk, γk = b.CHK[k]
        qsum .+= proposal_pdf(mk, γk, Xi, PRIOR)
    end
    wb = p_i ./ (qsum ./ K_)

    for K in KS
        G = grid(K)
        nev = row["amortized"][string(K)]["n_evals_actual"]
        curve_a = Dict(γ => (γ >= γmin ? sum(wb[ρi .<= γ]) / Ntot : nothing) for γ in G)

        # self-check: offline rebuild must reproduce the STORED estimates where they exist
        for pt in row["amortized"][string(K)]["per_threshold"]
            st = pt["estimate"]; st === nothing && continue
            mine = curve_a[snap(pt["gamma"])]
            (mine isa Real && mine > 0 && st > 0) || continue
            recon_maxrel[] = max(recon_maxrel[], abs(mine - st) / st); recon_n[] += 1
        end

        curve_p = Dict{Float64,Any}()
        nev_p = row["perthreshold"][string(K)]["n_evals_actual"]
        for pt in row["perthreshold"][string(K)]["per_threshold"]
            curve_p[snap(pt["gamma"])] = pt["estimate"]
        end

        for (arm, curve, ne) in (("amortized", curve_a, nev), ("perthreshold", curve_p, nev_p))
            old, nold = ile_existing(curve, G)
            fl_budget = 1.0 / ne
            new, ncov, ntot = ile_full(curve, G, fl_budget)
            n6, _, _ = ile_full(curve, G, 1e-6)
            n8, _, _ = ile_full(curve, G, 1e-8)
            # guard: with full coverage no substitution occurs, so the two metrics must agree
            if ncov == ntot && old !== nothing
                @assert isapprox(new, old; rtol=1e-12) "ILE_full != ILE_existing at 100% coverage"
            end
            key = "$(arm)_K$(K)"
            d = get!(results, key, Dict{String,Any}("arm"=>arm, "K"=>K,
                        "ile_existing"=>Float64[], "n_existing"=>Int[],
                        "ile_full"=>Float64[], "ile_full_1e6"=>Float64[], "ile_full_1e8"=>Float64[],
                        "coverage"=>Float64[], "floor_budget"=>fl_budget,
                        "n_grid"=>ntot, "seeds"=>Int[], "curves"=>Dict{String,Any}()))
            push!(d["ile_existing"], old === nothing ? NaN : old)
            push!(d["n_existing"], nold)
            push!(d["ile_full"], new); push!(d["ile_full_1e6"], n6); push!(d["ile_full_1e8"], n8)
            push!(d["coverage"], ncov / ntot)
            push!(d["seeds"], seed)
            d["curves"][string(seed)] = [get(curve, γ, nothing) for γ in G]
        end
    end
    @printf("  seed %d done (K_=%d, gamma_min=%.3g, covered=%d)\n", seed, K_, γmin, length(idx))
end

med(v) = (w = filter(isfinite, v); isempty(w) ? NaN : median(w))

println()
@printf("offline rebuild vs stored estimates: max rel diff = %.3e over %d points\n",
        recon_maxrel[], recon_n[])
println()
println("="^96)
@printf("%-14s %4s | %10s %6s | %9s %9s %9s %9s\n",
        "arm", "K", "ILE(exist)", "n_pts", "cover", "ILE@1/B", "ILE@1e-6", "ILE@1e-8")
println("="^96)
for K in KS, arm in ("amortized", "perthreshold")
    d = results["$(arm)_K$(K)"]
    @printf("%-14s %4d | %10.6g %6.1f | %8.1f%% %9.5g %9.5g %9.5g\n",
            arm, K, med(d["ile_existing"]), median(d["n_existing"]),
            100*median(d["coverage"]), med(d["ile_full"]),
            med(d["ile_full_1e6"]), med(d["ile_full_1e8"]))
end
println("="^96)

out = Dict{String,Any}(
    "source" => "rebuilt offline from paper/results/exp1/raw and the Exp1 raw-state bundle",
    "simulation_rerun" => false,
    "endpoint_snap" => Dict("tolerance"=>SNAP_TOL, "raw_endpoint"=>rawgrid(10)[end], "snapped_to"=>0.0),
    "floor_rule" => "est<=0 or null -> floor (SUBSTITUTION only; valid estimates never altered)",
    "floors_reported" => Dict("primary"=>"1/n_evals_actual", "sensitivity"=>[1e-6, 1e-8]),
    "ile_full_formula" => "mean over ALL grid points of |log(min(est_or_floor, 1)) - log(truth)|",
    "offline_rebuild_check" => Dict("max_rel_diff"=>recon_maxrel[], "n_points"=>recon_n[]),
    "cells" => results)
mkpath(dirname(OUTFILE))
open(OUTFILE, "w") do io; JSON.print(io, out, 2); end
println("wrote ", OUTFILE, "; run: python paper/scripts/metrics.py exp1 --exp1-curves ", OUTFILE)
