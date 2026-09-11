# Experiment 4 balance-MIS re-indexing and diagnostics.
#
# Recomputes pooled balance-MIS weights from stored raw state and proposal
# checkpoints, then evaluates the amortized estimator across the Exp4
# threshold grid without rerunning the simulator.
#
# Also reports proposal-of-origin estimates, weight-concentration diagnostics,
# and a consistency check against densities recorded at sample time.
#
# Usage:
#   RAW_STATE_DIR=<dir> julia --project=paper/envs/idm \
#       paper/scripts/exp4_reindex.jl SEED [OUTDIR]

using Serialization, LinearAlgebra, Statistics, Printf, JSON
using ARIS, Distributions
LinearAlgebra.BLAS.set_num_threads(1)

const PAPER = normpath(joinpath(@__DIR__, ".."))
include(joinpath(PAPER, "experiments", "conditional_sgm.jl"))
include(joinpath(PAPER, "experiments", "eta_mixture.jl"))
include(joinpath(PAPER, "experiments", "balance.jl"))
# the same method extensions as experiments/drivers/exp4_idm.jl
proposal_pdf(m::EtaMixture, γ, X, prior; kw...) =
    (1 - m.η) .* proposal_pdf(m.base, γ, X, prior; kw...) .+ m.η .* pdf(prior, X)
ARIS.conditional_logpdf_at(traj::ConditionalSGM, r::Real, X::AbstractMatrix) =
    Distributions.logpdf(_cond(traj, Float64(r)), X)

const RS = get(ENV, "RAW_STATE_DIR", joinpath(PAPER, "results", "exp4", "raw_state"))
const GG = Float64[0.434, 0.38, 0.32, 0.26, 0.20, 0.14, 0.08, 0.0]
const GMAX = maximum(GG)
seed = parse(Int, ARGS[1])
outdir = length(ARGS) >= 2 ? ARGS[2] : joinpath(PAPER, "rerun", "exp4_reindex")
mkpath(outdir)

man = JSON.parsefile(joinpath(RS, "seed$(seed).manifest.json"))
raw = read(joinpath(RS, "seed$(seed).arrays.bin"))
B = Dict{String,Any}()
for b in man["blocks"]
    off = Int(b["offset"]); nb = Int(b["nbytes"]); shp = Int.(b["shape"])
    v = reinterpret(Float64, @view raw[off+1:off+nb]); nm = split(String(b["name"]), "/")[end]
    B[nm] = length(shp) == 2 ? reshape(collect(v), shp[1], shp[2]) : collect(v)
end
X = B["X"]; ρ = B["rho"]; wown = B["w"]; logq = B["logq"]; iter = Int.(B["iter"]); def = B["defensive"] .!= 0
chks = Serialization.deserialize(joinpath(RS, "seed$(seed).models.jls"))
CHK = first(e for e in chks if String(e["tag"]) == "a_primary_sgm")["checkpoints"]
γ_chk = [Float64(g) for (_, g) in CHK]
γmin = minimum(γ_chk[2:end])                     # adaptive-schedule reach
PRIOR = product_distribution([Normal(0.0, 1.0) for _ in 1:150])
Ntot = length(ρ); N = 1000; K = Ntot ÷ N

idx = findall(ρ .<= GMAX)
Xi = X[:, idx]; ρi = ρ[idx]; iti = iter[idx]; wi = wown[idx]; lqi = logq[idx]; defi = def[idx]
p_i = pdf(PRIOR, Xi)
qsum = copy(p_i)                                  # batch-1 proposal is the prior
ctrl = Float64[]
t0 = time()
for k in 2:K
    mk, γk = CHK[k]
    qk = proposal_pdf(mk, γk, Xi, PRIOR)
    qsum .+= qk
    own = findall(iti .== k)
    isempty(own) || push!(ctrl, maximum(abs, filter(isfinite, log.(qk[own]) .- lqi[own])))
    @printf("  seed%-2d k=%2d γ=%.6f (%.0fs)\n", seed, k, γk, time() - t0); flush(stdout)
end
wb = p_i ./ (qsum ./ K)
secs = time() - t0

essabs(w) = (s = sum(w); (s <= 0 || !isfinite(s)) ? nothing : s^2 / sum(abs2, w))
rows = Any[]
for γ in GG
    m = ρi .<= γ; tot = sum(wb[m])
    push!(rows, Dict(
        "gamma" => γ,
        "estimate" => tot / Ntot,
        "n_hits" => count(m), "n_hits_def" => count(m .& defi),
        "P_own" => sum(wi[m]) / Ntot,
        "ess_abs" => (count(m) > 0 ? essabs(wb[m]) : nothing),
        "top1_share" => (tot > 0 ? maximum(wb[m]) / tot : nothing),
        "wmax" => (count(m) > 0 ? maximum(wb[m]) : nothing)))
end
res = Dict("seed" => seed, "K" => K, "N" => N, "Ntot" => Ntot, "n_reindexed" => length(idx),
    "gamma_min" => γmin, "gamma_checkpoints" => γ_chk, "chk1_equals_chk2" => (γ_chk[1] == γ_chk[2]),
    "secs_reindex" => secs, "control_max_abs_dlogq" => (isempty(ctrl) ? nothing : maximum(ctrl)),
    "wb_max" => maximum(wb), "wb_min" => minimum(wb),
    "fit_ess_reindex" => (sum(wb)^2 / (length(wb) * sum(abs2, wb))),
    "fit_ess_raw" => (sum(wown)^2 / (length(wown) * sum(abs2, wown))),
    "per_threshold" => rows)
open(joinpath(outdir, "seed$(seed).json"), "w") do io; JSON.print(io, res, 1); end
@printf("seed%-2d done %.0fs γmin=%.6f control=%.2e P(0.434)=%.8e P(0)=%.8e\n",
        seed, secs, γmin, res["control_max_abs_dlogq"], rows[1]["estimate"], rows[end]["estimate"])
