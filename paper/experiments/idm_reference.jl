# Threaded reference Monte Carlo for the 150-dimensional IDM crosswalk benchmark.
# Usage:
#   julia -t T --project=paper/envs/idm paper/experiments/idm_reference.jl [N_total] [chunk]
#
# Accumulates a fine robustness histogram for estimating the reference failure-probability
# curve. Progress is written incrementally so long runs can be resumed.

using ARIS
using Random, Statistics, LinearAlgebra, Printf, POMDPs, JSON
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "..", "..", "examples", "problems", "crosswalk.jl"))

const N_TOTAL = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20_000_000
const CHUNK   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1_000_000
const ASCALE  = 0.60;  const FACTOR = ASCALE / 0.25
const K_DIST  = 30;    const NSUB = 4;  const DT_DIST = 0.2      # dt_phys = 0.05 (collision-faithful)
# Fine robustness bins over [-0.02, 1.0] with width 5e-4.
const BLO = -0.02; const BW = 5e-4; const NB = Int(round((1.0 - BLO)/BW))
bin(ρ) = clamp(Int(floor((ρ - BLO)/BW)) + 1, 1, NB)

const OUT = joinpath(@__DIR__, "..", "results", "reference")
mkpath(OUT); const HF = joinpath(OUT, "idm_hist.json")

# Sample from the prior and evaluate with the shared crosswalk rollout.
one_rollout(sys) = crosswalk_rollout(sys, reduce(hcat, (rand(sys.px) for _ in 1:K_DIST)); want_traj=false)

# Resume from an existing histogram if available.
counts = zeros(Int, NB); ndone = 0
if isfile(HF)
    d = JSON.parsefile(HF); counts = Int.(d["counts"]); ndone = d["ndone"]
    @printf("RESUMING: %d already banked\n", ndone)
end
const NT = Threads.maxthreadid()   # Julia 1.12: threadid() can exceed nthreads() (interactive pool)
const SYSP = [deepcopy(AdversarialCrosswalk(; dt=DT_DIST/NSUB)) for _ in 1:NT]  # per-thread MDPs
# Separate simulator state per thread because rollouts mutate system state.

function persist(counts, ndone)
    tmp = HF * ".tmp"
    open(tmp, "w") do io
        JSON.print(io, Dict("counts"=>counts, "ndone"=>ndone, "blo"=>BLO, "bw"=>BW, "nb"=>NB,
                            "ascale"=>ASCALE, "d"=>5*K_DIST, "nsub"=>NSUB))
    end
    mv(tmp, HF; force=true)
end

@printf("REFERENCE MC: N=%d threads=%d ascale=%.2f d=%d dt_phys=%.3f  (chunk=%d)\n",
        N_TOTAL, nthreads(), ASCALE, 5*K_DIST, DT_DIST/NSUB, CHUNK)
t0 = time()
while ndone < N_TOTAL
    global ndone, counts
    m = min(CHUNK, N_TOTAL - ndone)
    tloc = [zeros(Int, NB) for _ in 1:NT]
    @threads for i in 1:m
        ρ = one_rollout(SYSP[threadid()])
        tloc[threadid()][bin(ρ)] += 1
    end
    for tl in tloc; counts .+= tl; end
    ndone += m
    persist(counts, ndone)
    ncoll = sum(counts[1:bin(0.0)])
    @printf("  banked %d/%d  (%.0f roll/s)  collisions=%d (P=%.2e)\n",
            ndone, N_TOTAL, ndone/(time()-t0), ncoll, ncoll/ndone)
end

# Report the reference curve with Wilson 95% confidence intervals.
wilson(k,n) = (p=k/n; z=1.96; d=1+z^2/n; c=(p+z^2/(2n))/d; h=z*sqrt(p*(1-p)/n+z^2/(4n^2))/d; (max(0,c-h), c+h))

# Exact threshold P(ρ ≤ γ). bin(γ) is the bin CONTAINING γ, i.e. [γ, γ+bw) when γ is a bin lower
# edge, so the count sums bins 1:bin(γ)-1 and excludes that bin.
cum(γ) = sum(counts[1:bin(γ)-1])
println("\nTRUTH CURVE  P(ρ≤γ′)  [Wilson 95% CI]  — γ′ from near-miss band to collision atom:")
for γ in (0.434, 0.30, 0.20, 0.10, 0.05, 0.02, 0.0)   # sep 4m … 0.45m(collision)
    k = cum(γ); lo, hi = wilson(k, ndone)
    @printf("  γ′=%.3f  P=%.3e  [%.3e, %.3e]  (n=%d)\n", γ, k/ndone, lo, hi, k)
end
println("IDM_REFERENCE_DONE")
