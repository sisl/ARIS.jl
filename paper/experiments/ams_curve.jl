# AMS curve estimator with explicit simulator-evaluation accounting.
#
# Each level contributes its conditional-probability estimate to the cumulative
# failure-probability curve. Requested thresholds use an exact level when
# available and otherwise log-linear interpolation between bracketing levels.
#
# `ams_curve` uses a fixed mutation scale; `ams_curve_rwm` uses a
# dimension-scaled random-walk-Metropolis scale.

using ARIS
using Distributions, Random, LinearAlgebra

# Counts all robustness evaluations, including level assessments and MCMC moves.
function ams_curve(system; m=200, m_elite=20, k_max=200, nmcmc=10, σ=0.5)
    p = System.prior(system)
    d = System.get_xdim(system)
    evalcount = Ref(0)
    ρ1(x) = (evalcount[] += 1; System.evaluate(system, System.simulate(system, x)))

    function mcmc_step(x, threshold)
        q = MvNormal(x, σ^2 * I(d))
        for _ in 1:nmcmc
            x′ = rand(q)
            fx  = (ρ1(x)  <= threshold) ? pdf(p, x)  : 0.0
            fx′ = (ρ1(x′) <= threshold) ? pdf(p, x′) : 0.0
            α = fx == 0.0 ? 0.0 : min(1.0, fx′ / fx)
            rand() <= α && (x = x′; q = MvNormal(x, σ^2 * I(d)))
        end
        return x
    end

    xs = [rand(p) for _ in 1:m]
    P̂ = 1.0
    levels = Tuple{Float64,Float64}[]
    for i in 1:k_max
        Y = [ρ1(x) for x in xs]                        # m evals
        order = sortperm(Y)
        γ = i == k_max ? 0.0 : max(0.0, Y[order[m_elite]])
        P̂ *= mean(Y .<= γ)
        push!(levels, (γ, P̂))                          # (level threshold, curve value there)
        γ == 0.0 && break
        xs = rand(xs[order[1:m_elite]], m)
        xs = [mcmc_step(x, γ) for x in xs]             # m·nmcmc·2 evals
    end
    return levels, evalcount[]
end

# Evaluate the AMS curve at γ′ using an exact level or log-linear interpolation.
# Returns `nothing` when γ′ lies outside the reached level range.
function ams_at(levels, γ′)
    γs = [l[1] for l in levels]; Ps = [l[2] for l in levels]
    # exact level hit?
    for (γ, P) in levels
        isapprox(γ, γ′; atol=1e-9) && return (P, false)
    end
    γmin, γmax = minimum(γs), maximum(γs)
    (γ′ < γmin || γ′ > γmax) && return (nothing, false)   # outside AMS's reached range
    # linear interpolation in (γ, log P) between bracketing levels (log because P spans decades)
    o = sortperm(γs); γo = γs[o]; lPo = log.(Ps[o])
    j = searchsortedfirst(γo, γ′)
    j <= 1 && return (Ps[o[1]], true)
    γa, γb = γo[j-1], γo[j]; la, lb = lPo[j-1], lPo[j]
    t = (γ′ - γa) / (γb - γa)
    return (exp(la + t * (lb - la)), true)
end

# Dimension-scaled AMS variant.
#
# A fixed per-coordinate mutation scale produces larger overall proposal
# displacements as dimension grows. `rwm_sigma(d) = 2.38 / sqrt(d)` applies
# the standard dimension-scaled random-walk-Metropolis rule.
#
# Interpolation returns explicit status tags so callers can distinguish exact,
# interpolated, degenerate, and out-of-range estimates.

"Standard dimension-scaled random-walk-Metropolis scale. Depends only on the input dimension."
rwm_sigma(d::Int) = 2.38 / sqrt(d)

# returns (levels, n_evals, diag). Cost accounting as in ams_curve: per level m + m*nmcmc*2.
function ams_curve_rwm(system; m=200, m_elite=20, k_max=200, nmcmc=10, sigma=nothing)
    p = System.prior(system)
    d = System.get_xdim(system)
    sigma_use = sigma === nothing ? rwm_sigma(d) : sigma          # dimension-scaled default
    evalcount = Ref(0)
    rho1(x) = (evalcount[] += 1; System.evaluate(system, System.simulate(system, x)))

    att = Ref(0); acc = Ref(0)                                    # diagnostics; no extra evaluations
    function mcmc_step_r(x, threshold)
        q = MvNormal(x, sigma_use^2 * I(d))
        for _ in 1:nmcmc
            x2 = rand(q)
            fx  = (rho1(x)  <= threshold) ? pdf(p, x)  : 0.0
            fx2 = (rho1(x2) <= threshold) ? pdf(p, x2) : 0.0
            alpha = fx == 0.0 ? 0.0 : min(1.0, fx2 / fx)
            att[] += 1
            rand() <= alpha && (x = x2; q = MvNormal(x, sigma_use^2 * I(d)); acc[] += 1)
        end
        return x
    end

    xs = [rand(p) for _ in 1:m]
    Phat = 1.0
    levels = Tuple{Float64,Float64}[]
    nuniq = Int[]
    for i in 1:k_max
        Y = [rho1(x) for x in xs]
        push!(nuniq, length(unique(Y)))
        order = sortperm(Y)
        gam = i == k_max ? 0.0 : max(0.0, Y[order[m_elite]])
        Phat *= mean(Y .<= gam)
        push!(levels, (gam, Phat))
        gam == 0.0 && break
        xs = rand(xs[order[1:m_elite]], m)
        xs = [mcmc_step_r(x, gam) for x in xs]
    end
    diag = Dict{String,Any}(
        "sigma_used"=>sigma_use, "sigma_rule"=>(sigma === nothing ? "2.38/sqrt(d)" : "explicit"),
        "d"=>d, "proposal_displacement"=>sigma_use * sqrt(d),
        "mutations_attempted"=>att[], "mutations_accepted"=>acc[],
        "acceptance_rate"=>(att[] == 0 ? nothing : acc[] / att[]),
        "n_levels"=>length(levels), "unique_rho_per_level"=>nuniq,
        "m"=>m, "m_elite"=>m_elite, "nmcmc"=>nmcmc,
        "mean_unique_fraction"=>(isempty(nuniq) ? nothing : sum(nuniq) / (length(nuniq) * m)))
    return levels, evalcount[], diag
end

# Returns `(estimate, status)`, where status is one of:
# `:exact_level`, `:interpolated`, `:out_of_range`, or `:degenerate`.
function ams_at_tagged(levels, gam_req)
    gs = [l[1] for l in levels]; Ps = [l[2] for l in levels]
    for (g, P) in levels
        isapprox(g, gam_req; atol=1e-9) && return (P, :exact_level)
    end
    gmin, gmax = minimum(gs), maximum(gs)
    (gam_req < gmin || gam_req > gmax) && return (nothing, :out_of_range)
    o = sortperm(gs); go = gs[o]; Po = Ps[o]
    j = searchsortedfirst(go, gam_req)
    j <= 1 && return (Po[1], :exact_level)
    ga, gb = go[j-1], go[j]; Pa, Pb = Po[j-1], Po[j]
    (Pa > 0 && Pb > 0 && isfinite(Pa) && isfinite(Pb)) || return (nothing, :degenerate)
    t = (gam_req - ga) / (gb - ga)
    v = exp(log(Pa) + t * (log(Pb) - log(Pa)))
    return (isfinite(v) ? v : nothing, :interpolated)
end
