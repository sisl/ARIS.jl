# Peaks-over-threshold GPD extrapolation for the lower tail of robustness.
#
# For an anchor threshold u, fit a weighted generalized Pareto distribution
# to exceedances e = u - ρ for samples with ρ ≤ u. For γ′ ≤ u,
#
#   P(ρ ≤ γ′) = P(ρ ≤ u) * (1 + ξ (u - γ′) / σ)^(-1 / ξ)
#
# with the exponential limit used as ξ → 0.
#
# This estimator assumes the tail composition is stable below u; substantial
# mode switching can invalidate the extrapolation.

using Distributions, Statistics, Optim

# Weighted GPD negative log-likelihood in (ξ, log σ).
function _gpd_wnll(params, e, w)
    ξ, logσ = params[1], params[2]; σ = exp(logσ)
    sw = sum(w); ll = 0.0
    for i in eachindex(e)
        z = e[i] / σ
        if abs(ξ) < 1e-6
            li = -logσ - z
        else
            base = 1 + ξ * z
            base <= 0 && return Inf                       # support violation
            li = -logσ - (1 + 1/ξ) * log(base)
        end
        ll += w[i] * li
    end
    return -ll / sw
end

# Fit a weighted GPD and report the effective exceedance count.
function fit_gpd_weighted(e, w)
    isempty(e) && return (ξ=NaN, σ=NaN, n_eff=0.0)
    σ0 = max(1e-3, mean(e))
    res = optimize(p -> _gpd_wnll(p, e, w), [0.1, log(σ0)], NelderMead())
    ξ, σ = Optim.minimizer(res)[1], exp(Optim.minimizer(res)[2])
    n_eff = sum(w)^2 / sum(w .^ 2)
    return (ξ=ξ, σ=σ, n_eff=n_eff)
end

# Sensitivity fit constrained to ξ ≥ 0, which excludes bounded-tail solutions.
function fit_gpd_weighted_xi_nonneg(e, w)
    isempty(e) && return (ξ=NaN, σ=NaN, n_eff=0.0)
    σ0 = max(1e-3, mean(e))
    res = optimize(p -> _gpd_wnll([max(p[1], 0.0), p[2]], e, w), [0.1, log(σ0)], NelderMead())
    ξ, σ = max(Optim.minimizer(res)[1], 0.0), exp(Optim.minimizer(res)[2])
    n_eff = sum(w)^2 / sum(w .^ 2)
    return (ξ=ξ, σ=σ, n_eff=n_eff)
end

# Extrapolate P(ρ ≤ γ′) from P(ρ ≤ u) and a fitted GPD.
function evt_extrapolate(P_u, u, γ′, ξ, σ)
    y = u - γ′
    y <= 0 && return P_u
    tail = abs(ξ) < 1e-6 ? exp(-y / σ) : (1 + ξ * y / σ)^(-1 / ξ)
    return P_u * tail
end

# Fit the tail model and extrapolate to γ′.
function evt_estimate(ρ, w, u, P_u, γ′; xi_nonneg::Bool=false)
    mask = ρ .<= u
    e = u .- ρ[mask]; we = w[mask]
    g = xi_nonneg ? fit_gpd_weighted_xi_nonneg(e, we) : fit_gpd_weighted(e, we)
    est = evt_extrapolate(P_u, u, γ′, g.ξ, g.σ)
    return (est=est, n_eff=g.n_eff, ξ=g.ξ, σ=g.σ)
end
