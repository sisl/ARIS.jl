# Balance-heuristic deterministic-mixture MIS estimator.
#
#   P_balance = (1 / (K N)) Σ_n 1{ρ_n ≤ 0} p(x_n) / q_mix(x_n),
#   q_mix = (1 / K) Σ_j q_j
#
# where the first proposal is the prior and later proposals are the
# threshold-specific marginals of their fitted JointModels.
#
# EtaMixture proposals use the corresponding mixture marginal
# (1 - η) q_base + η p.

using ARIS
using Distributions, Statistics, LinearAlgebra

# Alias for the shared conditional-density evaluator.
const _cond_logpdf_at = conditional_logpdf_at

# Evaluate the JointModel marginal with the package's canonical density
# implementation. Grid-specific arguments are used only in grid mode.
function jm_marginal_pdf(jm::JointModel, γ::Real, X::AbstractMatrix;
                         nodes=nothing, r_lower=nothing, kw...)
    if MARGINAL_METHOD[] === :grid
        return marginal_pdf(jm, γ, X;
                            nodes = nodes === nothing ? MARGINAL_NODES[] : nodes,
                            lower = r_lower === nothing ? MARGINAL_LOWER[] : r_lower, kw...)
    end
    return marginal_pdf(jm, γ, X; kw...)
end

# proposal density q_j(X): dispatch on proposal type (JointModel or EtaMixture)
proposal_pdf(m::JointModel, γ, X, prior; kw...) = jm_marginal_pdf(m, γ, X; kw...)

"""
    balance_estimate(checkpoints, prior, X, ρ; nodes=200, r_lower=-10.0, N)

Compute the deterministic-mixture balance-MIS estimate from proposal
checkpoints and the corresponding pooled samples.

The first batch uses the prior; later batches use the stored proposal and
threshold from each checkpoint.
"""
function balance_estimate(checkpoints, prior, X, ρ; nodes=200, r_lower=-10.0, N)
    Ntot = length(ρ); K = Ntot ÷ N
    fmask = ρ .<= 0.0; fidx = findall(fmask)
    Xf = X[:, fidx]
    p_f = pdf(prior, Xf)
    qsum = copy(p_f)
    for k in 2:K
        model_k, γk = checkpoints[k]
        qsum .+= proposal_pdf(model_k, γk, Xf, prior; nodes=nodes, r_lower=r_lower)
    end
    q_mix = qsum ./ K
    w_bal = p_f ./ q_mix
    P_balance = sum(w_bal) / Ntot
    return (P_balance = P_balance, wmax = maximum(w_bal), nfail = length(fidx), K = K)
end
