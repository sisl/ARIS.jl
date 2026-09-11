# Canonical marginal input-density evaluator:
#
#   q_X^γ(x) = ∫_{-∞}^{γ} q_ψ(r | r ≤ γ) q_θ(x | r) dr
#
# All estimator-side density evaluations use `marginal_pdf` or
# `marginal_logpdf`, so amortized and per-threshold estimates share the
# same numerical marginal-density implementation.
#
# TRUNCATION CONVENTION.
# The sampler draws r ~ truncated(ψ; upper=γ), with support (-∞, γ], and the
# evaluator normalizes ψ over the same interval so that its density matches
# the sampler. Setting `MARGINAL_LOWER[]` to a finite value instead normalizes
# over [lower, γ].
#
# METHODS (`MARGINAL_METHOD[]`):
#   * `:batch` (default) — adaptive quadrature of a vector-valued integrand,
#     evaluating q_θ(·|r) across all samples at each integration node.
#   * `:adaptive` — separate adaptive quadrature for each sample.
#   * `:grid` — fixed-grid quadrature using `MARGINAL_NODES[]` nodes.
#
# For `:batch`, a coarse per-sample estimate scales the vector-valued
# integrand before quadrature. Using the L∞ norm makes the stopping criterion
# reflect per-sample relative accuracy, which is the relevant quantity for
# importance weights.

const MARGINAL_METHOD = Ref(:batch)     # :batch (canonical) | :adaptive | :grid
const MARGINAL_RTOL   = Ref(1e-3)       # quadgk relative tolerance (:batch / :adaptive)
const MARGINAL_LOWER  = Ref(-Inf)       # -Inf ⇒ (-∞, γ], identical to the sampler
const MARGINAL_NODES  = Ref(1000)       # only used by method=:grid
const MARGINAL_COARSE = 64              # nodes in the :batch scaling pre-pass
const MARGINAL_QTAIL  = Ref(1e-12)      # ψ-mass excluded from the numerical integration domain

# Numerical integration domain.
#
# The robustness density is normalized over (-∞, γ] to match the sampler.
# Numerically integrating to -∞ can make adaptive quadrature miss a sharp
# integrand peak near γ. We therefore integrate over
# [quantile(ψ_trunc, MARGINAL_QTAIL), γ], which retains all but approximately
# `MARGINAL_QTAIL` of the truncated robustness mass while keeping the same
# (-∞, γ] normalization.

_integration_lower(tψ, γ::Real, lower::Real) = begin
    isfinite(lower) && return Float64(lower)
    lo = try
        quantile(tψ, MARGINAL_QTAIL[])
    catch
        -Inf
    end
    (isfinite(lo) && lo < γ) || (lo = Float64(γ) - 40.0)
    Float64(lo)
end

"""
    conditional_logpdf_at(traj, r, X) -> Vector

log q_θ(x | r) for every column of `X` at a SINGLE conditioning value `r`, building the
conditional distribution once. Trajectory-model types add methods here; the fallback works for
any model supporting `logpdf(traj, X, rs)`.
"""
conditional_logpdf_at(traj, r::Real, X::AbstractMatrix) =
    logpdf(traj, X, fill(Float64(r), size(X, 2)))

function conditional_logpdf_at(traj::ConditionalEMGMM, r::Real, X::AbstractMatrix)
    condfun = conditional(traj.gmm, traj.condition_idx)
    return logpdf(condfun([Float64(r)]), X)
end

# Truncated robustness model using the canonical or an overridden finite lower bound.
_trunc_psi(rm, γ::Real, lower::Real) =
    isfinite(lower) ? truncated(rm, lower, Float64(γ)) : truncated(rm; upper = Float64(γ))

# Deterministic initial panel boundaries for adaptive quadrature.
#
# For failure samples, the integrand can be sharply concentrated near γ.
# On a wide interval, adaptive quadrature may otherwise under-resolve this
# near-boundary peak. We seed the integrator with breakpoints from two sources:
#   * quantiles of ψ, to resolve regions containing most of its mass
#   * geometrically spaced points below γ, to resolve the near-boundary peak
# This gives single-sample and batched evaluations the same initial panels.

function _breakpoints(tψ, rm, γ::Real, lo::Real)
    γf = Float64(γ)
    σ = try
        s = std(rm)
        (isfinite(s) && s > 0) ? Float64(s) : (γf - lo) / 8
    catch
        (γf - lo) / 8
    end
    pts = Float64[]
    for p in (1e-8, 1e-5, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
        v = try quantile(tψ, p) catch; NaN end
        isfinite(v) && lo < v < γf && push!(pts, Float64(v))
    end
    for c in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.03, 0.01)
        v = γf - c * σ
        lo < v < γf && push!(pts, v)
    end
    sort!(pts); unique!(pts)
    return [lo; pts; γf]
end

# integrand value vector at one node, with Inf/NaN guarded to 0 (underflowed mass)
function _integrand_at(traj, tψ, r::Real, X::AbstractMatrix)
    lp = conditional_logpdf_at(traj, r, X) .+ logpdf(tψ, Float64(r))
    return [isfinite(v) ? exp(v) : 0.0 for v in lp]
end

# coarse fixed-grid estimate used only to scale the batch integrand
function _coarse_scale(traj, tψ, γ::Real, X::AbstractMatrix, lo::Real)
    grid = range(lo, Float64(γ); length = MARGINAL_COARSE)
    h = length(grid) > 1 ? (grid[2] - grid[1]) : 1.0
    acc = zeros(size(X, 2))
    for r in grid
        acc .+= _integrand_at(traj, tψ, r, X) .* h
    end
    return [(isfinite(a) && a > 0) ? a : 1.0 for a in acc]
end

"""
    marginal_pdf(jm::JointModel, γ, X::AbstractMatrix; kwargs...) -> Vector{Float64}

Canonical q_X^γ(x) for every column of `X`. Keyword defaults come from the `MARGINAL_*` refs so a
single setting governs every estimator-side call site.
"""
function marginal_pdf(jm::JointModel, γ::Real, X::AbstractMatrix;
                      method::Symbol = MARGINAL_METHOD[],
                      rtol::Real     = MARGINAL_RTOL[],
                      lower::Real    = MARGINAL_LOWER[],
                      nodes::Int     = MARGINAL_NODES[])
    traj = jm.trajectory_model
    tψ   = _trunc_psi(jm.robustness_model, γ, lower)
    n    = size(X, 2)
    n == 0 && return Float64[]

    if method === :grid
        lo   = _integration_lower(tψ, γ, lower)
        grid = range(lo, Float64(γ); length = nodes)
        ψw   = [pdf(tψ, r) for r in grid]
        s    = sum(ψw)
        s > 0 ? (ψw ./= s) : (ψw .= 1 / length(ψw))
        acc = zeros(n)
        for (i, r) in enumerate(grid)
            acc .+= exp.(conditional_logpdf_at(traj, r, X)) .* ψw[i]
        end
        return acc

    elseif method === :adaptive
        lo  = _integration_lower(tψ, γ, lower)
        bps = _breakpoints(tψ, jm.robustness_model, γ, lo)
        out = Vector{Float64}(undef, n)
        for j in 1:n
            Xj = view(X, :, j:j)
            I, _ = quadgk(r -> _integrand_at(traj, tψ, r, Xj)[1], bps...; rtol = rtol)
            out[j] = max(I, 0.0)
        end
        return out

    elseif method === :batch
        lo    = _integration_lower(tψ, γ, lower)
        bps   = _breakpoints(tψ, jm.robustness_model, γ, lo)
        scale = _coarse_scale(traj, tψ, γ, X, lo)
        I, _ = quadgk(r -> _integrand_at(traj, tψ, r, X) ./ scale, bps...;
                      rtol = rtol, norm = v -> maximum(abs, v))
        return max.(I .* scale, 0.0)
    else
        error("marginal_pdf: unknown method $(method); expected :batch, :adaptive or :grid")
    end
end

marginal_pdf(jm::JointModel, γ::Real, x::AbstractVector; kwargs...) =
    marginal_pdf(jm, γ, reshape(collect(Float64, x), :, 1); kwargs...)

"""
    marginal_logpdf(jm::JointModel, x, γ) -> Real

log of the canonical marginal density. This is what `logpdf(::JointModel, x, γ)` delegates to.
"""
function marginal_logpdf(jm::JointModel, x::AbstractVector, γ::Real; kwargs...)
    v = marginal_pdf(jm, γ, reshape(collect(Float64, x), :, 1); kwargs...)[1]
    return v > 0 ? log(v) : -Inf
end

"""
    marginal_reference(jm, γ, X) -> Vector{Float64}

High-accuracy reference integral (per-sample adaptive quadgk at rtol=1e-10) used by the validation
probes to measure the canonical evaluator's relative error. Too slow for production use.
"""
marginal_reference(jm::JointModel, γ::Real, X::AbstractMatrix; rtol::Real = 1e-10) =
    marginal_pdf(jm, γ, X; method = :adaptive, rtol = rtol)

"""
    marginal_settings() -> Dict

The active canonical settings, for recording in run provenance.
"""
marginal_settings() = Dict{String,Any}(
    "method"          => String(MARGINAL_METHOD[]),
    "rtol"            => MARGINAL_RTOL[],
    "lower"           => isfinite(MARGINAL_LOWER[]) ? MARGINAL_LOWER[] : "-Inf",
    "nodes_if_grid"   => MARGINAL_NODES[],
    "qtail"           => MARGINAL_QTAIL[],
    "domain_rule"     => "psi normalised over (-Inf, gamma]; integrated over " *
                         "[quantile(psi_trunc, qtail), gamma]",
    "truncation"      => isfinite(MARGINAL_LOWER[]) ?
                         "[$(MARGINAL_LOWER[]), gamma]" : "(-Inf, gamma] (matches sampler)",
    "formula"         => "q_X^g(x) = int_{-inf}^{g} q_psi(r | r<=g) q_theta(x|r) dr",
)
