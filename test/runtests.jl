using Test
using ARIS
using Distributions, LinearAlgebra, Random, Statistics

const EXAMPLES = joinpath(@__DIR__, "..", "examples")

# Unconditional proposal that counts its fits; used to check CrossEntropyMethod's update timing
# without depending on EM.
struct FitCounter
    μ::Vector{Float64}
    nfit::Int
end
Base.rand(rng::AbstractRNG, m::FitCounter, n::Int) = m.μ .+ randn(rng, length(m.μ), n)
Distributions.logpdf(m::FitCounter, X::AbstractMatrix) = logpdf(MvNormal(m.μ, I), X)
Distributions.fit(m::FitCounter, X::AbstractMatrix, w::AbstractVector) =
    FitCounter(vec(X * w) ./ sum(w), m.nfit + 1)

@testset "ARIS" begin

    @testset "public API" begin
        for name in (:System, :SystemParameters, :Buffer, :ConditionalValidation, :train!,
                     :JointModel, :TruncatedJointModel, :ConditionalEMGMM, :EMGMM,
                     :GaussianMixture, :ConditionalGaussian, :QuantileSampling,
                     :TargetSampling, :CrossEntropySampling, :ams, :pmc,
                     :CrossEntropyMethod, :ess, :is_estimate, :marginal_pdf)
            @test name in names(ARIS)
        end
    end

    @testset "utils" begin
        @test ess(ones(10)) ≈ 10
        @test ess([1.0, 0.0, 0.0, 0.0]) ≈ 1

        m = [1.0, 2.0, 3.0]
        Σ = [2.0 0.3 0.0; 0.3 1.0 0.2; 0.0 0.2 1.5]
        d = MvNormal(m, Σ)
        @test mean(marginal(d, [1, 3])) ≈ m[[1, 3]]
        @test cov(marginal(d, [1, 3])) ≈ Σ[[1, 3], [1, 3]]
        c = conditional(d, [3], [3.0])
        @test length(c) == 2
        @test mean(c) ≈ m[1:2]        # conditioning at the mean leaves the mean unmoved

        A = reshape(collect(1.0:12.0), 3, 4)
        @test bslice(A, 2) == A[:, 2]
    end

    @testset "example problems load" begin
        for p in ("toy.jl", "radial_shell.jl", "tailrace.jl", "crosswalk.jl")
            @test isfile(joinpath(EXAMPLES, "problems", p))
        end
        include(joinpath(EXAMPLES, "problems", "toy.jl"))
        sys = UnimodalToy(3.0)
        @test System.get_xdim(sys) == 2
        @test System.evaluate(sys, [0.0, 0.0]) ≈ 3.0
        @test System.prior(sys) isa MvNormal
    end

    # End-to-end smoke test: a deliberately tiny adaptive run. This checks that the
    # fit / sample / reweight loop executes, not that the estimate is accurate.
    @testset "train! smoke" begin
        Random.seed!(1)
        sys = UnimodalToy(3.0)
        model = JointModel(Normal(), ConditionalEMGMM(2, 3, 3))
        cv = ConditionalValidation(model = model, xdim = 2, depth = 1, sdim = 2,
                                   n_samples = 100, n_iter = 3,
                                   sampling_strategy = QuantileSampling(0.2; target = 1.0))
        train!(cv, sys)
        @test length(cv.buffer[:ρ]) == 300
        @test all(isfinite, cv.buffer[:w])
        @test all(≥(0), cv.buffer[:w])
        p̂ = is_estimate(cv.buffer; threshold = 1.0)
        @test 0 < p̂ < 1
    end

    @testset "ConditionalGaussianCEM" begin
        cv = ConditionalGaussianCEM(xdim = 2, depth = 1, sdim = 2, n_samples = 100, n_iter = 3)
        @test cv.model isa ConditionalGaussian
        @test cv.𝒫.dim == 2
    end

    # The prior batch's elites must be fit before batch 2 is drawn (batch 2 previously came from
    # the untouched initial model), without changing the number of batches.
    @testset "CrossEntropyMethod fits the prior batch" begin
        Random.seed!(5)
        N, K = 200, 3
        m0 = FitCounter(zeros(2), 0)
        seen = FitCounter[]
        cem = CrossEntropyMethod(model = m0, xdim = 2, depth = 1, sdim = 2,
                                 n_samples = N, n_iter = K, n_elite = 20,
                                 log_callback = (cv, D) -> push!(seen, cv.model))
        train!(cem, UnimodalToy(3.0))
        @test length(cem.buffer[:ρ]) == N * K           # K batches of N evaluations
        @test seen[2].nfit == 1                         # batch 2's proposal: fit once, on the prior batch
        X2 = Float64.(cem.buffer[:x][:, N+1:2N])
        lq2 = Float64.(cem.buffer[:logpdf][N+1:2N])
        @test lq2 ≈ logpdf(seen[2], X2) rtol = 1e-4     # batch 2 was drawn from that fitted proposal
        @test !isapprox(lq2, logpdf(m0, X2); rtol = 1e-2)  # and not from the initial model
        @test cem.model.nfit == K                       # one fit per batch
    end

    @testset "ConditionalGaussian fit" begin
        Random.seed!(3)
        X = randn(2, 200); w = rand(200) .+ 0.5
        for r in (randn(200), zeros(200))          # correlated-r path and constant-r fallback
            q = fit(ConditionalGaussian(2), X, r, w)
            @test q isa ConditionalGaussian
            @test length(q.μ1) == 2 && size(q.Σ11) == (2, 2) && size(q.Σ12) == (2, 1)
            @test all(isfinite, q.μ1) && isposdef(Symmetric(q.Σ11))
            xs = rand(Random.default_rng(), q, [0.0, 0.5, 1.0])
            @test size(xs) == (2, 3)
            @test all(isfinite, logpdf(q, xs, [0.0, 0.5, 1.0]))
        end
    end

    @testset "fit_mle dispatch bridges" begin
        Random.seed!(4)
        X = randn(3, 300); w = rand(300) .+ 0.1
        same(a, b) = typeof(a) == typeof(b) && mean(a) == mean(b) && cov(a) == cov(b)
        fallback = Tuple{Type{D}, AbstractMatrix{<:Real}, AbstractVector{<:Real}} where {D<:MvNormal}
        for T in (FullNormal, DiagNormal, IsoNormal), W in (Float64, Float32, Int)
            @test which(fit_mle, Tuple{Type{T}, Matrix{Float64}, Vector{W}}).module === ARIS
        end
        for ww in (w, Float32.(w), ones(Int, 300))
            @test same(fit_mle(FullNormal, X, ww), invoke(fit_mle, fallback, FullNormal, X, ww))
        end
        # Distributions' diagonal/isotropic estimators accept Float64 weights only (other weight types
        # fail inside Distributions, with or without ARIS), so they are checked with Float64 weights.
        for T in (DiagNormal, IsoNormal)
            d = fit_mle(T, X, w)
            @test d isa T
            @test same(d, invoke(fit_mle, Tuple{Type{T}, AbstractMatrix{Float64}, AbstractVector}, T, X, w))
        end
        @test isdiag(cov(fit_mle(DiagNormal, X, w)))
        c = cov(fit_mle(IsoNormal, X, w))
        @test isdiag(c) && allequal(diag(c))

        # EM's M-step refits each component on its own type; Diag/Iso components must come back as such.
        for comps in ([MvNormal(randn(3), 1.0I) for _ in 1:2],
                      [MvNormal(randn(3), Diagonal(ones(3))) for _ in 1:2])
            m = fit_mle(MixtureModel(comps), X, w; atol = 1e-3, robust = true, infos = false)
            @test eltype(m.components) == eltype(comps)
        end
    end
end
