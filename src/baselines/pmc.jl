function pmc(system; N::Int=8, T::Int=10, σ::Real=1.0, dₓ::Int=2, use_dm=false)
	p = System.prior(system)
    μ = [rand(p) for _ in 1:N]
	Σ = σ^2*I(dₓ)
	S = []
	for t in 1:T
		qₜ = [MvNormal(μ[n], Σ) for n in 1:N]
		xₜ = [rand(qₜ[n]) for n in 1:N]
		if use_dm # Deterministic mixture multiple importance sampling (DM-MIS)
			wₜ = [pdf(p, xₜ[n]) / mean(pdf(qₜ[j], xₜ[n]) for j in 1:N) for n in 1:N]
		else # Standard multimple importance sampling (s-MIS)
			wₜ = [pdf(p, xₜ[n]) / pdf(qₜ[n], xₜ[n]) for n in 1:N]
		end
        ss = [System.simulate(system , x) for x in xₜ]
        rs = [System.evaluate(system, s) for s in ss]
        fs = rs .<= 0.0
        wₜ = wₜ .* fs

        # If all weights are zero, set them to 1/N
        if sum(wₜ) == 0
            wₜ .= 1/N
        end

		push!(S, (xₜ, rs, wₜ))
		μ = StatsBase.sample(xₜ, Weights(wₜ), N, replace=true)
	end
	samples = reduce(vcat, first.(S))
	rvals = reduce(vcat, getindex.(S, 2))
	weights = reduce(vcat, last.(S))
	return μ, Σ, (xs=samples, rs=rvals, ws=weights)
end

snis(g, X, W) = sum(g(X[k]) * W[k] for k in eachindex(X)) / sum(W)
stderr(X) = std(X) / sqrt(length(X))

PMC_DEFAULTS = (T_max=100, N_seeds=3, N_qs=50)