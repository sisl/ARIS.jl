function mcmc_target(system, threshold)
    (x) -> begin
        p = System.prior(system)
        s = System.simulate(system, x)
        r = System.evaluate(system, s)
        f = r .≤ threshold
        return pdf(p, x) * f
    end
end

mvn_kernel(system, x, σ) = MvNormal(x, σ^2*I(System.get_xdim(system)))

function mcmc(system, x; niter=10, σ=0.5, threshold=0.0)
    f = mcmc_target(system, threshold)
    q = mvn_kernel(system, x, σ)

    for i in 1:niter
        x′ = rand(q)
        fx = f(x)
        fx′ = f(x′)
        if fx == 0.0
            α = 0.0
        else
            α = min(1, fx′ / fx)
        end
        if rand() ≤ α
            x = x′
        end
    end

    return x
end

function ams(system; m=100, m_elite=10, k_max=100, nmcmc=10, σ=0.5)
    p = System.prior(system)
    xs = [rand(p) for i in 1:m]
    all_xs = [xs]


    p̂fail = 1.0
    for i in 1:k_max
        τs = [System.simulate(system, x) for x in xs]
        Y = [System.evaluate(system, τ) for τ in τs]
        order = sortperm(Y)
        γ = i == k_max ? 0 : max(0, Y[order[m_elite]])
        p̂fail *= mean(Y .≤ γ)
        γ == 0 && break
        xs = rand(xs[order[1:m_elite]], m)
        xs = [mcmc(system, x, niter=nmcmc, σ=σ, threshold=γ) for x in xs]
        push!(all_xs, xs)
    end
    return p̂fail, all_xs
end