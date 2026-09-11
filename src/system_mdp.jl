# Conditional risk estimation MDP
@with_kw mutable struct RMDP <: MDP{Vector, Vector}
    sys::System.SystemParameters
    cost_fn
    added_states = :none # :none, :time, :acc_reward, :time_and_acc_reward
    dt = 0.1
    maxT = 10.0
    disturbance_type=:arg #:arg if passed as argument, :noise if used as noise on the state, :action_noise for noise on action
    xscale = 1f0
    rshift = 0f0
    rscale = 1f0
    c = 0f0
end

function POMDPs.initialstate(mdp::RMDP; c=0f0)
    s0 = []
    if mdp.sys isa System.SystemParameters
        s0 = rand(System.initialstate_prior(mdp.sys))
    else
        s0 = rand(initialstate(mdp.sys))
    end
    mdp.c = c
    if mdp.added_states in [:time, :acc_reward]
        return ImplicitDistribution((rng) -> [mdp.c, 0f0, s0...])
    elseif mdp.added_states == :none
        return ImplicitDistribution((rng)->s0)
    else
        @error "unrecognized added state: $(mdp.added_states)"
    end
end

function get_s(mdp::RMDP, s)
    if mdp.added_states in [:time, :acc_reward]
        return s[3:end]
    else
        return s
    end
end
        
function POMDPs.isterminal(mdp::RMDP, s)
    isterm = false
    if mdp.sys isa System.SystemParameters
        isterm = System.isterminal(mdp.sys, get_s(mdp, s))
    else
        isterm = isterminal(mdp.sys, get_s(mdp, s))
    end
    if mdp.added_states in [:time, :time_and_acc_reward]
        isterm = isterm || (s[2] > (mdp.maxT + mdp.dt/2))
    end
    isterm
end

function isfailure(mdp::RMDP, s)
    isfailure(mdp.sys, get_s(mdp,s))
end
    
POMDPs.discount(mdp::RMDP) = 1f0

function POMDPs.gen(mdp::RMDP, s, x, rng::AbstractRNG = Random.GLOBAL_RNG; kwargs...)
    x = x .* mdp.xscale
    if mdp.sys isa MDP
        if mdp.disturbance_type == :arg
            sp, r = gen(mdp.sys, get_s(mdp, s), action(mdp.π, get_s(mdp, s)), x, rng; kwargs...)
        elseif mdp.disturbance_type == :noise
            sp, r = gen(mdp.sys, get_s(mdp, s), action(mdp.π, get_s(mdp, s) .+ x), rng; kwargs...)
        elseif mdp.disturbance_type == :action_noise
            sp, r = gen(mdp.sys, get_s(mdp, s), action(mdp.π, get_s(mdp, s)) .+ x, rng; kwargs...)
        elseif mdp.disturbance_type == :both
            sp, r = gen(mdp.sys, get_s(mdp, s), action(mdp.π, get_s(mdp, s) .+ x[3:end]), x[2], rng; kwargs...)
        else
            @error "Unrecognized disturbance type $(mdp.disturbance_type)"
        end

    elseif mdp.sys isa System.SystemParameters
        sp = System.step(mdp.sys, get_s(mdp, s), x)
    end
    
    if mdp.added_states == :time
        sp = [s[1], s[2]+mdp.dt, sp...]
    elseif mdp.added_states == :acc_reward
        sp = [s[1], s[2] + r, sp...]
    end
    (sp=sp, r=mdp.cost_fn(mdp, s, sp))
end

POMDPs.reward(mdp::RMDP, s, sp) = mdp.cost_fn(mdp, s, sp)
