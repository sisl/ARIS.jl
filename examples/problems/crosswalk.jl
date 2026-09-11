using AdversarialDriving, AutomotiveSimulator, AutomotiveVisualization
using Distributions, POMDPs, Crux, Random, POMDPTools

struct PointOfClosestApproachMDP{T, A} <: MDP{T, A}
    mdp
    PointOfClosestApproachMDP(mdp::MDP{S,A}) where {S,A} = new{Tuple{Float32, S},A}(mdp)
end

POMDPs.discount(mdp::PointOfClosestApproachMDP) = 1f0

function POMDPs.initialstate(mdp::PointOfClosestApproachMDP{Tuple{Float32, S}, A}, rng::AbstractRNG = Random.GLOBAL_RNG) where {S, A}
    ImplicitDistribution((rng) -> (0f0, rand(rng, initialstate(mdp.mdp))))
end

function POMDPs.convert_s(::Type{AbstractArray}, s, mdp::PointOfClosestApproachMDP{T, A}) where {T,A}
    Float32.([s[1], convert_s(AbstractArray, s[2], mdp.mdp)...])
end

function POMDPs.gen(mdp::PointOfClosestApproachMDP{Tuple{Float32, S}, A}, s::Tuple{Float32, S}, a, rng::Random.AbstractRNG = Random.GLOBAL_RNG) where {S,A}
    result = step(mdp.mdp, s[2], a, rng)
    sp, r = result.sp, result.r
    biggest_reward = Float32(max(s[1], r))
    if isterminal(mdp.mdp, sp)
        return (;sp=(biggest_reward, sp), r=biggest_reward)
    else
        return (;sp=(biggest_reward, sp), r=0f0)
    end
end

POMDPs.isterminal(mdp::PointOfClosestApproachMDP, s) = isterminal(mdp.mdp, s[2])

struct AdvDrivingAction
    a
end

Base.iterate(v::AdvDrivingAction) = (v, nothing)
Base.iterate(v::AdvDrivingAction, n::Nothing) = nothing

function POMDPs.gen(mdp::AdversarialDrivingMDP, s::Scene, a::Vector{AbstractFloat}, rng::Random.AbstractRNG = Random.GLOBAL_RNG)# where A
    ascale=0.25
    a = a .* ascale
    pc = PedestrianControl(da = VecE2(a[1], a[2]), noise=Noise(VecE2(a[3], a[4]), a[5]))
    return gen(mdp, s, Disturbance[pc], rng)
end

function step(mdp::AdversarialDrivingMDP, s::Scene, a::Vector{<:AbstractFloat}, rng::Random.AbstractRNG = Random.GLOBAL_RNG)# where A
    ascale=0.25
    a = a .* ascale
    pc = PedestrianControl(da = VecE2(a[1], a[2]), noise=Noise(VecE2(a[3], a[4]), a[5]))
    return gen(mdp, s, Disturbance[pc], rng)
end

function POMDPs.initialstate(mdp::MDP{Scene, A}, rng::AbstractRNG = Random.GLOBAL_RNG) where A
    # Generate challengin scenario
    veh_target = 18.
    veh_start = 0.
    
    ped_target = 8.


    collision_time = 2.
    
    veh_v = 8. #(veh_target - veh_start) / (collision_time)
    ped_v = 1.5

    ped_s = 5. #ped_target - ped_v*collision_time
    
    veh_s = veh_start # vehicle is slowing down
    
    s = Scene([ez_pedestrian(;id=2, s=ped_s, v=ped_v), ez_ped_vehicle(;id=1, s=veh_s, v=veh_v)])
    ImplicitDistribution((rng)->s)
end

function POMDPs.reward(mdp::AdversarialDrivingMDP, s::Scene, a::Vector{Disturbance}, sp::Scene)
    iscollision = length(sp) > 0 && ego_collides(sutid(mdp), sp)
    iscollision ? 1f0 : 20f0 / (Float32(AdversarialDriving.min_dist(s, sutid(mdp)))^2 + 20f0)
end

function POMDPs.isterminal(mdp::MDP{Scene, A}, s::Scene) where A
    isterm_orig = !(sutid(mdp) in s)|| any_collides(s)
    isterm_orig || posf(get_by_id(s, 1)).s > 35 || posf(get_by_id(s, 2)).s > 13
end

function gen_crosswalk_problem(; dt=0.2)   # dt parameterized for collision-faithful sub-stepping
    ## Construct the MDP
    sut_agent = BlinkerVehicleAgent(get_ped_vehicle(id=1, s=0., v=0.), TIDM(ped_TIDM_template, noisy_observations = true))
    adv_ped = NoisyPedestrianAgent(get_pedestrian(id=2, s=0., v=0.), AdversarialPedestrian())
    mdp = AdversarialDrivingMDP(sut_agent, [adv_ped], ped_roadway, dt)
    mdp.agents[end].model.idm.v_des=10
    
    px = product_distribution([Normal(0, 1) for _=1:5])

    px, PointOfClosestApproachMDP(mdp)
end

mutable struct AdversarialCrosswalk <: System.SystemParameters
    mdp::PointOfClosestApproachMDP
    px
    state
end

function AdversarialCrosswalk(; dt=0.2)
    px, mdp = gen_crosswalk_problem(; dt=dt)
    s0 = rand(initialstate(mdp))
    return AdversarialCrosswalk(mdp, px, s0)
end


function System.evaluate(sparams::AdversarialCrosswalk, x::AbstractMatrix; kwargs...)
    return 0.99f0 - x[1, end]
end

function System.evaluate(sparams::AdversarialCrosswalk, x::Array{T, 3}; kwargs...) where T<:Real
    ntrajs = size(x, 3)
    rhos = zeros(ntrajs)
    for i in 1:ntrajs
        rhos[i] = System.evaluate(sparams, x[:, :, i])
    end
    return rhos
end

const CW_ASCALE = 0.60
const CW_FACTOR = CW_ASCALE / 0.25     # 2.4
const CW_NSUB   = 4
const CW_K      = 30
const CW_SDIM   = 9                     # convert_s = [PoCA_reward, 8×scene]

function crosswalk_rollout(sparams::AdversarialCrosswalk, X::AbstractMatrix; want_traj::Bool=true)
    s = rand(System.initialstate_prior(sparams))          # deterministic challenging IC (sets sparams.state)
    K = size(X, 2)
    traj = want_traj ? zeros(Float32, CW_SDIM, K) : nothing
    for k in 1:K
        x = @views X[:, k] .* CW_FACTOR
        term = false
        for _ in 1:CW_NSUB
            s = System.step(sparams, s, x)
            if POMDPs.isterminal(sparams.mdp, sparams.state); term = true; break; end
        end
        if want_traj; @views traj[:, k] .= s; end
        if term
            if want_traj; for j in (k+1):K; @views traj[:, j] .= s; end; end
            break
        end
    end
    ρ = Float64(0.99f0 - s[1])
    return want_traj ? (ρ, traj) : ρ
end
crosswalk_rollout(sparams::AdversarialCrosswalk, x::AbstractVector; kw...) =
    crosswalk_rollout(sparams, reshape(x, 5, :); kw...)

# System.simulate (amortized arm): trajectory via the shared rollout. evaluate(traj)=0.99−traj[1,end] gives ρ.
# ::AbstractVector (not ::Vector) so it matches whatever train! passes (views/SubArrays), else dispatch
# falls through to the abstract System.simulate (empty body → nothing → convert error).
System.simulate(sparams::AdversarialCrosswalk, x::AbstractVector) = crosswalk_rollout(sparams, x)[2]

function System.simulate(sparams::AdversarialCrosswalk, x::AbstractMatrix)
    K = System.get_depth(sparams)
    trajs = zeros(Float32, CW_SDIM, K, size(x, 2))
    for i in 1:size(x, 2)
        @views trajs[:, :, i] .= System.simulate(sparams, x[:, i])
    end
    return trajs
end


function System.prior(sparams::AdversarialCrosswalk)
    # Flat 150-dim iid N(0,1) prior (5 disturbances × CW_K steps), built from one vector of Normals: a
    # splat-of-vectors form (product_distribution(vec₁, …, vec₃₀)) has no method in current Distributions.jl.
    return product_distribution([Normal(0.0, 1.0) for _ in 1:5 * System.get_depth(sparams)])
end

function System.initialstate_prior(sparams::AdversarialCrosswalk)
    s0_scene = rand(initialstate(sparams.mdp))
    sparams.state = s0_scene
    s0 = convert_s(AbstractArray, s0_scene, sparams.mdp)
    return Dirac(s0)
end

function System.disturbance_prior(sparams::AdversarialCrosswalk)
    return sparams.px
end

function System.step(sparams::AdversarialCrosswalk, s::AbstractVector, x::AbstractVector)
    sp_scene, r = POMDPs.gen(sparams.mdp, sparams.state, x)
    
    # Double check that the current state matches the internal state
    if s != convert_s(AbstractArray, sparams.state, sparams.mdp)
        @warn "State mismatch: $(s) != $(convert_s(AbstractArray, sparams.state, sparams.mdp))"
    end

    sparams.state = sp_scene
    sp = convert_s(AbstractArray, sp_scene, sparams.mdp)
    return sp
end


function System.isterminal(sparams::AdversarialCrosswalk, s::AbstractVector)
    return false
end

function System.isfailure(sparams::AdversarialCrosswalk, s::AbstractVector)
    return (0.99f0 - s[1]) <= 0.0
end

function System.get_depth(sparams::AdversarialCrosswalk)
    return CW_K            # 30 disturbance-steps
end

function System.get_sdim(sparams::AdversarialCrosswalk)
    return CW_SDIM   # 9 = reward + 8 scene
end

function System.get_xdim(sparams::AdversarialCrosswalk)
    return 5 * CW_K        # d = 150
end
