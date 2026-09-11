module ARIS

using Random
using LinearAlgebra
using Statistics
using Distributions
using StatsBase
using Parameters
using ExpectationMaximization
using Clustering
using QuadGK
using Optim
using ForwardDiff
# Optional evaluation dependency.
# using StructuredGaussianMixtures
import Base.rand

# Optional feature gates for flow and policy/MDP integrations.
# Both are disabled by default and require their external dependencies to be
# available in the active environment.
const ENABLE_FLOW = get(ENV, "CGV_ENABLE_FLOW", "false") == "true"
const ENABLE_POLICY = get(ENV, "CGV_ENABLE_POLICY", "false") == "true"

@static if ENABLE_FLOW
    using PyCall
    const torch = PyNULL()
    const nf = PyNULL()
    const np = PyNULL()

    function __init__()
        copy!(torch, pyimport("torch"))
        copy!(nf, pyimport("normflows"))
        copy!(np, pyimport("numpy"))
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("medium")
    end
end

# Compatibility methods for Gaussian fitting.
include("compat_distributions.jl")

include("System.jl")
using .System

export
    System,
    SystemParameters

include("buffer.jl")
export Buffer, push!

include("logging.jl")
export LoggerParams, log_failure_probability, elapsed, log_target_performance, log_weights, log_failrate

include("condition_strategies.jl")
export TargetSampling,
    QuantileSampling,
    CrossEntropySampling,
    GaussianNES

include("ccem.jl")
export ConditionalValidation,
    ConditionalGaussianCEM,
    train!,
    n_samples,
    n_iter,
    defensive_mask,
    decouple_adaptation,
    adaptive_idx

include("conditional_gaussian.jl")
export ConditionalGaussian

@static if ENABLE_FLOW
    include("flows.jl")
    export PyConditionalNormalizingFlow,
        AutoregressiveSplineFlow,
        FlowTrainingParams
end

include("utils.jl")
export bslice, ess, is_estimate, conditional, marginal

include("baselines/ams.jl")
export ams

include("baselines/pmc.jl")
export pmc

include("baselines/cem.jl")
export CrossEntropyMethod, train!

@static if ENABLE_POLICY
    include("system_mdp.jl")
    export RMDP

    include("policy_proposal.jl")
    export ConditionalPolicy, ConditionalPolicySolver, to_buffer, train!, mle_loss, vanilla_mle_loss, policy_training, conditional_episodes!
end

include("gmm.jl")
export GaussianMixture

include("gmm_em.jl")
export EMGMM, ConditionalEMGMM, reset_fit_stats!, get_fit_stats

include("joint_model.jl")
export JointModel, TruncatedJointModel

include("marginal_density.jl")
export marginal_pdf, marginal_logpdf, marginal_reference, marginal_settings,
    conditional_logpdf_at,
    MARGINAL_METHOD, MARGINAL_RTOL, MARGINAL_LOWER, MARGINAL_NODES

end
