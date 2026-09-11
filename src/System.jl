module System

export
    SystemParameters,
    reset,
    initialize,
    evaluate

"""
Abstract base type for parameters used by the system under test.
"""
abstract type SystemParameters end

# Referenced by the abstract `simulate` return annotation below (reached e.g. on the crosswalk path,
# whose simulate can dispatch through the abstract method).
const VectorOrMatrix = Union{AbstractVector, AbstractMatrix, AbstractArray}

"""
Interface function to simulate the system under test (SUT) given the generated input.
"""
function simulate(sparams::SystemParameters, inputs::AbstractVector; kwargs...)::VectorOrMatrix end

"""
Interface function to call/evaluate the system under test (SUT) given the generated input.
Returns robustness, with zero indicating failure and positive values indicating success.
"""
function evaluate(sparams::SystemParameters, inputs::AbstractVector; kwargs...)::Vector end

function get_depth(sparams::SystemParameters)::Int end

function get_xdim(sparams::SystemParameters)::Int end

function get_sdim(sparams::SystemParameters)::Int end

function prior(sparams::SystemParameters)::Vector end

## For sequential problems

function initialstate_prior(sparams::SystemParameters)
end

function disturbance_prior(sparams::SystemParameters)
end

function step(sparams::SystemParameters, s::AbstractVector, x::AbstractVector)
end

function isterminal(sparams::SystemParameters, s::AbstractVector)
end

function isfailure(sparams::SystemParameters, s::AbstractVector)
end

end # module
