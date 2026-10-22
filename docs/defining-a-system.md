# Defining a system

A problem is a subtype of `System.SystemParameters` plus a handful of methods. Nothing else in the
package needs to know what your system is.

## The interface

Subtype and extend these, all in the `System` module:

| method | returns | purpose |
|---|---|---|
| `System.prior(s)` | a `Distribution` | the nominal input distribution p |
| `System.get_xdim(s)` | `Int` | input dimension |
| `System.get_sdim(s)` | `Int` | state dimension |
| `System.get_depth(s)` | `Int` | simulation depth (1 for single-step problems) |
| `System.simulate(s, X)` | `sdim × depth × N` array | run the system on inputs |
| `System.evaluate(s, S)` | `Vector` of length N | robustness of each simulated state |

!!! info "Two call shapes"
    `train!` works in **batches**: `simulate` receives a `xdim × N` matrix and returns an
    `sdim × depth × N` array; `evaluate` receives that 3-D array and returns a length-N vector.

    The [`ams`](api-estimators.md#ams) and [`pmc`](api-estimators.md#pmc) baselines work
    **per sample**: `simulate` receives one vector and `evaluate` receives what `simulate`
    returned. Define both shapes if you want to run the baselines on your system.

## A complete example

A linear limit state in two dimensions, whose answer is known exactly:

<div class="aris-math" markdown="0">
ρ(x) = β − (x₁ + x₂)/√2,&nbsp;&nbsp; x ~ 𝒩(0, I₂)&nbsp;&nbsp; ⟹ &nbsp;&nbsp;ℙ[ρ ≤ 0] = Φ(−β)
</div>

```julia
using ARIS
using Distributions, LinearAlgebra

struct Linear2D <: System.SystemParameters
    β::Float64
end

System.prior(::Linear2D)     = MvNormal(zeros(2), I)
System.get_xdim(::Linear2D)  = 2
System.get_sdim(::Linear2D)  = 2
System.get_depth(::Linear2D) = 1

# Batched form, used by `train!`: inputs are columns, states are sdim × depth × N.
System.simulate(::Linear2D, X::AbstractMatrix) = reshape(X, 2, 1, size(X, 2))
System.evaluate(s::Linear2D, S::AbstractArray{<:Real,3}) =
    s.β .- (S[1, 1, :] .+ S[2, 1, :]) ./ √2

# Per-sample form, used by `ams` and `pmc`.
System.simulate(::Linear2D, x::AbstractVector) = x
System.evaluate(s::Linear2D, x::AbstractVector) = s.β - (x[1] + x[2]) / √2
```

Use it exactly like the toy system:

```julia
sys = Linear2D(3.5)                       # truth: Φ(−3.5) ≈ 2.3 × 10⁻⁴

cv = ConditionalValidation(
    model = JointModel(Normal(), ConditionalEMGMM(2, 3, 3)),
    xdim = 2, depth = 1, sdim = 2,
    n_samples = 1000, n_iter = 25,
    sampling_strategy = QuantileSampling(0.2; target = 0.0),
)

train!(cv, sys)
is_estimate(cv.buffer; threshold = 0.0)
```

## Getting robustness right

This is the part that determines whether the method works at all.

!!! warning "Robustness must be informative away from failure"
    **Failure is ρ ≤ 0** ([convention](rare-events.md#the-robustness-convention)), but a robustness
    that is only meaningful *at* failure defeats the method. ARIS learns q<sub>θ</sub>(x | r) over
    a whole ladder of r values, and the schedule descends through them. If ρ is effectively a step
    function — large and flat until it suddenly drops — there is no gradient for the schedule to
    follow and no structure for the conditional model to learn.

Practical guidance:

- **Make ρ continuous in x** wherever you can. Signed distance to the unsafe set is the usual
  choice: minimum clearance, minimum time-to-collision, margin to a constraint boundary.
- **Scale it sensibly.** The schedule works in quantiles so it is scale-free, but the robustness
  model is a univariate distribution (`Normal()` by default). Robustness spanning many orders of
  magnitude, or with a hard floor, is worth transforming.
- **Keep it deterministic given x** if possible. Stochastic simulators are usable, but the
  conditional model then has to represent a noisier relationship, and the gap between the requested
  r and the realised ρ widens.

For a sequential problem, fold the whole trajectory into one number — `min` over timesteps of a
per-step margin is the standard pattern, and it is what the bundled crosswalk problem does.

## Dimensions

`xdim`, `sdim` and `depth` appear both on the system and on `ConditionalValidation`. They must
agree: the buffer is preallocated from the `ConditionalValidation` values, while `train!` reads
`xdim` and `depth` from the **system**.

```julia
cv = ConditionalValidation(
    model = ...,
    xdim  = System.get_xdim(sys),
    sdim  = System.get_sdim(sys),
    depth = System.get_depth(sys),
    ...
)
```

The conditional model must match too: `ConditionalEMGMM(k, xdim + 1, xdim + 1)` — `xdim + 1`
features because robustness is stacked onto the inputs, and the conditioning index points at that
last coordinate. See [Models and distributions](api-models.md#conditionalemgmm).

## Sequential systems

For multi-step problems the `System` module declares a second group of methods —
`initialstate_prior`, `disturbance_prior`, `step`, `isterminal`, `isfailure`. These support the
MDP-style path used by the crosswalk example.

!!! note
    The `RMDP` and conditional-policy code that consumes these is behind the
    `CGV_ENABLE_POLICY` gate, disabled by default, and is not used by any paper experiment
    ([Installation](install.md#optional-integrations)). For a sequential problem you can still use
    the main path: implement batched `simulate` to roll out the whole trajectory and `evaluate` to
    reduce it to one robustness value.

## Next

- [Training a proposal](training.md): configuring the model and schedule for your system.
- [Example problems](problems.md): the four systems that ship with the package.
- [Systems and interfaces](api-systems.md): the full signature list.
