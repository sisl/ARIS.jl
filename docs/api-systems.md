# Systems and interfaces

What `using ARIS` exports for defining problems and holding results. The guide version of this
page is [Defining a system](defining-a-system.md).

## `System`

A submodule, exported by name. Problem definitions extend its functions, which is why they are
written `System.prior`, `System.evaluate` and so on rather than being imported directly.

```julia
using ARIS          # brings System and SystemParameters into scope
```

### `SystemParameters`

```text
abstract type System.SystemParameters
```

The base type for every problem. Also exported unqualified as `SystemParameters`.

### Required methods

Extend all of these for a problem used with `train!`:

```text
System.prior(s)      -> Distribution      # nominal input distribution p
System.get_xdim(s)   -> Int               # input dimension
System.get_sdim(s)   -> Int               # state dimension
System.get_depth(s)  -> Int               # simulation depth; 1 for single-step problems
System.simulate(s, X::AbstractMatrix)            -> AbstractArray{<:Real,3}
System.evaluate(s, S::AbstractArray{<:Real,3})   -> Vector
```

`simulate` receives a `xdim × N` matrix of inputs as columns and returns an `sdim × depth × N`
array. `evaluate` receives that array and returns a length-N vector of robustness values.

!!! info "Robustness convention"
    `evaluate` returns **robustness**: failure is ρ ≤ 0, positive means success, and more negative
    means a deeper failure. See [Rare-event estimation](rare-events.md#the-robustness-convention).

### Per-sample methods

The [`ams`](api-estimators.md#ams) and [`pmc`](api-estimators.md#pmc) baselines call the
single-sample forms instead:

```text
System.simulate(s, x::AbstractVector)
System.evaluate(s, x::AbstractVector)
```

Define both shapes if you want to run those baselines on your system.

### Sequential methods

Declared for MDP-style problems:

```text
System.initialstate_prior(s)
System.disturbance_prior(s)
System.step(s, state, x)
System.isterminal(s, state)
System.isfailure(s, state)
```

!!! note
    The `RMDP` and conditional-policy code that consumes these is behind the `CGV_ENABLE_POLICY`
    gate, disabled by default, and unused by every paper experiment. A sequential problem can be
    expressed through the batched `simulate`/`evaluate` path instead.

## `Buffer`

```text
Buffer(xdim::Int, sdim::Int, depth::Int, capacity::Int, ignored_keys::Vector)
```

Preallocated storage for evaluated samples. `ConditionalValidation` builds one for you; you rarely
construct it directly.

### Columns

```julia
buffer[:x]        # xdim × N          inputs
buffer[:ρ]        # N                 realised robustness
buffer[:s]        # sdim × depth × N  simulated states
buffer[:w]        # N                 raw importance weights
buffer[:r]        # N                 conditioning value in force (+Inf for the prior batch)
buffer[:logpdf]   # N                 log proposal density at the sample
buffer[:defensive]# N (BitVector)     defensive-component indicator; all false for shipped models
```

Indexing returns a view of the filled rows only.

!!! warning "Weights are raw"
    `buffer[:w]` holds the unclamped proposal-of-origin weights. `max_weight` clamps weights
    **for fitting**; it does not alter what is stored, so estimation is unaffected
    ([Methodology](methodology.md#weight-clamping)).

### Methods

| call | returns |
|---|---|
| `length(buffer)` | number of samples stored |
| `keys(buffer)` | stored column names |
| `haskey(buffer, :w)` | whether a column is stored |
| `push!(buffer, x, ρ, s, w, r, logpdf[, defensive])` | append one sample, or a batch when the arguments are batched |

### Dropping columns

`ignored_keys` (via `ConditionalValidation`'s `ignore_keys`) skips allocation of a column
entirely. Dropping `:s` is the usual win when states are large and only robustness and weights
matter:

```julia
cv = ConditionalValidation(..., ignore_keys = [:s])
```

`is_estimate` needs only `:w` and `:ρ`.

## Utilities

### `ess`

```text
ess(weights) -> Float64
```

Kish's effective sample size, (Σw)² / Σw². See
[diagnostics](estimating.md#diagnostics).

### `bslice`

```text
bslice(v, i)
```

Slices the last dimension of an array of any rank, returning a view for rank ≥ 2. Used internally
by `Buffer`.

### `conditional`

```text
conditional(d::MvNormal, cidx::Vector{Int}, cval::Vector{Float64}) -> MvNormal
```

The Gaussian conditional of `d` with the coordinates in `cidx` fixed at `cval`. A small ridge
(10⁻³ I) is added to the conditional covariance for numerical stability.

### `marginal`

```text
marginal(d::MvNormal, dims::Vector{Int}) -> MvNormal
marginal(d::MvNormal, dim::Int)          -> Normal
```

The Gaussian marginal over the given coordinates.

## Next

- [Models and distributions](api-models.md): the proposal models.
- [Training and sampling](api-training.md): `ConditionalValidation`, `train!`, schedules.
