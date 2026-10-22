# Training and sampling

Running a proposal against a system. The guide version is
[Training a proposal](training.md); the loop is described in [Methodology](methodology.md).

## `ConditionalValidation`

```text
ConditionalValidation(; model, xdim, depth, sdim, n_samples, n_iter, sampling_strategy, ...)
```

A keyword-constructed, mutable container for one adaptive run. `train!` mutates it in place.

### Core options

| keyword | default | meaning |
|---|---|---|
| `model` | — | the proposal, usually a [`JointModel`](api-models.md#jointmodel) |
| `xdim` | — | input dimension |
| `depth` | — | simulation depth |
| `sdim` | — | state dimension |
| `n_samples` | `1000` | evaluations per iteration |
| `n_iter` | `1` | iterations; total budget is `n_iter × n_samples` |
| `sampling_strategy` | `TargetSampling(0.0)` | the conditioning schedule |
| `max_weight` | `10.0` | weight clamp **for fitting only** |

### Buffer options

| keyword | default | meaning |
|---|---|---|
| `buffer_size` | `n_samples * n_iter` | capacity; the default holds the whole run |
| `ignore_keys` | `Symbol[]` | columns not to allocate; `[:s]` saves memory on large states |
| `buffer` | built from the above | the [`Buffer`](api-systems.md#buffer) results land in |

### Other options

| keyword | default | meaning |
|---|---|---|
| `sample_batchsize` | `nothing` | draw each batch in chunks instead of all at once |
| `log_callback` | `(cv, D) -> nothing` | called each iteration with that iteration's batch |
| `log` | `nothing` | a [`LoggerParams`](#loggerparams) for periodic logging |
| `𝒫` | `(;)` | algorithm-specific parameters; used by `ConditionalGaussianCEM` |
| `collect_states` | `false` | reserved |
| `i` | `0` | iteration counter, advanced by `train!` |
| `objective_smoothing` | `0.01` | smoothing for the KL objective used by some schedules |
| `r_strategy` | `:recent` | which robustness values the flow path fits against |
| `γk` | `Inf` | reserved |

### Accessors

```text
n_samples(cv) -> Int
n_iter(cv)    -> Int
```

!!! warning "Dimensions are read from two places"
    The buffer is preallocated from the `ConditionalValidation` values, but `train!` reads `xdim`
    and `depth` from the **system**. They must agree, or the buffer will be the wrong shape.

## `train!`

```text
train!(cv::ConditionalValidation, system::System.SystemParameters;
       rng = Random.GLOBAL_RNG, show_progress = false)
```

Runs the adaptive loop. Mutates `cv`: `cv.model` ends up fitted and `cv.buffer` holds every
evaluated sample.

```julia
train!(cv, sys; rng = MersenneTwister(42))
```

The first iteration draws from the system's prior with unit weights and conditioning value +∞;
subsequent iterations draw from the model at the schedule's current threshold. The simulator is
called exactly `n_iter × n_samples` times.

There is also a method for the [`CrossEntropyMethod`](api-estimators.md#crossentropymethod)
baseline with the same shape.

!!! note
    `show_progress` is accepted for interface compatibility and does not currently drive a
    progress display. Use `log_callback` to observe a run
    ([example](training.md#watching-a-run)).

## `ConditionalGaussianCEM`

```text
ConditionalGaussianCEM(; xdim = 1, depth = 1, sdim = 1,
                         thresh_train = Inf, thresh_target = 0.0, min_elite = 1, kwargs...)
```

Convenience constructor for a `ConditionalValidation` whose model is a
[`ConditionalGaussian`](api-models.md#conditionalgaussian), with the elite-selection parameters
packed into `𝒫`. Remaining keywords are forwarded.

This path fits the **current batch's** elites — samples below `thresh_train`, at least
`min_elite` of them — rather than the accumulated buffer
([Methodology](methodology.md#which-data-the-refit-sees)).

## Sampling strategies

The schedule that decides each batch's conditioning threshold. All subtype
`ARIS.SamplingStrategy`.

### `QuantileSampling`

```text
QuantileSampling(quantile::AbstractFloat; target::Real = 0.0)
```

The usual choice. After each batch, the next conditioning value becomes

<div class="aris-math" markdown="0">
γ<sub>k</sub> = max( quantile(ρ<sup>(k−1)</sup>, q), target )
</div>

so the schedule descends toward `target` and stops there. All draws in a batch share one
γ<sub>k</sub>.

```julia
QuantileSampling(0.2; target = 0.0)
```

| field | meaning |
|---|---|
| `quantile` | the quantile taken each iteration |
| `value` | the current conditioning value; starts at `0.0` and is set on the first update |
| `target` | the descent floor |

!!! warning
    `target` is where the schedule **stops**, not where it is guaranteed to **arrive**. Check
    `minimum(cv.buffer[:r])` after training — see [reach](multiple-thresholds.md#reach).

### `TargetSampling`

```text
TargetSampling(thresh_target::Float64)
```

Every batch conditions on a fixed threshold; no ladder. The default
`sampling_strategy` for `ConditionalValidation`, and only sensible when the proposal already
covers the target region.

### `CrossEntropySampling`

```text
CrossEntropySampling(; distribution::Distribution, quantile::Float64, n::Int = 1)
```

Draws conditioning values from `distribution`, then refits it to the values whose smoothed KL
objective falls in the best `quantile`. `n` repeats each drawn value that many times within the
batch.

### `GaussianNES`

```text
GaussianNES(; mu::Float64, sigma::Float64, lr::Float64, n::Int = 1)
```

Maintains a Gaussian over conditioning values, updated by natural-evolution-strategy gradient
steps on the same objective. `n` repeats each drawn value within the batch.

## Stratified adaptation

Hooks for defensive-mixture proposals, where some draws come from a safe fallback component:

```text
defensive_mask(model, n::Int)         -> BitVector   which draws were defensive; default falses(n)
decouple_adaptation(model)            -> Bool        opt in to excluding them; default false
adaptive_idx(cv, buffer)              -> Vector{Int} or nothing
```

A model opting in has its schedule and fit computed from the adaptive stratum only, while
estimation uses every draw. `adaptive_idx` returns `nothing` — meaning no filtering — unless the
model opts in **and** the buffer holds a mix of both strata. Inert for every model shipped with
the package.

## `LoggerParams`

```text
LoggerParams(; dir = "log/", period = 1, verbose = true, fns = [...], writeout = Dict(),
               use_wandb = false, config = nothing, project = nothing, entity = nothing,
               notes = nothing, logger = nothing)
```

Periodic logging during training. `fns` defaults to `[log_failure_probability, log_weights,
log_failrate]`; `writeout` maps a period to a function called every that many iterations.

```julia
cv = ConditionalValidation(..., log = LoggerParams(period = 5))
```

Exported alongside it: `elapsed`, `log_failure_probability`, `log_target_performance`,
`log_weights`, `log_failrate`.

!!! note
    `logger` is untyped and defaults to `nothing` because TensorBoard and Weights & Biases are not
    dependencies of this package. With no logger attached, logging prints and is otherwise inert.
    For most uses `log_callback` is simpler ([example](training.md#watching-a-run)).

## Next

- [Estimators and baselines](api-estimators.md): turning a buffer into an estimate.
- [Methodology](methodology.md): what the loop does, step by step.
