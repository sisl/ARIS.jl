# Training a proposal

[`ConditionalValidation`](api-training.md#conditionalvalidation) holds a run: the model, the
budget, the schedule, and the buffer that collects results. [`train!`](api-training.md#train)
executes it.

```julia
cv = ConditionalValidation(
    model = JointModel(Normal(), ConditionalEMGMM(2, 3, 3)),
    xdim = 2, depth = 1, sdim = 2,
    n_samples = 1000, n_iter = 25,
    sampling_strategy = QuantileSampling(0.2; target = 0.0),
    max_weight = 10.0,
)

train!(cv, sys)
```

`train!` mutates `cv` in place: `cv.model` ends up fitted and `cv.buffer` holds every evaluated
sample.

## Choosing the model

The ARIS proposal is `JointModel(robustness_model, conditional_model)`.

### The robustness factor

A univariate `Distributions.jl` distribution over ρ, refitted by weighted MLE each iteration.
`Normal()` is the default choice and what the examples use. Whatever you pass must support
`Distributions.fit(typeof(d), values, weights)` and `truncated(d; upper = γ)`.

### The conditional factor

| model | when |
|---|---|
| [`ConditionalEMGMM(k, xdim+1, xdim+1)`](api-models.md#conditionalemgmm) | the default. A `k`-component mixture, able to represent multiple failure modes |
| [`ConditionalGaussian(xdim)`](api-models.md#conditionalgaussian) | a single Gaussian; cheap and stable, but unimodal |

Both arguments after `k` are determined by your input dimension, and both are `xdim + 1`:

```julia
ConditionalEMGMM(k, xdim + 1, xdim + 1)
#                  ──┬─────   ──┬─────
#            features: [x; ρ]   conditioning index: the ρ coordinate
```

**Choosing `k`.** Start at 2–3. The mixture is fitted over the *joint* `[x; ρ]` space, so
components are spent on both failure modes and on the shape of the x–ρ relationship. Too many
components on too little data collapses covariances; the fit guards against this by inflating
near-singular components and counting the event — check `get_fit_stats()` afterwards
([diagnostics](api-models.md#fit-diagnostics)).

!!! warning "Multimodality is a modelling decision, not an automatic one"
    If failure can occur in genuinely distinct regions of input space, a single Gaussian cannot
    represent it and will collapse onto whichever mode it finds first — returning a confident
    underestimate. The package will not warn you. The bundled `tailrace.jl` problem exists to
    exercise exactly this case.

## Choosing the schedule

The schedule decides which conditioning threshold each batch targets.

### `QuantileSampling` — the usual choice

```julia
QuantileSampling(0.2; target = 0.0)
```

Each batch conditions on the `q`-quantile of the previous batch's robustness, floored at `target`.
With `q = 0.2`, each step aims at roughly the worst fifth of what was just seen, so the schedule
descends geometrically rather than jumping.

- **Smaller `q`** (0.05–0.1): descends faster, fewer iterations to reach the target, but each step
  is a bigger extrapolation and weights degrade faster.
- **Larger `q`** (0.3–0.5): a gentler ladder, better-behaved weights, more iterations needed.
- **`target`** is a floor, not a goal — the schedule stops descending there. Set it to the
  threshold you care about, normally `0.0`.

!!! note "Reach is not guaranteed"
    `target` is where the schedule *stops*, not where it is promised to *arrive*. If the budget
    runs out first, the conditioning value never gets there. After training, check:

    ```julia
    minimum(cv.buffer[:r])     # the deepest conditioning value actually used
    minimum(cv.buffer[:ρ])     # the deepest robustness actually observed
    ```

### The others

| strategy | behaviour |
|---|---|
| [`TargetSampling(γ)`](api-training.md#targetsampling) | every batch conditions on a fixed γ. No ladder — only sensible when the proposal is already close |
| [`CrossEntropySampling`](api-training.md#crossentropysampling) | draws conditioning values from a distribution refitted to the elite values of a smoothed KL objective |
| [`GaussianNES`](api-training.md#gaussiannes) | maintains a Gaussian over conditioning values, updated by natural-evolution-strategy gradient steps |

`QuantileSampling` is what the examples and the paper's reported runs use.

## Choosing the budget

`n_iter × n_samples` is the exact simulator-evaluation count.

- **`n_samples`** must be large enough for the refit. A GMM over `xdim + 1` dimensions with `k`
  components needs enough samples per component for the covariances to be estimable; 1000 is a
  reasonable floor for low-dimensional problems.
- **`n_iter`** must be enough for the schedule to walk from the prior down to `target`. With
  `q = 0.2`, each iteration moves the conditioning value roughly to the 20th percentile of the
  last, so the number of steps needed scales with how many orders of magnitude separate the prior
  from the target.

Both too-small failure modes look the same from the outside — a plausible number — so check the
diagnostics rather than the estimate.

## Other options worth knowing

| option | default | effect |
|---|---|---|
| `max_weight` | `10.0` | clamps weights **for fitting only**; the buffer keeps raw weights for estimation |
| `sample_batchsize` | `nothing` | draws the batch in chunks instead of all at once, for memory-bound models |
| `ignore_keys` | `Symbol[]` | buffer fields not to store. `[:s]` saves a lot of memory when states are large and you only need robustness and weights |
| `buffer_size` | `n_samples * n_iter` | buffer capacity; the default holds the whole run |
| `log_callback` | no-op | `(cv, D) -> ...`, called each iteration with that iteration's batch |
| `log` | `nothing` | a [`LoggerParams`](api-training.md#loggerparams) for periodic logging |

### Watching a run

```julia
cv = ConditionalValidation(
    model = JointModel(Normal(), ConditionalEMGMM(2, 3, 3)),
    xdim = 2, depth = 1, sdim = 2,
    n_samples = 1000, n_iter = 25,
    sampling_strategy = QuantileSampling(0.2; target = 0.0),
    log_callback = function (cv, D)
        ρs, ws = D[:ρ], D[:w]
        println("  r = ", round(maximum(D[:r]); digits = 4),
                "  min ρ = ", round(minimum(ρs); digits = 4),
                "  hits = ", count(<=(0.0), ρs),
                "  ESS = ", round(ess(ws); digits = 1))
    end,
)

train!(cv, sys)
```

Printed per iteration, this is the fastest way to see whether the schedule is descending, whether
failures are being found, and whether the weights are degenerating.

## Reproducibility

`train!` takes an `rng`:

```julia
train!(cv, sys; rng = MersenneTwister(42))
```

Without it, the global RNG is used, so `Random.seed!` before the call also works. Results are
reproducible for a fixed Julia version **and** dependency set — the pinned `Manifest.toml` is part
of the specification, not a convenience.

## Next

- [Estimating failure probabilities](estimating.md): what to do with a trained run.
- [Methodology](methodology.md): what each iteration does internally.
