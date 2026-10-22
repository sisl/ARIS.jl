# Methodology

This page describes what [`train!`](api-training.md#train) actually does, in the order it does it.
It is the reference for the loop; [Training a proposal](training.md) is the practical guide to
configuring it.

## The loop

A run is `n_iter` batches of `n_samples` evaluations each. The simulator budget is exactly
`n_iter × n_samples` — there are no hidden evaluations.

### Batch 1: the prior

The first batch is drawn from the system's own input distribution, untruncated:

| quantity | value |
|---|---|
| inputs | x ~ p, where `p = System.prior(system)` |
| robustness | ρ = `System.evaluate(system, System.simulate(system, x))` |
| weights | **1** — the proposal *is* the prior, so p(x)/q(x) = 1 |
| log density | log p(x) |
| conditioning bound r | **+∞** — nothing was truncated |

This batch costs `n_samples` evaluations and buys the first look at the robustness distribution.
It is stored in the buffer like every other batch, so it contributes to later estimates.

### Then, for each subsequent batch

<div class="aris-math" markdown="0">
1. γ<sub>k</sub> ← schedule update &nbsp;→&nbsp; 2. r<sub>i</sub> ~ q<sub>ψ</sub>(· | r ≤ γ<sub>k</sub>) &nbsp;→&nbsp; 3. x<sub>i</sub> ~ q<sub>θ</sub>(· | r<sub>i</sub>) &nbsp;→&nbsp; 4. evaluate &nbsp;→&nbsp; 5. weight &nbsp;→&nbsp; 6. refit
</div>

1. **Update the schedule.** `QuantileSampling(q; target)` sets the next conditioning value to the
   `q`-quantile of the robustness values just observed, clamped so it never descends past
   `target`:

    <div class="aris-math" markdown="0">
    γ<sub>k</sub> = max( quantile(ρ<sup>(k−1)</sup>, q), target )
    </div>

    All `n_samples` draws in the batch share this one γ<sub>k</sub>.

2. **Draw robustness values** r<sub>i</sub> from the learned robustness model truncated above at
   γ<sub>k</sub>.

3. **Draw inputs** x<sub>i</sub> ~ q<sub>θ</sub>(· | r<sub>i</sub>) from the conditional model.

4. **Evaluate** through `simulate` then `evaluate` to get the realised ρ<sub>i</sub>. Note that
   r<sub>i</sub> is what the model was *asked for* and ρ<sub>i</sub> is what the system *returned*;
   they are different numbers, and the gap between them is how well the model has learned the
   system.

5. **Weight** each sample by the proposal it came from:

    <div class="aris-math" markdown="0">
    w<sub>i</sub> = p(x<sub>i</sub>) / q<sup>γ<sub>k</sub></sup><sub>X</sub>(x<sub>i</sub>)
    <span class="aris-where">the denominator is the marginal proposal density at this batch's threshold, computed by numerical integration.</span>
    </div>

6. **Refit** the model, and push everything into the buffer.

## Which data the refit sees

This differs by model type, and it matters.

| model | fitted on | selection |
|---|---|---|
| [`JointModel`](api-models.md#jointmodel) (the ARIS path) | the **accumulated buffer** — every batch so far | none; all rows |
| [`ConditionalGaussian`](api-models.md#conditionalgaussian) via `ConditionalGaussianCEM` | the **current batch** only | elites: the lowest-ρ samples below `𝒫.thresh_train`, at least `min_elite` |
| [`CrossEntropyMethod`](api-estimators.md#crossentropymethod) (baseline) | the current batch | the samples with ρ ≤ 0, or the `n_elite` lowest if there are fewer |

The ARIS path accumulating the whole buffer is deliberate: the conditional model is trying to
learn q<sub>θ</sub>(x | r) across the *whole* range of r, so throwing away earlier, higher-ρ
batches would discard exactly the data that pins down the easy end of the conditional.

!!! note "This is a method difference, not just an implementation detail"
    The paper's Experiment 3 compares the conditioned method against an unconditional baseline, and
    records that the conditioned method fits the accumulated buffer while the baseline fits the
    current batch's elites. The comparison therefore evaluates two complete methods, not
    conditioning in isolation. See [Paper and method](paper.md).

### Weight clamping

Weights are clamped to `max_weight` (default `10.0`) **for fitting only**. The buffer retains the
raw weights, so estimation is unaffected. This stops a single enormous weight from dominating the
EM fit and collapsing the mixture, without biasing the estimator.

## How `JointModel` refits

`Distributions.fit(jm, xs, rs, weights)` fits the two factors separately:

- **q<sub>ψ</sub>**: a weighted fit of the same univariate distribution type it started as. If you
  passed `Normal()`, you get a weighted-MLE `Normal` back.
- **q<sub>θ</sub>**: delegated to the conditional model's own `fit`.

For [`ConditionalEMGMM`](api-models.md#conditionalemgmm), "conditional" is achieved by fitting a
*joint* Gaussian mixture over the stacked vector [x; r] — which is why it takes `xdim + 1` features
— and then conditioning that mixture analytically on the last coordinate. A Gaussian mixture
conditioned on one coordinate is again a Gaussian mixture, in closed form, so sampling and density
evaluation of q<sub>θ</sub>(· | r) are cheap once the joint fit exists.

!!! note "Jitter on the conditioning coordinate"
    Because `QuantileSampling` gives every sample in a batch the *same* γ<sub>k</sub>, the realised
    r values used for fitting would be highly concentrated, leaving the joint mixture near-singular
    in that coordinate. The fit adds Gaussian jitter of scale 10⁻⁴ to the robustness column before
    the EM step. It is a numerical guard, and it is small relative to any threshold spacing you
    would use.

The EM fit runs several restarts from a weighted k-means initialisation, inflates near-singular
component covariances, and falls back to the k-means initialisation if every EM restart fails.
Those events are counted rather than printed; read them with `get_fit_stats()` after a run
([Models and distributions](api-models.md#fit-diagnostics)).

## The estimator

After training, the buffer holds every evaluated sample with its raw weight and its realised
robustness. [`is_estimate`](api-estimators.md#is_estimate) is a plain weighted mean over all of
them:

<div class="aris-math" markdown="0">
P̂(γ) = (1/N<sub>total</sub>) Σ<sub>i</sub> 1{ρ<sub>i</sub> ≤ γ} · w<sub>i</sub>
<span class="aris-where">N<sub>total</sub> = n_iter × n_samples; the sum runs over every batch, each sample weighted by its own proposal of origin.</span>
</div>

This is **not** self-normalised — the denominator is the fixed sample count, not Σw. Each
sample's weight already refers to the proposal that produced it, so pooling batches is just
averaging.

Note what this means for amortization: γ appears only in the indicator. The weights were fixed at
sampling time and do not depend on the threshold you ask about. That is why one trained run
answers a whole grid of thresholds for free — and also why
[reach and support still have to be checked](multiple-thresholds.md).

## Unbiasedness

!!! warning "What is and is not established"
    Proposal-of-origin adaptive importance sampling is unbiased **under the conditional-history and
    support assumptions stated in the paper**. The pooled adaptive estimator does not come with an
    unbiasedness guarantee, because later proposals depend on earlier samples — the batches are not
    independent draws.

    The paper's multi-threshold results use a pooled **balance-MIS** estimator, which is
    implemented under `paper/`, not in the package. `is_estimate` is the proposal-of-origin
    estimator described above.

## Stratified adaptation

Two hooks exist for defensive-mixture proposals, where some draws come from a safe fallback
component:

```julia
defensive_mask(model, n)      # which draws came from the defensive component
decouple_adaptation(model)    # opt in to excluding them from fitting
```

A model that opts in has its schedule and its fit computed from the adaptive draws only, while
estimation still uses every draw. The defaults — `falses(n)` and `false` — make this inert for all
the models shipped with the package.

## Next

- [Training a proposal](training.md): choosing the model, schedule and budget.
- [Estimating failure probabilities](estimating.md): reading estimates and diagnostics off a run.
- [Training and sampling](api-training.md): signatures for everything named here.
