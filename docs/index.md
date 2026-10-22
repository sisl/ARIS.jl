# Welcome to ARIS.jl! { .aris-home-title }

[:simple-github: GitHub Repository](https://github.com/sisl/ARIS.jl){ .md-button aria-label="GitHub repository" }
[:material-file-document-outline: Paper and Reproducibility](https://github.com/sisl/ARIS.jl/tree/main/paper){ .md-button aria-label="Paper and reproducibility materials" }
{ .aris-resources }

**Amortized Rare-Event Importance Sampling: one proposal, reused across failure thresholds.**

`ARIS.jl` is a Julia package for estimating the probability that a system fails, when failure is
too rare for naive Monte Carlo to see. It learns a *joint* model over system inputs and the
robustness value they produce. Because the model is conditional on robustness, a single trained
proposal yields an importance-sampling proposal for **any** threshold, by truncating the learned
robustness distribution — rather than being specialised to the one threshold it was trained on.

Developed by the Stanford Intelligent Systems Laboratory (SISL).

[Quickstart](quickstart.md){ .md-button .md-button--primary }
[Installation](install.md){ .md-button }

```julia
using ARIS
using Distributions, Random

include("examples/problems/toy.jl")

Random.seed!(1)
sys = UnimodalToy(3.0)

model = JointModel(Normal(), ConditionalEMGMM(2, 3, 3))

cv = ConditionalValidation(
    model = model,
    xdim = 2, depth = 1, sdim = 2,
    n_samples = 1000, n_iter = 25,
    sampling_strategy = QuantileSampling(0.2; target = 0.0),
)

train!(cv, sys)

is_estimate(cv.buffer; threshold = 0.0)
```

!!! info "Robustness convention"
    A system returns a **robustness** value ρ for each input. Failure means ρ ≤ 0, and more
    negative values are deeper failures. Every estimator in the package targets ρ ≤ 0 unless you
    pass another threshold. See [Rare-event estimation](rare-events.md#the-robustness-convention).

## What amortization buys you

<div class="grid cards" markdown>

-   **One fit, many thresholds**

    ---

    The proposal is a joint model q(x, r) over inputs and robustness. Restricting r ≤ γ turns it
    into a proposal aimed at the failure region {ρ ≤ γ}, for any γ, with no refitting and no extra
    simulator calls.
    See [Amortized importance sampling](amortization.md).

-   **An adaptive schedule, not a leap**

    ---

    Training walks the conditioning threshold down from the prior toward the target, so the
    proposal is never asked to jump straight into the rare set.
    See [Methodology](methodology.md).

</div>

## Where to go next

- [Quickstart](quickstart.md): a complete run, end to end, with the numbers it produces.
- [Rare-event estimation](rare-events.md): the problem, the robustness convention, and why
  importance sampling is needed.
- [Defining a system](defining-a-system.md): the five methods your problem must implement.
- [Working with multiple thresholds](multiple-thresholds.md): reading a whole curve off one run,
  and the assumptions that carries.
- [API reference](api-systems.md): what the package exports, by signature.

## Scope

!!! warning "Package API versus paper code"
    This site documents the **package** — what `using ARIS` gives you. The paper's
    multi-threshold estimator (pooled balance-MIS), its experiment drivers, metrics and figures
    live under [`paper/`](https://github.com/sisl/ARIS.jl/tree/main/paper) and are not part of the
    public API. [Paper and method](paper.md) says which is which.

The package is not registered in Julia's General registry; it is installed from a clone
([Installation](install.md)). Optional normalizing-flow and policy/MDP integrations exist behind
environment-variable gates, are disabled by default, and are not used by any paper experiment.
