# Example problems

Four `System` implementations ship under
[`examples/problems/`](https://github.com/sisl/ARIS.jl/tree/main/examples/problems). Each is a
plain script, not part of the `ARIS` module — `include` it:

```julia
include(joinpath("examples", "problems", "toy.jl"))
```

They are kept outside the package so that they can depend on things the package does not, which
matters for the crosswalk problem in particular.

Each one is also the problem behind one of the paper's experiments
([Reproducing experiments](reproduce.md)).

| file | type | xdim | failure modes | closed form | experiment |
|---|---|---:|---|---|---|
| `toy.jl` | `UnimodalToy` | 2 | one | yes | Exp1 |
| `radial_shell.jl` | `RadialShell` | 20 | a shell | yes | Exp2 |
| `tailrace.jl` | `TailRaceToy` | 2+ | two | no | Exp3 |
| `crosswalk.jl` | `AdversarialCrosswalk` | 150 | — | no | Exp4 |

## `UnimodalToy` — the corner

```julia
UnimodalToy(γ)
```

<div class="aris-math" markdown="0">
ρ(x) = γ − min(x₁, x₂),&nbsp;&nbsp; x ~ 𝒩(0, I₂)&nbsp;&nbsp; ⟹ &nbsp;&nbsp;ℙ[ρ ≤ g] = Φ(g − γ)²
</div>

`xdim = 2`, `sdim = 2`, `depth = 1`. The failure region is the corner where both coordinates are
large, so it is unimodal and compact. The closed form makes it the right problem for checking that
a setup works at all, and it is what [Basic usage](basic-usage.md) uses.

```julia
sys = UnimodalToy(3.0)
truth(g) = cdf(Normal(), g - 3.0)^2
```

## `RadialShell` — high dimensions

```julia
RadialShell(; r_crit = 7.5, d = 20)
```

<div class="aris-math" markdown="0">
ρ(x) = r<sub>crit</sub> − ‖x‖,&nbsp;&nbsp; x ~ 𝒩(0, I<sub>d</sub>)&nbsp;&nbsp; ⟹ &nbsp;&nbsp;ℙ[ρ ≤ g] = ℙ[χ²<sub>d</sub> > (r<sub>crit</sub> − g)²]
</div>

`xdim = sdim = d` (20 by default), `depth = 1`. The script provides the exact curve:

```julia
shell_truth(s::RadialShell, γ) = ccdf(Chisq(s.d), (s.r_crit - γ)^2)
```

Failure is everything *outside* a radius, so the failure region is a thin spherical shell — it has
no preferred direction, and in 20 dimensions it is nowhere near any particular point. This is the
problem that punishes a proposal for collapsing onto one direction, and it is where the schedule's
**reach** becomes the limiting factor rather than the model's flexibility. The paper's Experiment 2
reports estimates only at thresholds the schedule actually got to.

Remember to scale the conditional model with the dimension:

```julia
sys = RadialShell(; r_crit = 7.5, d = 20)
model = JointModel(Normal(), ConditionalEMGMM(2, 21, 21))   # d + 1, d + 1
cv = ConditionalValidation(model = model, xdim = 20, sdim = 20, depth = 1, ...)
```

## `TailRaceToy` — the multimodal tail

```julia
TailRaceToy(; c1 = 5.5, c2 = 23.0, d = 2)
```

<div class="aris-math" markdown="0">
ρ(x) = min( c₁ − x₁ ,&nbsp; c₂ − x₂² ),&nbsp;&nbsp; x ~ 𝒩(0, I<sub>d</sub>)
</div>

`xdim = sdim = d`, `depth = 1`. Dimensions 3…`d` are nuisance coordinates that do not enter ρ.

The point is the **x₂² term**: it is symmetric, so the second failure mode occurs at both
x₂ ≈ +√c₂ and x₂ ≈ −√c₂. Two mechanisms race to be the binding constraint, and one of them is
itself two-sided.

!!! warning "This problem exists to break unimodal proposals"
    A `ConditionalGaussian`, or a mixture with too few components, will find one mode and report a
    confident underestimate — with no error and nothing in the diagnostics that obviously says
    "wrong", because the samples it does find are perfectly valid. Use enough components, and check
    `get_fit_stats()` for collapse:

    ```julia
    model = JointModel(Normal(), ConditionalEMGMM(3, 3, 3))
    ```

There is no closed form here. The paper treats it as a single-threshold study.

## `AdversarialCrosswalk` — the realistic one

```julia
AdversarialCrosswalk(; dt = 0.2)
```

A sequential autonomous-driving scenario: an adversarial pedestrian crossing in front of a vehicle
running an intelligent-driver model. Robustness is derived from the point of closest approach —
`0.99 − x[1, end]` on the converted trajectory, so failure means the approach got too close.

| | |
|---|---|
| `xdim` | **150** — 5 disturbances × 30 steps |
| `depth` | 30 |
| `sdim` | 9 — the point-of-closest-approach reward plus 8 scene variables |
| prior | 150 iid 𝒩(0, 1) disturbances |

!!! warning "Separate environment required"
    This problem needs `AdversarialDriving`, `AutomotiveSimulator`, `AutomotiveVisualization`,
    `POMDPs`, `POMDPTools` and `Crux`, none of which are dependencies of `ARIS`. Use the
    Experiment 4 environment:

    ```bash
    julia --project=paper/envs/idm -e 'using Pkg; Pkg.instantiate()'
    ```

    That environment includes Git-based dependencies and takes a while to instantiate on a fresh
    machine. See [`paper/README.md`](https://github.com/sisl/ARIS.jl/blob/main/paper/README.md).

This is the problem where the amortization claim is worth something: a 150-dimensional input, a
simulator in the loop, and a whole threshold grid wanted rather than a single number. The paper
reports an eight-point grid over ten seeds.

## Writing your own

See [Defining a system](defining-a-system.md). The toy problems are deliberately short — `toy.jl`
is about 40 lines — and are the best templates to start from.

## Next

- [Basic usage](basic-usage.md): the runnable script, walked through.
- [Reproducing experiments](reproduce.md): how these map onto the paper's results.
