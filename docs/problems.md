# Example problems

Four `System` implementations ship under
[`examples/problems/`](https://github.com/sisl/ARIS.jl/tree/main/examples/problems). Each is a
plain script, not part of the `ARIS` module — `include` it:

```julia
include(joinpath("examples", "problems", "toy.jl"))
```

They are kept outside the package so that they can depend on things the package does not, which
matters for the crosswalk problem in particular.

| file | type | xdim | failure modes | closed form |
|---|---|---:|---|---|
| `toy.jl` | `UnimodalToy` | 2 | one | yes |
| `radial_shell.jl` | `RadialShell` | 20 | a shell | yes |
| `tailrace.jl` | `TailRaceToy` | 2+ | two | no |
| `crosswalk.jl` | `AdversarialCrosswalk` | 150 | — | no |

## `UnimodalToy`

```julia
UnimodalToy(γ)
```

<div class="aris-math" markdown="0">
ρ(x) = γ − min(x₁, x₂),&nbsp;&nbsp; x ~ 𝒩(0, I₂)&nbsp;&nbsp; ⟹ &nbsp;&nbsp;ℙ[ρ ≤ g] = Φ(g − γ)²
</div>

`xdim = 2`, `sdim = 2`, `depth = 1`. The failure region is the corner where both coordinates are
large, and is therefore unimodal and compact. The closed form makes it suitable for verifying a
configuration; it is used in [Basic usage](basic-usage.md).

```julia
sys = UnimodalToy(3.0)
truth(g) = cdf(Normal(), g - 3.0)^2
```

## `RadialShell`

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

Failure is everything outside a radius, so the failure region is a thin spherical shell with no
preferred direction, and in 20 dimensions it is far from any particular point. A proposal that
collapses onto one direction will underestimate here, and schedule reach rather than model
flexibility is the limiting factor.

Scale the conditional model with the dimension:

```julia
sys = RadialShell(; r_crit = 7.5, d = 20)
model = JointModel(Normal(), ConditionalEMGMM(2, 21, 21))   # d + 1, d + 1
cv = ConditionalValidation(model = model, xdim = 20, sdim = 20, depth = 1, ...)
```

## `TailRaceToy`

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

!!! warning "Unimodal proposals fail on this problem"
    A `ConditionalGaussian`, or a mixture with too few components, will find one mode and report a
    underestimate. No error is raised, and the diagnostics do not flag it, because the samples it
    does find are valid. Use enough components and check `get_fit_stats()` for collapse:

    ```julia
    model = JointModel(Normal(), ConditionalEMGMM(3, 3, 3))
    ```

There is no closed form here.

## `AdversarialCrosswalk`

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
    separate environment kept in the repository for it:

    ```bash
    julia --project=paper/envs/idm -e 'using Pkg; Pkg.instantiate()'
    ```

    That environment includes Git-based dependencies and takes a while to instantiate on a fresh
    machine.

This problem combines a 150-dimensional input, a simulator in the loop, and a threshold grid
rather than a single threshold.

## Defining additional problems

See [Defining a system](defining-a-system.md). The toy problems are deliberately short — `toy.jl`
is about 40 lines — and are the best templates to start from.

## See also

- [Basic usage](basic-usage.md): the runnable script, walked through.
