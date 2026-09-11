# ARIS.jl

[![CI](https://github.com/sisl/ARIS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sisl/ARIS.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/sisl/ARIS.jl/graph/badge.svg)](https://codecov.io/gh/sisl/ARIS.jl)
[![Julia](https://img.shields.io/badge/Julia-1.12-9558B2?logo=julia&logoColor=white)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Amortized Rare-Event Importance Sampling (ARIS)** learns a shared proposal
model that can be reused across multiple failure thresholds. By jointly modeling
system inputs and robustness values, ARIS constructs threshold-specific
proposals by truncating the learned robustness distribution.

## Installation

Clone the repository and instantiate the Julia environment:

```bash
git clone https://github.com/sisl/ARIS.jl
cd ARIS.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`Manifest.toml` pins the dependency set used for the paper with Julia 1.12.6.

## Basic Usage

```julia
using ARIS
using Distributions, Random

include("examples/problems/toy.jl")

Random.seed!(1)
sys = UnimodalToy(3.0)

model = JointModel(
    Normal(),
    ConditionalEMGMM(2, 3, 3),
)

cv = ConditionalValidation(
    model = model,
    xdim = 2,
    depth = 1,
    sdim = 2,
    n_samples = 1000,
    n_iter = 25,
    sampling_strategy = QuantileSampling(0.2; target = 0.0),
)

train!(cv, sys)

is_estimate(cv.buffer; threshold = 0.0)
```

## API Overview

A validation problem is a `System`; `ConditionalValidation` and `train!` adapt a
proposal to it, and `is_estimate` turns the samples into a failure-probability
estimate. Failure means robustness ρ ≤ 0 unless another threshold is given.

- `System` — problem interface: subtype `System.SystemParameters` and extend
  `prior`, `simulate`, `evaluate`, and `get_xdim` / `get_sdim` / `get_depth`.
- `ConditionalValidation(; model, xdim, depth, sdim, n_samples, n_iter, sampling_strategy)`
  — an adaptive run; evaluated samples and weights are stored in `cv.buffer`.
- `train!(cv, sys)` — runs the adaptive sampling loop.
- `QuantileSampling(q; target = 0.0)` — moves the training threshold toward
  `target` using the `q`-quantile of each batch's robustness values.
- `JointModel(robustness_model, conditional_model)` — the ARIS proposal: a
  distribution over robustness (e.g. `Normal()`) and a model of inputs given it.
- `ConditionalEMGMM(k, xdim + 1, xdim + 1)` — `k`-component Gaussian-mixture
  conditional model over inputs and robustness.
- `ConditionalGaussian(xdim)` — single-Gaussian conditional model for `JointModel`.
- `is_estimate(buffer; threshold = 0.0)` — importance-sampling estimate of
  P(ρ ≤ `threshold`) from a training buffer.

Baselines, each targeting ρ ≤ 0:

- `CrossEntropyMethod(; model, xdim, depth, sdim, n_samples, n_iter)` — unconditional
  cross-entropy method; train with `train!` and estimate with `is_estimate`.
- `ams(sys)` — adaptive multilevel splitting; returns the estimate and per-level samples.
- `pmc(sys; dₓ)` — population Monte Carlo with Gaussian kernels in `dₓ` dimensions.

The multi-threshold estimator used for the paper experiments is implemented
under [`paper/`](paper/) rather than exposed as part of the package API.

### Defining a custom system

```julia
using LinearAlgebra

# ρ(x) = β − (x₁ + x₂)/√2 with x ~ N(0, I₂)
struct Linear2D <: System.SystemParameters
    β::Float64
end
System.prior(::Linear2D) = MvNormal(zeros(2), I)
System.get_xdim(::Linear2D) = 2
System.get_sdim(::Linear2D) = 2
System.get_depth(::Linear2D) = 1
# batched form used by `train!`: inputs are columns, states are sdim × depth × N
System.simulate(::Linear2D, X::AbstractMatrix) = reshape(X, 2, 1, size(X, 2))
System.evaluate(s::Linear2D, S::AbstractArray{<:Real,3}) = s.β .- (S[1, 1, :] .+ S[2, 1, :]) ./ √2
# per-sample form used by `ams` and `pmc`
System.simulate(::Linear2D, x::AbstractVector) = x
System.evaluate(s::Linear2D, x::AbstractVector) = s.β - (x[1] + x[2]) / √2
```

Use it in place of the toy system in Basic Usage, e.g. `train!(cv, Linear2D(3.5))`;
the exact answer is Φ(−3.5) ≈ 2.3 × 10⁻⁴.

## Examples

[`examples/basic_usage.jl`](examples/basic_usage.jl) provides a runnable version
of the example above.

[`examples/problems/`](examples/problems/) contains four `System`
implementations:

- `toy.jl`
- `radial_shell.jl`
- `tailrace.jl`
- `crosswalk.jl`

The crosswalk example uses the additional environment described in
[`paper/README.md`](paper/README.md).

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Reproducing the Paper

Paper-specific experiment drivers, configurations, seed-level results, metric
and diagnostic scripts, figure generators, and figures live under
[`paper/`](paper/).

See [`paper/README.md`](paper/README.md) for reproduction instructions,
protocol notes, result files, and the raw-state policy.

## Citation

Paper citation will be added upon publication.

## License

MIT. See [`LICENSE`](LICENSE).

[`NOTICE`](NOTICE) records the provenance of derived code and of the pinned
StructuredGaussianMixtures.jl dependency used by Exp4.
