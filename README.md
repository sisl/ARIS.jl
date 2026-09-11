# ARIS.jl

[![CI](https://github.com/lea-m-hadzic/ARIS.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/lea-m-hadzic/ARIS.jl/actions/workflows/CI.yml)
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

The main package components are:

- `System` — interface for defining a validation problem;
- `JointModel`, `ConditionalEMGMM`, and `ConditionalGaussian` — proposal models;
- `ConditionalValidation` and `train!` — adaptive proposal training;
- `is_estimate` — single-threshold importance-sampling estimation;
- `ams`, `pmc`, and `CrossEntropyMethod` — baseline methods.

The multi-threshold estimator used for the paper experiments is implemented
under [`paper/`](paper/) rather than exposed as part of the package API.

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
