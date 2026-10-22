# Installation

## Requirements

- **Julia 1.12** or later. The pinned `Manifest.toml` was resolved with Julia 1.12.6; continuous
  integration runs the test suite on 1.12 and 1.13.
- A C toolchain is not required; all dependencies are pure Julia or ship binaries through
  Julia's artifact system.

## Install

!!! warning "Not in the General registry"
    `ARIS` is not registered, so `Pkg.add("ARIS")` will not find it. Install from a clone.

```bash
git clone https://github.com/sisl/ARIS.jl
cd ARIS.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`Pkg.instantiate()` installs the exact versions recorded in `Manifest.toml`. That is the
environment the paper's results were produced in, and it is what you want for reproduction. To
work against freshly resolved dependencies instead, delete `Manifest.toml` before instantiating
— and expect numerical results to move.

## Check the install

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The suite covers the public API surface, the utility functions, the buffer, the conditional
models, the sampling strategies, the estimator, and the baselines. It does not run the paper
experiments.

A quick end-to-end check, which takes a couple of minutes:

```bash
julia --project=. examples/basic_usage.jl
```

## Using it from another project

Add the clone as a development dependency:

```julia
using Pkg
Pkg.develop(path = "/path/to/ARIS.jl")
```

Then `using ARIS` as normal. Note that the example *problems* under `examples/problems/` are not
part of the package module — they are plain scripts you `include`, so that they can depend on
things (plotting, driving environments) that the package itself does not.

## Optional integrations

Two integrations are gated off by default and are **not** used by any paper experiment:

| gate | environment variable | brings in |
|---|---|---|
| normalizing flows | `CGV_ENABLE_FLOW=true` | `PyCall`, and Python `torch` + `normflows` |
| policy / MDP proposals | `CGV_ENABLE_POLICY=true` | the `RMDP` and conditional-policy code paths |

These are read at module load, so set them before `using ARIS`. Their dependencies are not in
`Project.toml`; you must provide them in the active environment yourself. Leave both unset unless
you specifically need them.

## Reproducing the paper

Experiment 4 uses a **separate** environment, because it pulls a pinned Git dependency:

```bash
julia --project=paper/envs/idm -e 'using Pkg; Pkg.instantiate()'
```

The figure and metric scripts are Python and need `numpy` and `matplotlib`. See
[Reproducing experiments](reproduce.md).
