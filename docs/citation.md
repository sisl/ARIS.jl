# Citation

## The paper

**Amortized Rare-Event Importance Sampling**<br>
Lea M. Hadzic, Harrison Delecki, and Mykel J. Kochenderfer<br>
Stanford University

!!! warning "Not yet published — do not cite a venue, volume, or DOI"
    The manuscript is typeset with the JMLR article template, but its publication fields are
    still unfilled placeholders: the volume is `\#`, the paper number is `21-0000`, the editor
    is `My editor`, and the published date is `M/YY`. **No venue, volume, page range, DOI or
    publication status has been established**, and none should be cited.

    The repository's `README.md` records the same thing: *"Paper citation will be added upon
    publication."* This page will be updated then.

Until then, cite it as an unpublished manuscript:

```bibtex
@unpublished{hadzic2026amortized,
  title  = {Amortized Rare-Event Importance Sampling},
  author = {Hadzic, Lea M. and Delecki, Harrison and Kochenderfer, Mykel J.},
  year   = {2026},
  note   = {Manuscript; publication venue and identifiers not yet assigned}
}
```

!!! note "About the year"
    `2026` is the year carried in the manuscript header, not a publication year. Replace the
    entry entirely once the paper appears.

The title, author list and author order above are taken from
[`paper/manuscript/main.tex`](https://github.com/sisl/ARIS.jl/blob/main/paper/manuscript/main.tex), which is the
canonical manuscript source in this repository.

## The software

To cite the package itself:

```bibtex
@software{aris_jl,
  title   = {{ARIS.jl}: Amortized Rare-Event Importance Sampling},
  author  = {Hadzic, Lea M. and Delecki, Harrison and Kochenderfer, Mykel J.},
  version = {0.1.0},
  url     = {https://github.com/sisl/ARIS.jl},
  note    = {Stanford Intelligent Systems Laboratory}
}
```

`version` follows
[`Project.toml`](https://github.com/sisl/ARIS.jl/blob/main/Project.toml); check it for the
current value.

!!! note "Package metadata lags the paper"
    `Project.toml` records the author as `hdelecki <hdelecki@stanford.edu> and contributors`.
    That string is inherited from the upstream project this library derives from — see
    [below](#provenance) — rather than being an authorship statement for ARIS. The entry above
    uses the manuscript's author list instead.

There is no `CITATION.cff` in this repository, so GitHub's "Cite this repository" button is not
available for it.

## Provenance

The package is **MIT** licensed; see
[`LICENSE`](https://github.com/sisl/ARIS.jl/blob/main/LICENSE).

[`NOTICE`](https://github.com/sisl/ARIS.jl/blob/main/NOTICE) records that the core library in
`src/` and the problem definitions in `examples/problems/` derive from
*ConditionalGenerativeValidation.jl* by Harrison Delecki (MIT, Copyright © 2025), whose package
UUID this repository retains under the module name `ARIS`. `NOTICE` also records the pinned
`StructuredGaussianMixtures.jl` fork used by Experiment 4.

If you redistribute any part of this repository, read `NOTICE` — the MIT grant in `LICENSE` does
not by itself describe the third-party material.

## Reporting results

If you report numbers produced with this package, state:

- the **Julia version** and whether you used the pinned `Manifest.toml`
  ([Installation](install.md)) — results move across dependency sets;
- the **simulator budget** (`n_iter × n_samples`), and for baselines the realised evaluation
  count, since [`ams`](api-estimators.md#ams)'s cost varies by seed;
- the **seeds**, and the spread across them rather than a single run
  ([why](estimating.md#variance-and-getting-an-error-bar));
- which **estimator** you used — the package's proposal-of-origin
  [`is_estimate`](api-estimators.md#is_estimate), or the paper's pooled balance-MIS implemented
  under `paper/` ([the distinction](paper.md#package-versus-paper-code)).

For the paper's own results, see [Reproducing experiments](reproduce.md).
