# Experiments

Experiment code, notebooks, and result data for the Lár-JEPA paper series.
Each subfolder corresponds to one paper and contains its own README with
setup, reproduction steps, and a results summary.

| Folder | Paper | DOI |
|---|---|---|
| [`DAS/`](DAS/README.md) | Divergence Is Not Noise | [10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781) |
| — | Universal Concept Routing — see [`examples/powergrid_full_stack.py`](https://github.com/snath-ai/Lar-JEPA) in the Lár engine repo | [10.5281/zenodo.20278775](https://doi.org/10.5281/zenodo.20278775) |
| [`LTL/`](LTL/README.md), [`continual_learning/`](continual_learning/README.md) | The Lár Training Loop | [10.5281/zenodo.20581128](https://doi.org/10.5281/zenodo.20581128) |
| [`EIM/`](EIM/README.md) | The Encoder Is Not the Memory | [10.5281/zenodo.20583318](https://doi.org/10.5281/zenodo.20583318) |
| [`pav/`](pav/README.md) | Physics Assumption Violations | [10.5281/zenodo.20682615](https://doi.org/10.5281/zenodo.20682615) |
| [`persist/`](persist/README.md) | PERSIST | [10.5281/zenodo.20820042](https://doi.org/10.5281/zenodo.20820042) |
| [`VALIDATION_GAP/`](VALIDATION_GAP/README.md) | Supplementary: Lár vs. LangChain in regulated workflows (not a core paper) | — |
| [`archive/`](archive/README.md) | Superseded result files, kept for provenance | — |

## Structure

Each experiment folder is self-contained: a driver script (or notebook), its
result files, and a README. Where a folder depends on the shared Lár routing
engine, the entry-point script resolves the path back to
`JEPA_Playground/lar_jepa/` at import time — see the individual README for
the exact reproduction command.

`snath-research/experiments/` (a separate, sibling repository — see the LTL
paper's Code and Data Availability section) holds the cross-domain peer-review
pilot referenced by the LTL paper and is not duplicated here.

## License

Apache 2.0, matching the parent repository.
