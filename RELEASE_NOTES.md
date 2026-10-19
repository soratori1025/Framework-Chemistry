# Release notes

## v1.0.0 — manuscript release (paper 1)

This release is the code and results archive for the manuscript

> *Evidence-Guided Molecular Property Modelling with a Configuration-Driven Modular Framework*
> (target journal: *Journal of Cheminformatics*)

### What is in this release

* The `chemistry-framework` package (features, encoders, training/evaluation,
  dataset advisor, feature-evidence scorecard and recommender).
* The benchmark runners that produced every table and figure of the manuscript,
  including the Chemprop v2.3.1 external comparison.
* **Archived per-split results** under `benchmarks/results/` (43 files) and the
  20 Bemis-Murcko scaffold manifests per endpoint under `benchmarks/splits/`, so
  the reported statistics can be recomputed without retraining.
* Statistics tooling:
  * `benchmarks/paper_statistics.py` — recomputes every paired comparison with
    the Nadeau-Bengio corrected resampled *t* interval and Holm-Bonferroni
    multiplicity control; writes `benchmarks/results/paper_statistics.json`.
  * `benchmarks/generate_si_tables.py` — renders the Additional-file tables from
    those comparisons.
  * `benchmarks/validate_paper_statistics.py` — regression check that the
    archived results reproduce the 30 transcribed published values.
* `REPRODUCING.md` — environment pinning, per-artifact commands, and an
  explicit list of what reproduces exactly and what does not.

### Changes relative to the pre-release state of the repository

* **Scope.** The follow-up feature-space routing work (SIREN) is **not** part of
  this release; it lives on the `siren-dev` branch and is reserved for the
  second paper.
* **Statistics.** The first draft of the manuscript used the naive paired *t*
  interval over five splits. This release adds the corrected resampled *t*
  interval (Nadeau-Bengio: variance inflated by `1/k + n_test/n_train`) and
  Holm-Bonferroni control within each table's comparison family. The
  consequence is that several differences that were reported as conclusive in
  the first draft are now reported as inconclusive; the corrected intervals are
  the intervals printed in the manuscript.
* **Ten-split capability.** Every runner accepts `--seeds`, so the study can be
  repeated on any subset of the twenty prepared manifests with one extra flag
  (see `REPRODUCING.md`, section 6). The statistics scripts read the number of
  splits from the archive, so the intervals, degrees of freedom, and Holm
  adjustment follow automatically.
* **Structure audit.** `benchmarks/run_structure_audit.py` regenerates the
  duplicate/conflict inventory (`duplicate_audit_report.json`,
  `duplicate_audit_summary.csv`, `bbbp_duplicate_groups.csv`) and the
  range-based OOD counts (`ood_shift_all_splits.json`) that previously had no
  committed generator.
* **Documentation.** `README.md` now points at the manuscript results and the
  reproduction instructions; the stale development ADMET table was removed.

### Release checklist before minting the Zenodo DOI

1. Fill in the author names and affiliations:
   * `.zenodo.json` → `creators`
   * `CITATION.cff` → `authors`
   * `README.md` / `REPRODUCING.md` contact links, if the repository owner
     changes.
2. Confirm the license holder in `LICENSE` matches the manuscript.
3. Push the release branch to GitHub (`main` after the fast-forward merge).
4. Enable the repository in Zenodo (*Settings → GitHub → Zenodo*), then create a
   GitHub release tagged `v1.0.0`. Zenodo reads `.zenodo.json` for the metadata
   and mints the DOI automatically.
5. Copy the DOI badge/URL into:
   * `README.md` (data-availability section),
   * the manuscript's *Availability of data and materials* statement,
   * `CITATION.cff` (`doi:` field) and `.zenodo.json`
     (`related_identifiers`, relation `isIdenticalTo`).
6. Optional: add the conceptual DOI (the one Zenodo shows for "all versions")
   to `REPRODUCING.md` for citations of "the software" rather than
   "this version".

### Known limitations recorded at release time

* The archived results were produced with **RDKit 2023.09.6**; the 210-descriptor
  block is generated from `rdkit.Chem.Descriptors.descList`, which exposes 217
  entries in RDKit 2026.09.1. Rerunning descriptor-based steps with a different
  RDKit version changes the feature block. Pin the version (see
  `REPRODUCING.md`, section 2).
* The statistics are computed on **five splits**; the minimum detectable
  differences at 80% power are reported in the manuscript. Ten- and
  twenty-split reruns are supported by the code but were not part of the
  archived results.
* Neural runs are seeded per split but are not bit-for-bit deterministic on GPU.
