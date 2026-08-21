# Changelog

All notable changes to the analysis code are recorded here.

## [Unreleased]

### Fixed

- Polynomial ridge memory guard sized the degree-3 expansion as a dense
  array, estimating 1,379-1,797 GB and skipping the model at every
  validation cutoff. The encoded matrix is sparse with about four
  non-zeros per row, so the expansion holds about thirty-four non-zeros
  per row and costs roughly 9 MB. The guard now measures the non-zero
  rate on a sample and extrapolates from it, and reports which layout it
  costed. Polynomial ridge fits all five cutoffs in about two seconds and
  is restored to the results table.
- Validation summary counted attempted cutoffs rather than scored runs,
  so a model skipped at every cutoff still reported a full count beside
  empty metrics.
- Notebook no longer references pipeline variables that were never
  defined. `neural_pipeline` and `dtpipeline` are now built and fitted
  through `pipelines.build_pipeline`, so the notebook runs from a clean
  kernel instead of depending on state from a prior session.
- Axis-limit calculation in the supply-stage and commodity comparison
  plots raised `ValueError: min() arg is an empty sequence` when the
  requested year had no observations, because every actual value was
  missing and the empty series was passed to `min()`. Limits are now
  computed over the finite values across both series, with a fallback
  when none exist. This affected both plotting functions, not just the
  one that surfaced it.

### Changed

- Year ablation records configurations that cannot be fitted instead of
  aborting the run. The multi-layer perceptron does not converge with an
  unscaled `year` feature (the solver produces non-finite weights), which
  is now reported as a failed fit with its reason. Previously this case
  was given a standardised `year` while still being labelled as the
  unscaled configuration, so the table reported a figure that
  configuration never produced.

- Visualization helpers moved from the notebook into
  `src/visualization.py`, leaving the notebook as orchestration only.
- The separate year-aware and year-agnostic prediction helpers are
  replaced by single functions taking an optional `year`. The duplicated
  pair was the reason the missing-year fix had to be applied twice.
- Notebook outputs are not stored; generated tables are written to
  `results/` instead.

### Added

- Repository structure: `src/` package holding the analysis modules,
  `data/` for the UN FLW extract, `results/` and `figures/` for generated
  output, and `notebooks/` for orchestration.
- `src/data_prep.py`: dataset loading, the missing-supply-stage filter, an
  optional dense-year-span filter, and the dataset summary counts.
- `src/pipelines.py`: all eight model configurations and the
  hyperparameter search spaces they were selected from, as the single
  source of truth. No other module instantiates a regressor.
- `src/baselines.py`: group-level persistence and per-group linear time
  trend forecasts, each reporting its fallback rate.
- `src/validation.py`: walk-forward time-aware validation across multiple
  training cutoffs, with per-row predictions persisted alongside the
  aggregate metrics.
- `src/ablations.py`: year-treatment ablation and hyperparameter search
  drivers.
- `requirements.txt` pinning the scientific stack.
