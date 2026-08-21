# Changelog

All notable changes to the analysis code are recorded here.

## [Unreleased]

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
