# Changelog

All notable changes to the analysis code are recorded here.

## [Unreleased]

### Changed

- Charts no longer draw a heading above the plotting area. The heading
  restated the caption placed beneath the figure, so the same sentence
  appeared twice wherever a figure was used. Panel headings inside
  multi-panel figures are kept, since they identify which axes is which.
  `plot_food_loss_comparison` gains a `compact` flag matching the one the
  other plotting helpers already use, so callers that want the heading
  keep it.

### Fixed

- Distribution figure hardcoded a share that contradicted the bars beside
  it: the value quoted was the share among held-out validation records
  while the chart showed the full modelling frame. The share is now
  derived from the data plotted.
- Predicted-against-actual figure reported R-squared and mean absolute
  error pooled over every held-out record, which does not equal the mean
  across splits reported in the results table. The basis is recorded so
  the two are not read as the same quantity.
- Error concentration figure reported its shares to the nearest whole
  number, which did not match the text quoting one decimal place.
- Regional weighting analysis was being run on the full modelling frame
  although it splits on time, so its test folds included the sparse
  2022-2024 years that the validation protocol excludes. It now uses the
  same dense span as the other time-splitting procedures.
- Figures no longer average across models where the accompanying text
  quotes one. The representation figure showed a mean over three models
  while the text reported the random forest, so the two disagreed.
- Hyperparameter searches ran on a random split spanning every year while
  evaluation held out later years, so configurations were selected with
  the test period present in the search data. Selection now sees only
  years at or before each cutoff, and the inner folds are time-ordered
  expanding windows rather than random. The severity threshold used by
  the ranking objective is likewise derived from the training window
  alone. Effect on results is small: mean absolute error moves by at most
  0.036 for five of six models, and improves by 0.163 for the decision
  tree.
- Search spaces recorded for polynomial ridge, which previously had none,
  and extended for the random forest to include `bootstrap` and
  `max_features`, which the configuration set without any search.
- Persistence baseline no longer depends on row order. It selected a
  group's most recent training observation by sorting on year and taking
  the last row, but about a third of rows share their full feature tuple
  with another row, so a group's latest year commonly holds several
  observations and the sort picked an arbitrary member of the tie. It now
  averages across every observation at the group's latest training year.
  The baseline scores substantially better as a result.
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

- Figures write vector copies alongside the raster ones, with text kept as
  text rather than converted to outlines, so a label can be adjusted in a
  drawing program without regenerating the figure.
- Descriptive counts are taken from the full modelling frame and
  validation results from the dense span, consistently. Both were
  previously in use for the same quantities in different places.
- Comparison plotting consolidated into a single function taking the
  dimension to vary as an argument, with the remaining two held fixed. The
  previous per-dimension functions duplicated the same logic, which is why
  an earlier fix had to be applied in more than one place. The
  consolidated form also supports varying by country, which none of the
  originals did.
- Flagship model for the prediction and visualization section changed from
  the decision tree to the random forest. The random forest leads on
  unseen years (0.356), on unseen country-commodity pairs (0.389), and on
  the random split (0.645); the decision tree scores -0.084 on unseen
  years, below a constant predictor and below the persistence baseline.
- One-hot encoding no longer drops a category level for the distance,
  tree, kernel, and neural models. The encoder ignores unknown categories,
  so an unseen category encodes as all zeros -- identical to how a dropped
  reference level encodes. The two were indistinguishable, and a country
  absent from training was silently treated as the reference country.
  Dropping is retained for the three least-squares models, where it
  removes the exact collinearity between the indicator columns and the
  intercept; those models accept the ambiguity knowingly and the reason is
  recorded next to the policy.
- Results regenerated under the corrected encoding. Scores for the
  affected models fall slightly: dropping a level left reference-level
  rows with one fewer non-zero, which pulled them toward the origin and
  distorted nearest-neighbour distances in particular. The three
  least-squares models are unchanged to the last digit, confirming the
  change is scoped as intended.
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

- Figure and table generation module producing every numbered figure and
  table from the result tables rather than by hand, so a figure cannot
  drift from the finding it depicts. Sixteen figures and ten tables, with
  a manifest recording the topic group each belongs to. The two method
  illustrations are drawn from the fitted models: the network architecture
  as configured, and the first three levels of the fitted regression tree.
- Ranking and screening evaluation: average precision across several
  severity thresholds, rank correlation, precision/recall/lift at k, and
  share of observed loss captured at k against a perfect-ranking ceiling.
  Average precision is preferred to ROC-AUC because at roughly 2%
  prevalence the latter is dominated by easy negatives. Severity is set at
  the top decile of observed losses rather than a round number.
- Paired bootstrap on the difference in average precision between a model
  and a reference, resampled within each split so class balance is
  preserved.
- Stage comparisons exclude the values of `food_supply_stage` that span
  several chain positions rather than naming one. A span accumulates loss
  across everything it covers, so including it decides the comparison by
  construction.
- Stage coverage reports the plausible gap alongside the naive one. Only
  counting stages at which a commodity is measured somewhere reduces the
  apparent gap from 24,952 combinations to 9,185.
- Adjusted R-squared, a skill score against a reference predictor, and a
  helper reporting the encoded width of a fitted preprocessor.
- Nested hyperparameter selection under the walk-forward protocol, with
  two objectives and results for both.
- Sensitivity analysis for the excluded `cause_of_loss` feature. The
  field is profiled (coverage, distinct values, value lengths) and then
  tested four ways over repeated random splits: with and without the
  feature, on all rows with missing values filled as an explicit Unknown
  level, and on the subset where the field is actually recorded. Including
  it changes the full-data score by about 0.002, and the apparent gain on
  the recorded subset sits inside the run-to-run standard deviation.
- Country representation summary and an inverse-frequency weighting
  analysis, scored separately for sparsely observed and well represented
  countries under the walk-forward split. Weighting degrades mean absolute
  error for every model tested, including for the sparse countries it is
  intended to help.
- Residual diagnostics computed from held-out predictions: predicted
  against actual with the identity line, residuals against fitted values,
  error broken down by country or commodity, and error broken down by the
  size of the observed loss.
- Per-row predictions now carry country, commodity, and supply stage, so
  error can be attributed to a group without refitting.
- Random forest ablation over `bootstrap`, `max_features`, and
  `max_depth`, scored on the walk-forward split rather than a random one.
  These three settings were fixed in the configuration without appearing
  in any search. All 32 combinations are now recorded.
- Comparison of free-text commodity labels against standardised CPC
  codes, run through the identical walk-forward protocol so the encoding
  is the only difference, plus a count of the categories that free text
  splits and the code does not.
- Paired significance tests comparing every model against the baseline
  within each split, on both squared and absolute error. Reporting a mean
  and standard deviation across folds conflates model differences with
  fold difficulty, and the second dominates here: the baseline's mean
  squared error ranges from about 22 at the earliest cutoff to about 8 at
  the latest. Scoring both predictors on identical rows and taking the
  per-row difference cancels that. Reported as a percentile bootstrap
  interval over 10,000 resamples plus a Wilcoxon signed-rank test, and a
  model is credited only where the whole interval falls below zero at
  every split.
- Grouped cross-validation holding out whole country-commodity pairs,
  reported alongside the walk-forward split. The two answer different
  questions: whether a model reaches a later year, and whether it reaches
  a country and commodity combination absent from training. Both
  baselines forecast within a group that never appears in training under
  this split, so they fall back completely and collapse to a global
  constant; their fallback rate is reported to make that explicit.
- Duplicate feature-tuple audit reporting the repeat rate, the largest
  repeated group, and the share of test rows whose exact feature tuple
  also appears in training under the random split.
- Experiment scoring a random split before and after collapsing repeated
  tuples, and a partition of the test set by whether each row's tuple was
  present in training. Together these test whether repeated tuples let a
  model recall training targets across a random split. They do not:
  already-seen rows carry higher error, because repeated tuples hold
  conflicting targets and so act as a noise floor rather than a shortcut.
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
