# Predicting Global Food Loss and Waste with Regression Models

Comparative regression modelling of food loss and waste (FLW) percentages
using the United Nations FLW database, with time-aware validation against
naive forecasting baselines.

## Data

`data/flwData2.csv` is an extract from the UN/FAO Food Loss and Waste
Database. After dropping rows with no recorded `food_supply_stage`, the
modelling frame holds:

| | |
|---|---|
| Rows | 26,970 |
| Countries | 159 |
| Commodities | 203 |
| Supply-chain stages | 20 |
| CPC codes | 186 |
| Year range | 2000–2024 |

Validation defaults to the dense span (2000–2021, 26,635 rows). The
2022–2024 tail holds only 147 / 44 / 144 rows against roughly 1,000–1,700
rows per year before it, which reflects reporting lag in the source
database rather than a change in food loss.

## Layout

```
src/
  data_prep.py   loading, filtering, feature/target definitions
  pipelines.py   all eight model configurations (single source of truth)
  baselines.py   persistence and linear-trend forecasts
  validation.py  walk-forward time-aware validation
  ablations.py   year ablation, hyperparameter searches
notebooks/
  flwNotebook.ipynb   orchestration; imports from src/
results/         generated tables (CSV)
figures/         generated plots (PNG)
```

No module outside `pipelines.py` may instantiate a regressor directly, so
each hyperparameter is defined in exactly one place and cannot drift
between experiments.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Pinned versions matter: several models (KNN, tree ensembles) shift in the
third decimal place across library versions, so reported numbers are only
reproducible against the pinned stack.

## Running

```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from data_prep import load_flw_data, DENSE_YEAR_MAX; from validation import walk_forward, summarise, save_results; df=load_flw_data(year_max=DENSE_YEAR_MAX); r,p=walk_forward(df); save_results(r,p,summarise(r))"
```

The walk-forward writes three files to `results/`: per-cutoff metrics, the
summary table, and every per-row prediction. Predictions are kept so that
paired significance tests and residual plots can be recomputed without
refitting.

## Models

Eight regressors are compared: linear regression, ridge, polynomial ridge,
k-nearest neighbours, support vector regression, decision tree, random
forest, and a multi-layer perceptron. Each is scored against two naive
baselines — group-level persistence and a per-group linear time trend.
