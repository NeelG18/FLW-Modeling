"""Dataset loading and feature/target definitions for the FLW models.

Single source of truth for how the raw UN FLW extract is turned into the
modelling frame. Every experiment module imports from here so that a change
to the filtering rule propagates everywhere at once.
"""

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "flwData2.csv"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

# --------------------------------------------------------------------------
# Column roles
# --------------------------------------------------------------------------
TARGET = "loss_percentage"

# The three categorical predictors. Named GROUP_COLS because they also define
# the unit that the persistence and linear-trend baselines forecast within.
GROUP_COLS = ["country", "commodity", "food_supply_stage"]

# Full predictor set used by the models that take a temporal feature.
FEATURES = GROUP_COLS + ["year"]

# Sparse tail: the UN database has roughly 1,000-1,700 rows/year from
# 2000-2021, then 147 / 44 / 144 rows for 2022 / 2023 / 2024. That is a
# reporting lag in the source database rather than a collapse in food loss,
# so validation defaults to the dense span.
DENSE_YEAR_MAX = 2021


def load_flw_data(path=DATA_PATH, drop_missing_stage=True, year_max=None):
    """Load the FLW extract and apply the standard filtering.

    Parameters
    ----------
    path : path-like
        Location of the CSV extract.
    drop_missing_stage : bool
        Drop rows with no ``food_supply_stage``. This is the filter the
        models have always used: the stage is a required predictor, so rows
        lacking it cannot be scored.
    year_max : int or None
        If given, keep only rows with ``year <= year_max``. Pass
        ``DENSE_YEAR_MAX`` to exclude the sparse 2022-2024 tail.

    Returns
    -------
    pandas.DataFrame
    """
    df = pd.read_csv(path, low_memory=False)

    if drop_missing_stage:
        df = df.dropna(subset=["food_supply_stage"])

    if year_max is not None:
        df = df[df["year"] <= year_max]

    return df.copy()


def describe_dataset(df):
    """Return the dataset facts that the methods section has to report.

    Returns a dict rather than printing so callers can serialise it.
    """
    return {
        "n_rows": int(len(df)),
        "n_countries": int(df["country"].nunique()),
        "n_commodities": int(df["commodity"].nunique()),
        "n_supply_stages": int(df["food_supply_stage"].nunique()),
        "n_cpc_codes": int(df["cpc_code"].nunique()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "target_mean": float(df[TARGET].mean()),
        "target_median": float(df[TARGET].median()),
        "target_max": float(df[TARGET].max()),
    }


def split_features_target(df, include_year=True):
    """Split a frame into the model matrix and the target vector."""
    cols = FEATURES if include_year else GROUP_COLS
    return df[cols], df[TARGET]
