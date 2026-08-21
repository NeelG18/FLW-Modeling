"""Ablation and hyperparameter-search drivers.

Everything here answers a "does this choice matter?" question and writes a
table to results/. None of it is needed to fit a model -- these are the
experiments that justify the configuration choices in ``pipelines.py``.
"""

import time
import warnings

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_prep import GROUP_COLS, TARGET, RESULTS_DIR
from pipelines import MODEL_NAMES, SEARCH_SPACES, build_pipeline

RANDOM_SPLIT_SEED = 42
TEST_SIZE = 0.20

YEAR_MODES = ["none", "raw", "scaled"]

_METRIC_KEYS = (
    "test_mse", "test_mae", "test_r2",
    "train_mse", "train_mae", "train_r2",
    "delta_r2", "delta_mse", "delta_mae",
)


def make_random_split(df, include_year=True, seed=RANDOM_SPLIT_SEED):
    """The original random 80/20 split, kept for continuity of comparison."""
    cols = GROUP_COLS + (["year"] if include_year else [])
    X = df[cols]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)


def _score_both_sides(pipe, X_train, X_test, y_train, y_test):
    """Score a fitted pipeline on train and test, plus the overfit deltas."""
    y_test_pred = pipe.predict(X_test)
    y_train_pred = pipe.predict(X_train)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    return {
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "train_mse": train_mse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "delta_r2": test_r2 - train_r2,
        "delta_mse": test_mse - train_mse,
        "delta_mae": test_mae - train_mae,
    }


def year_ablation(df, model_names=None, year_modes=None, verbose=True):
    """Compare each model with year dropped, passed raw, and standardised.

    Answers whether the temporal feature earns its place, and whether the
    scaling of that feature is what drives any difference.
    """
    model_names = model_names or MODEL_NAMES
    year_modes = year_modes or YEAR_MODES

    rows = []
    for name in model_names:
        for year_mode in year_modes:
            t0 = time.time()
            X_train, X_test, y_train, y_test = make_random_split(
                df, include_year=(year_mode != "none")
            )

            status = "ok"
            metrics = {k: float("nan") for k in _METRIC_KEYS}

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe = build_pipeline(name, year_mode=year_mode)
                    pipe.fit(X_train, y_train)
                    metrics = _score_both_sides(pipe, X_train, X_test, y_train, y_test)
            except (ValueError, MemoryError) as exc:
                # A configuration that cannot be fitted is a result, not an
                # error to route around. Recording it keeps the comparison
                # honest: substituting a different preprocessing step and
                # still labelling the row with the requested one would
                # report a number the configuration never produced.
                status = f"failed: {exc}"

            elapsed = time.time() - t0
            rows.append(
                dict(model=name, year_mode=year_mode, seconds=elapsed, status=status, **metrics)
            )

            if verbose:
                if status == "ok":
                    print(
                        f"{name:<18}{year_mode:<10}"
                        f"test_mse={metrics['test_mse']:<12.4f}"
                        f"test_r2={metrics['test_r2']:<10.4f}"
                        f"train_r2={metrics['train_r2']:<10.4f}"
                        f"delta_r2={metrics['delta_r2']:<10.4f}"
                        f"{elapsed:.1f}s"
                    )
                else:
                    print(f"{name:<18}{year_mode:<10}DID NOT FIT — {status[8:]}")
        if verbose:
            print()

    return pd.DataFrame(rows)


def run_hyperparameter_search(df, model_name, year_mode="scaled", verbose=True):
    """Re-run the search that selected one model's tuned configuration.

    Kept so the values in ``pipelines.TUNED_PARAMS`` are reproducible rather
    than asserted. Returns the fitted search object.
    """
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"No search space recorded for {model_name!r}")

    spec = SEARCH_SPACES[model_name]
    X_train, X_test, y_train, y_test = make_random_split(
        df, include_year=(year_mode != "none")
    )
    pipe = build_pipeline(model_name, year_mode=year_mode)

    if spec["kind"] == "grid":
        search = GridSearchCV(
            estimator=pipe,
            param_grid=spec["space"],
            cv=spec["cv"],
            scoring="r2",
            n_jobs=-1,
            verbose=1 if verbose else 0,
        )
    else:
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=spec["space"],
            n_iter=spec["n_iter"],
            cv=spec["cv"],
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
            refit="mae",
            n_jobs=-1,
            verbose=1 if verbose else 0,
            random_state=spec["random_state"],
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_train, y_train)

    if verbose:
        print(f"{model_name} best parameters: {search.best_params_}")
        metrics = _score_both_sides(search.best_estimator_, X_train, X_test, y_train, y_test)
        for k, v in metrics.items():
            print(f"  {k} = {v:.6f}")

    return search


def save_table(df, filename):
    """Write an ablation table to results/."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    return path
