"""Ablation and hyperparameter-search drivers.

Everything here answers a "does this choice matter?" question and writes a
table to results/. None of it is needed to fit a model -- these are the
experiments that justify the configuration choices in ``pipelines.py``.
"""

import time
import warnings

import numpy as np
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


# --------------------------------------------------------------------------
# Duplicate-tuple audit
# --------------------------------------------------------------------------
# The models see exactly four features. Any two rows agreeing on all four are
# indistinguishable to them, so when a random split puts one in training and
# its twin in test, the "prediction" is partly recall. The audit below
# measures how much of the random-split score rests on that.
FEATURE_TUPLE = GROUP_COLS + ["year"]


def duplicate_tuple_audit(df, seed=RANDOM_SPLIT_SEED):
    """Quantify repeated feature tuples and how many survive a random split.

    Returns a one-row frame: the duplicate rate in the data, and the share
    of test rows whose exact feature tuple also appears in training.
    """
    counts = df.groupby(FEATURE_TUPLE, sort=False).size()
    duplicated_rows = int((counts[counts > 1]).sum() - (counts > 1).sum())

    X_train, X_test, _, _ = make_random_split(df, include_year=True, seed=seed)
    train_tuples = set(map(tuple, X_train[FEATURE_TUPLE].to_numpy()))
    test_tuples = list(map(tuple, X_test[FEATURE_TUPLE].to_numpy()))
    seen_in_train = sum(1 for t in test_tuples if t in train_tuples)

    return pd.DataFrame([{
        "n_rows": len(df),
        "n_distinct_tuples": int(counts.size),
        "n_rows_sharing_a_tuple": duplicated_rows,
        "pct_rows_sharing_a_tuple": round(100 * duplicated_rows / len(df), 2),
        "largest_tuple_group": int(counts.max()),
        "n_distinct_groups": int(df.groupby(GROUP_COLS, sort=False).ngroups),
        "n_test_rows": len(test_tuples),
        "n_test_rows_seen_in_train": seen_in_train,
        "pct_test_rows_seen_in_train": round(100 * seen_in_train / len(test_tuples), 2),
    }])


def deduplicate_tuples(df):
    """Collapse rows sharing a feature tuple, averaging the target.

    Averaging rather than dropping keeps every observation's contribution;
    what changes is that a tuple can no longer appear on both sides of a
    split.
    """
    return (
        df.groupby(FEATURE_TUPLE, as_index=False, sort=False)[TARGET]
        .mean()
    )


def duplicate_removal_experiment(df, model_names=None, seed=RANDOM_SPLIT_SEED, verbose=True):
    """Score a random split before and after collapsing duplicate tuples.

    Collapsing raises the scores rather than lowering them: averaging the
    conflicting targets of a repeated tuple removes noise the model could
    never have fitted. See ``seen_vs_unseen_tuples`` for the test that
    separates that effect from memorisation.
    """
    model_names = model_names or ["Random Forest", "KNN", "Decision Tree", "Ridge"]
    deduped = deduplicate_tuples(df)

    rows = []
    for label, frame in (("with duplicates", df), ("deduplicated", deduped)):
        X_train, X_test, y_train, y_test = make_random_split(
            frame, include_year=True, seed=seed
        )
        for name in model_names:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe = build_pipeline(name, year_mode="scaled")
                pipe.fit(X_train, y_train)
                metrics = _score_both_sides(pipe, X_train, X_test, y_train, y_test)
            rows.append(dict(dataset=label, model=name, n_rows=len(frame), **metrics))
            if verbose:
                print(f"  {label:<16}{name:<18}test_r2={metrics['test_r2']:.4f}")

    out = pd.DataFrame(rows)
    wide = out.pivot(index="model", columns="dataset", values="test_r2")
    wide["inflation"] = (wide["with duplicates"] - wide["deduplicated"]).round(4)
    return out, wide


def seen_vs_unseen_tuples(df, model_names=None, seed=RANDOM_SPLIT_SEED, verbose=True):
    """Score test rows by whether their feature tuple appeared in training.

    This is the direct test for memorisation across a random split. If
    repeated tuples let a model recall training targets, the rows whose
    tuple it has already seen should be the easy ones. Holding the fit and
    the split fixed and partitioning only the test set keeps everything
    else constant.

    Reports MSE as well as R-squared: the two subsets have different target
    variances, so R-squared alone does not compare them cleanly.
    """
    model_names = model_names or ["Random Forest", "KNN", "Decision Tree", "Ridge"]
    X_train, X_test, y_train, y_test = make_random_split(df, include_year=True, seed=seed)

    train_tuples = set(map(tuple, X_train[FEATURE_TUPLE].to_numpy()))
    seen = np.array([tuple(r) in train_tuples for r in X_test[FEATURE_TUPLE].to_numpy()])

    rows = []
    for name in model_names:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = build_pipeline(name, year_mode="scaled")
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

        rows.append({
            "model": name,
            "n_seen": int(seen.sum()),
            "n_unseen": int((~seen).sum()),
            "mse_seen": mean_squared_error(y_test[seen], preds[seen]),
            "mse_unseen": mean_squared_error(y_test[~seen], preds[~seen]),
            "r2_seen": r2_score(y_test[seen], preds[seen]),
            "r2_unseen": r2_score(y_test[~seen], preds[~seen]),
            "target_var_seen": float(y_test[seen].var()),
            "target_var_unseen": float(y_test[~seen].var()),
        })
        if verbose:
            r = rows[-1]
            print(f"  {name:<18}MSE seen={r['mse_seen']:.4f}  unseen={r['mse_unseen']:.4f}")

    return pd.DataFrame(rows)
