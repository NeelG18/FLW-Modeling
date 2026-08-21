"""Nested hyperparameter selection under the walk-forward protocol.

The earlier searches tuned on a random split spanning every year, then the
walk-forward evaluated on held-out years. The model weights never saw the
future, but the *configuration* was chosen with the test years in the search
data, which is selection leakage and makes the reported scores optimistic.

Here the search for a given cutoff sees only years at or before that cutoff.
The inner folds are themselves time-ordered expanding windows rather than
random folds, so a configuration cannot be chosen on its ability to
interpolate within the training window either.

Two objectives are supported because the work makes two claims:

``mae``
    For estimating a loss percentage. Absolute error spreads weight across
    the whole distribution rather than concentrating it in the rare large
    values.

``average_precision``
    For ranking cells so that severe losses surface first. Binarised at the
    top decile of observed losses, which is where the models beat the naive
    baseline and where enough positives remain for a stable estimate.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from data_prep import GROUP_COLS, TARGET, RESULTS_DIR
from pipelines import SEARCH_SPACES, build_pipeline

# The top decile of observed loss percentages. Recomputed from the training
# window at fit time; this is the fallback and the value used for reporting.
SEVERITY_QUANTILE = 0.90

TRACTABLE_MODELS = [
    "Random Forest", "Poly Ridge", "Ridge", "KNN", "Decision Tree", "Linear Regression",
]


def average_precision_at(y_true, y_pred, threshold):
    """Average precision for identifying losses at or above ``threshold``.

    Returns NaN when a fold contains no positives, so the search skips it
    rather than scoring an undefined case as zero.
    """
    y_bin = (np.asarray(y_true) >= threshold).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == y_bin.size:
        return np.nan
    return average_precision_score(y_bin, y_pred)


def make_objective(name, threshold):
    """Return a scikit-learn scorer for one of the two objectives."""
    if name == "mae":
        return "neg_mean_absolute_error"
    if name == "average_precision":
        return make_scorer(
            average_precision_at, greater_is_better=True, threshold=threshold
        )
    raise ValueError(f"Unknown objective: {name!r}")


def expanding_year_folds(years, n_splits=3):
    """Time-ordered inner folds: train on earlier years, validate on later.

    Random folds inside the training window would let a configuration be
    chosen for interpolating between years it will not have at prediction
    time, which is the same mistake the outer protocol exists to avoid.
    """
    years = np.asarray(years)
    distinct = np.sort(np.unique(years))
    if len(distinct) <= n_splits:
        n_splits = max(1, len(distinct) - 1)

    boundaries = distinct[-n_splits:] if n_splits else []
    folds = []
    for boundary in boundaries:
        train_idx = np.flatnonzero(years < boundary)
        val_idx = np.flatnonzero(years == boundary)
        if len(train_idx) and len(val_idx):
            folds.append((train_idx, val_idx))
    return folds


def nested_search(df, cutoff_years, model_names=None, objectives=("mae", "average_precision"),
                  n_iter=20, random_state=0, verbose=True, n_jobs=2,
                  checkpoint_prefix=None):
    """Tune within each training window, then score on the held-out years.

    Parameters
    ----------
    n_jobs : int
        Parallel workers per search. Kept small deliberately: the degree-3
        polynomial expansion is large enough that one copy per worker
        exhausts memory on the widest training windows.
    checkpoint_prefix : str, optional
        Write results after each cutoff completes. A run of this length
        should not lose four finished cutoffs because the fifth failed.

    Returns ``(results_df, predictions_df)``: one result row per
    (cutoff, model, objective) with the chosen parameters and held-out
    metrics, plus the per-row predictions for later paired testing.
    """
    model_names = model_names or TRACTABLE_MODELS
    results, predictions = [], []

    for cutoff in cutoff_years:
        train_df = df[df["year"] <= cutoff]
        test_df = df[df["year"] > cutoff]
        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[GROUP_COLS + ["year"]]
        y_train = train_df[TARGET]
        X_test = test_df[GROUP_COLS + ["year"]]
        y_test = test_df[TARGET].to_numpy()

        # Threshold from the training window only -- deriving it from the
        # full data would leak the test distribution into the objective.
        threshold = float(y_train.quantile(SEVERITY_QUANTILE))
        folds = expanding_year_folds(train_df["year"].to_numpy())

        for name in model_names:
            spec = SEARCH_SPACES.get(name)
            if spec is None:
                continue

            for objective in objectives:
                scoring = make_objective(objective, threshold)
                pipe = build_pipeline(name, year_mode="scaled")

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if spec["kind"] == "grid":
                        search = GridSearchCV(pipe, spec["space"], cv=folds,
                                              scoring=scoring, n_jobs=n_jobs,
                                              error_score=np.nan)
                    else:
                        search = RandomizedSearchCV(
                            pipe, spec["space"], n_iter=n_iter, cv=folds,
                            scoring=scoring, n_jobs=n_jobs,
                            random_state=random_state, error_score=np.nan)
                    search.fit(X_train, y_train)
                    preds = search.best_estimator_.predict(X_test)

                y_bin = (y_test >= threshold).astype(int)
                results.append({
                    "cutoff": cutoff, "model": name, "objective": objective,
                    "threshold": round(threshold, 3),
                    "mae": mean_absolute_error(y_test, preds),
                    "average_precision": average_precision_at(y_test, preds, threshold),
                    "n_test": len(test_df), "n_positives": int(y_bin.sum()),
                    "best_params": str(search.best_params_),
                })
                frame = test_df[GROUP_COLS + ["year"]].reset_index(drop=True).copy()
                frame["cutoff"] = cutoff
                frame["model"] = name
                frame["objective"] = objective
                frame["y_true"] = y_test
                frame["y_pred"] = preds
                predictions.append(frame)

                if verbose:
                    r = results[-1]
                    print(f"  cutoff {cutoff}  {name:<18}{objective:<18}"
                          f"mae={r['mae']:.4f}  ap={r['average_precision']:.4f}", flush=True)

        if checkpoint_prefix:
            RESULTS_DIR.mkdir(exist_ok=True)
            pd.DataFrame(results).to_csv(
                RESULTS_DIR / f"{checkpoint_prefix}_full.csv", index=False)
            if verbose:
                print(f"  [checkpoint written after cutoff {cutoff}]", flush=True)

    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True)


def summarise_nested(results_df):
    """Mean held-out performance per (model, objective) across cutoffs."""
    return (
        results_df.groupby(["model", "objective"])
        .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"),
             ap_mean=("average_precision", "mean"), ap_std=("average_precision", "std"),
             n_cutoffs=("cutoff", "nunique"))
        .round(4).reset_index()
    )


def save_nested(results_df, predictions_df, summary_df, prefix="nested_tuning"):
    RESULTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(RESULTS_DIR / f"{prefix}_full.csv", index=False)
    predictions_df.to_csv(RESULTS_DIR / f"{prefix}_predictions.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / f"{prefix}_summary.csv", index=False)
    return sorted(p.name for p in RESULTS_DIR.glob(f"{prefix}_*.csv"))


if __name__ == "__main__":
    # Run as: python src/tuning.py
    #
    # Kept out of the notebook deliberately. The search refits every model
    # across five cutoffs and two objectives, which takes roughly an hour;
    # embedding that would make the notebook impractical to re-run. The
    # notebook reads the tables this writes.
    import time

    from data_prep import load_flw_data, DENSE_YEAR_MAX
    from validation import CUTOFF_YEARS

    warnings.filterwarnings("ignore")
    frame = load_flw_data(year_max=DENSE_YEAR_MAX)

    started = time.time()
    results, predictions = nested_search(
        frame, cutoff_years=CUTOFF_YEARS, model_names=TRACTABLE_MODELS,
        verbose=True, n_jobs=2, checkpoint_prefix="nested_tuning",
    )
    summary = summarise_nested(results)

    print(f"\nelapsed {time.time() - started:.0f}s")
    print(summary.to_string(index=False))
    print("\nSaved:", save_nested(results, predictions, summary))
