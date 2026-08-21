"""Time-aware walk-forward validation.

Instead of a single random train/test split, the procedure "train on years
<= Y, test on years > Y" is repeated for several cutoffs Y and the spread
across cutoffs is reported alongside the means. A single split yields one
number with no sense of how stable it is.

``year`` is standardised inside each pipeline, so the scaler is fit on the
training fold only and nothing leaks across the cutoff boundary.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from baselines import BASELINES
from data_prep import GROUP_COLS, TARGET, RESULTS_DIR
from pipelines import MODEL_NAMES, TUNED_PARAMS, build_pipeline, make_column_transformer

# Walk-forward cutoffs. Each trains on <= Y and tests on > Y.
CUTOFF_YEARS = [2016, 2017, 2018, 2019, 2020]

# Poly Ridge's degree-3 expansion is checked against this before every fit.
MEMORY_SAFETY_LIMIT_GB = 8


def poly_ridge_memory_estimate_gb(X_train, ct):
    """Estimate the memory the degree-3 expansion would need.

    Returns ``(estimated_gb, n_base_features, n_poly_features)``.
    """
    ct_fitted = ct.fit(X_train)
    X_encoded = ct_fitted.transform(X_train)
    n_base = X_encoded.shape[1]

    poly_check = PolynomialFeatures(
        degree=TUNED_PARAMS["Poly Ridge"]["poly_degree"], include_bias=False
    )
    n_poly = poly_check.fit(X_encoded[:5]).n_output_features_

    estimated_gb = (X_train.shape[0] * n_poly * 8) / 1e9
    return estimated_gb, n_base, n_poly


def score(y_true, y_pred):
    """Return the metric triple reported for every model and baseline."""
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def walk_forward(
    df,
    cutoff_years=None,
    model_names=None,
    year_mode="scaled",
    verbose=True,
):
    """Run the walk-forward validation.

    Parameters
    ----------
    df : DataFrame
        The modelling frame, already filtered by ``data_prep.load_flw_data``.
    cutoff_years : list of int, optional
        Defaults to ``CUTOFF_YEARS``.
    model_names : list of str, optional
        Defaults to all eight models.
    year_mode : str
        Passed through to the pipeline builder.

    Returns
    -------
    (results_df, predictions_df)
        ``results_df`` has one row per (cutoff, model) with the metrics.
        ``predictions_df`` holds every per-row prediction, so that paired
        significance tests and residual plots can be computed later without
        refitting anything.
    """
    cutoff_years = cutoff_years or CUTOFF_YEARS
    model_names = model_names or MODEL_NAMES

    results = []
    predictions = []

    for cutoff in cutoff_years:
        train_df = df[df["year"] <= cutoff].copy()
        test_df = df[df["year"] > cutoff].copy()

        if train_df.empty or test_df.empty:
            if verbose:
                print(f"Skipping cutoff {cutoff}: empty train or test set.")
            continue

        if verbose:
            print(f"=== Cutoff Y={cutoff}  (train n={len(train_df)}, test n={len(test_df)}) ===")

        X_train = train_df[GROUP_COLS + ["year"]]
        y_train = train_df[TARGET]
        X_test = test_df[GROUP_COLS + ["year"]]
        y_test = test_df[TARGET]

        # --- Baselines ---
        for base_name, base_fn in BASELINES.items():
            preds, fallback_rate = base_fn(train_df, test_df)
            metrics = score(y_test, preds)
            results.append(
                dict(
                    cutoff=cutoff,
                    model=base_name,
                    n_test=len(test_df),
                    fallback_rate=fallback_rate,
                    fitted=True,
                    **metrics,
                )
            )
            predictions.append(
                pd.DataFrame(
                    dict(
                        cutoff=cutoff,
                        model=base_name,
                        y_true=y_test.to_numpy(),
                        y_pred=np.asarray(preds),
                        year=test_df["year"].to_numpy(),
                    )
                )
            )
            if verbose:
                print(
                    f"  {base_name:<22}: MSE={metrics['mse']:.4f} "
                    f"MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f} "
                    f"(fallback rate: {fallback_rate:.1%})"
                )

        # --- Models ---
        for name in model_names:
            if name == "Poly Ridge":
                est_gb, n_base, n_poly = poly_ridge_memory_estimate_gb(
                    X_train, make_column_transformer(year_mode)
                )
                if est_gb > MEMORY_SAFETY_LIMIT_GB:
                    if verbose:
                        print(
                            f"  {name:<22}: SKIPPED — estimated {est_gb:,.0f} GB "
                            f"({n_poly:,} degree-3 features from {n_base} base "
                            f"features) exceeds {MEMORY_SAFETY_LIMIT_GB} GB limit."
                        )
                    results.append(
                        dict(
                            cutoff=cutoff,
                            model=name,
                            n_test=len(test_df),
                            fallback_rate=np.nan,
                            fitted=False,
                            mse=np.nan,
                            mae=np.nan,
                            r2=np.nan,
                        )
                    )
                    continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe = build_pipeline(name, year_mode=year_mode)
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)

            metrics = score(y_test, preds)
            results.append(
                dict(
                    cutoff=cutoff,
                    model=name,
                    n_test=len(test_df),
                    fallback_rate=np.nan,
                    fitted=True,
                    **metrics,
                )
            )
            predictions.append(
                pd.DataFrame(
                    dict(
                        cutoff=cutoff,
                        model=name,
                        y_true=y_test.to_numpy(),
                        y_pred=np.asarray(preds),
                        year=test_df["year"].to_numpy(),
                    )
                )
            )
            if verbose:
                print(
                    f"  {name:<22}: MSE={metrics['mse']:.4f} "
                    f"MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f}"
                )

        if verbose:
            print()

    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True)


def summarise(results_df):
    """Aggregate per-cutoff results into the mean +/- std table."""
    return (
        results_df.groupby("model")
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            n_cutoffs=("cutoff", "count"),
        )
        .round(4)
        .sort_values("r2_mean", ascending=False)
    )


def save_results(results_df, predictions_df, summary_df, prefix="time_aware_validation"):
    """Write results, per-row predictions, and the summary to results/."""
    RESULTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(RESULTS_DIR / f"{prefix}_full_results.csv", index=False)
    predictions_df.to_csv(RESULTS_DIR / f"{prefix}_predictions.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / f"{prefix}_summary.csv")
    return sorted(p.name for p in RESULTS_DIR.glob(f"{prefix}_*.csv"))
