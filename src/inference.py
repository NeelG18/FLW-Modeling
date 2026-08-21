"""Paired significance tests comparing models against a baseline.

Reporting a mean and a standard deviation across folds conflates two very
different sources of variation: how much models differ from each other, and
how much the folds differ in difficulty. In this data the second dominates
-- the persistence baseline's MSE ranges from about 22 at the earliest
cutoff to about 8 at the latest -- so a standard deviation across folds
mostly measures which years were tested, not whether one model beats
another.

Pairing removes that. Both predictors are scored on the identical rows, so
taking the per-row difference in error cancels fold difficulty entirely and
leaves only the contrast of interest.

Two tests are reported because they answer different questions:

* squared error drives MSE and R-squared, and is dominated by large misses
* absolute error drives MAE, and reflects the typical row

These can disagree, and when they do the disagreement is the finding.
"""

import numpy as np
import pandas as pd
from scipy import stats

from data_prep import RESULTS_DIR

DEFAULT_N_BOOT = 10000
DEFAULT_SEED = 0
BASELINE_MODEL = "Persistence baseline"


def paired_bootstrap_ci(errors_model, errors_baseline, n_boot=DEFAULT_N_BOOT,
                        seed=DEFAULT_SEED, alpha=0.05):
    """Percentile bootstrap CI for the mean paired error difference.

    Resamples rows, not predictors: each draw keeps the pairing intact, so
    the interval describes uncertainty about the contrast rather than about
    either predictor separately.

    Returns ``(mean_difference, ci_low, ci_high)``. Negative values mean the
    model has lower error than the baseline.
    """
    diff = np.asarray(errors_model) - np.asarray(errors_baseline)
    n = diff.size
    if n == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)

    return (
        float(diff.mean()),
        float(np.percentile(boot_means, 100 * alpha / 2)),
        float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    )


def _wilcoxon(errors_model, errors_baseline):
    """Wilcoxon signed-rank on paired errors, tolerant of degenerate input."""
    diff = np.asarray(errors_model) - np.asarray(errors_baseline)
    if diff.size == 0 or np.allclose(diff, 0):
        return np.nan, np.nan
    try:
        stat, p = stats.wilcoxon(diff)
        return float(stat), float(p)
    except ValueError:
        return np.nan, np.nan


def compare_against_baseline(predictions_df, split_col="cutoff",
                             baseline=BASELINE_MODEL, n_boot=DEFAULT_N_BOOT,
                             seed=DEFAULT_SEED):
    """Compare every model to the baseline, within each split, on both metrics.

    Parameters
    ----------
    predictions_df : DataFrame
        Per-row predictions as written by ``walk_forward`` or ``grouped_cv``:
        columns ``y_true``, ``y_pred``, ``model``, and the split column.
    split_col : str
        ``"cutoff"`` for the walk-forward, ``"fold"`` for grouped CV.

    Returns
    -------
    DataFrame
        One row per (split, model, metric). ``beats_baseline`` is True only
        when the whole confidence interval lies below zero, i.e. the model's
        error is lower and the interval does not straddle no-difference.
    """
    rows = []

    for split_value, split_df in predictions_df.groupby(split_col, sort=True):
        base = split_df[split_df["model"] == baseline]
        if base.empty:
            continue
        base = base.reset_index(drop=True)

        for name, model_df in split_df.groupby("model", sort=False):
            if name == baseline:
                continue
            model_df = model_df.reset_index(drop=True)
            if len(model_df) != len(base):
                continue

            # Both predictors were scored on the same test rows in the same
            # order, so position aligns them. Verify rather than assume.
            if not np.allclose(model_df["y_true"], base["y_true"]):
                raise ValueError(
                    f"y_true misaligned for {name!r} at {split_col}={split_value}"
                )

            resid_model = model_df["y_pred"].to_numpy() - model_df["y_true"].to_numpy()
            resid_base = base["y_pred"].to_numpy() - base["y_true"].to_numpy()

            for metric, transform in (
                ("squared_error", np.square),
                ("absolute_error", np.abs),
            ):
                err_model = transform(resid_model)
                err_base = transform(resid_base)
                mean_diff, lo, hi = paired_bootstrap_ci(
                    err_model, err_base, n_boot=n_boot, seed=seed
                )
                stat, p = _wilcoxon(err_model, err_base)

                rows.append({
                    split_col: split_value,
                    "model": name,
                    "metric": metric,
                    "n": len(model_df),
                    "model_mean_error": float(err_model.mean()),
                    "baseline_mean_error": float(err_base.mean()),
                    "mean_difference": mean_diff,
                    "ci_low": lo,
                    "ci_high": hi,
                    "wilcoxon_stat": stat,
                    "wilcoxon_p": p,
                    "beats_baseline": bool(hi < 0),
                    "worse_than_baseline": bool(lo > 0),
                })

    return pd.DataFrame(rows)


def summarise_comparison(comparison_df, split_col="cutoff"):
    """Collapse per-split comparisons into a per-model verdict.

    A model is only credited with beating the baseline on a metric where it
    does so at every split. One split out of five is not a result.
    """
    out = (
        comparison_df.groupby(["model", "metric"])
        .agg(
            n_splits=(split_col, "nunique"),
            n_splits_better=("beats_baseline", "sum"),
            n_splits_worse=("worse_than_baseline", "sum"),
            mean_difference=("mean_difference", "mean"),
            worst_p=("wilcoxon_p", "max"),
        )
        .reset_index()
    )
    out["verdict"] = np.where(
        out["n_splits_better"] == out["n_splits"], "beats baseline at every split",
        np.where(
            out["n_splits_worse"] == out["n_splits"], "worse at every split",
            "mixed",
        ),
    )
    return out.round(6)


def save_comparison(comparison_df, summary_df, prefix="paired_comparison"):
    """Write the per-split comparison and its summary to results/."""
    RESULTS_DIR.mkdir(exist_ok=True)
    comparison_df.to_csv(RESULTS_DIR / f"{prefix}_full.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / f"{prefix}_summary.csv", index=False)
    return sorted(p.name for p in RESULTS_DIR.glob(f"{prefix}_*.csv"))
