"""Prediction helpers, comparison plots, and residual diagnostics.

One plotting entry point serves every comparison. The earlier code had a
separate function per dimension being varied and per whether the model took
a year, which meant the same bug had to be fixed in each copy. Here the
dimension is an argument, and a model without a temporal feature is called
with ``year=None``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_prep import GROUP_COLS, TARGET

# The dimensions a comparison can vary over. The other two are held fixed.
VARY_DIMENSIONS = tuple(GROUP_COLS)

_AXIS_LABELS = {
    "country": "Country",
    "commodity": "Commodity",
    "food_supply_stage": "Supply Chain Stage",
}


def _build_input(country, commodity, food_supply_stage, year=None):
    """One-row frame in the column order the pipelines expect."""
    row = {
        "country": [country],
        "commodity": [commodity],
        "food_supply_stage": [food_supply_stage],
    }
    if year is not None:
        row["year"] = [year]
    return pd.DataFrame(row)


def _finite_span(*series):
    """Min and max across every finite value, or None if there are none.

    Guarding here keeps axis limits working when a requested year has no
    observations and every actual value is missing.
    """
    values = []
    for s in series:
        arr = pd.to_numeric(pd.Series(s), errors="coerce").to_numpy(dtype=float)
        values.append(arr[np.isfinite(arr)])
    stacked = np.concatenate(values) if values else np.array([])
    if stacked.size == 0:
        return None
    return float(stacked.min()), float(stacked.max())


def predict_food_loss(model, country, commodity, food_supply_stage, year=None, verbose=True):
    """Predict the loss percentage for a single country/commodity/stage."""
    prediction = float(
        model.predict(_build_input(country, commodity, food_supply_stage, year))[0]
    )
    if verbose:
        where = f"'{commodity}' in {country}, {food_supply_stage} stage"
        when = f", year {year}" if year is not None else ""
        print(f"Predicted food loss for {where}{when}: {prediction:.2f}%")
    return prediction


def food_loss_comparison(model, data, vary, year=None, **fixed):
    """Actual vs predicted loss across one dimension, others held fixed.

    Parameters
    ----------
    vary : str
        Which of country, commodity, or food_supply_stage to vary.
    **fixed
        The remaining two dimensions, as keyword arguments.

    Returns a frame with one row per level of ``vary``. Levels come from the
    rows actually observed for the fixed values, so a prediction is never
    offered for a combination the data has no basis for.
    """
    if vary not in VARY_DIMENSIONS:
        raise ValueError(f"vary must be one of {VARY_DIMENSIONS}, got {vary!r}")

    missing = set(VARY_DIMENSIONS) - {vary} - set(fixed)
    if missing:
        raise ValueError(f"must fix the remaining dimensions: {sorted(missing)}")

    mask = pd.Series(True, index=data.index)
    for col, value in fixed.items():
        mask &= data[col] == value
    relevant = data[mask]
    if relevant.empty:
        described = ", ".join(f"{k}={v!r}" for k, v in fixed.items())
        print(f"No data found for {described}.")
        return None

    # An unobserved year still gets predictions; it simply has no actuals.
    year_observed = year is None or year in set(relevant["year"].unique())
    if year is not None and not year_observed:
        print(f"Year {year} not observed; showing predictions only.")

    rows = []
    for level in relevant[vary].unique():
        subset = relevant[relevant[vary] == level]
        if year_observed and year is not None:
            subset = subset[subset["year"] == year]
        actual = float(subset[TARGET].mean()) if len(subset) else np.nan

        args = {**fixed, vary: level}
        rows.append({
            vary: level,
            "actual_loss_percentage": actual if year_observed else np.nan,
            "predicted_loss_percentage": predict_food_loss(
                model,
                args["country"], args["commodity"], args["food_supply_stage"],
                year, verbose=False,
            ),
        })

    return pd.DataFrame(rows).sort_values("predicted_loss_percentage").reset_index(drop=True)


def plot_food_loss_comparison(model, data, vary, year=None, ax=None, **fixed):
    """Draw the comparison returned by ``food_loss_comparison``."""
    results = food_loss_comparison(model, data, vary, year=year, **fixed)
    if results is None or results.empty:
        return None

    if ax is None:
        _, ax = plt.subplots(figsize=(12, max(4, 0.42 * len(results) + 2)))

    positions = np.arange(len(results))
    height = 0.38
    has_actual = results["actual_loss_percentage"].notna().any()

    if has_actual:
        bars_actual = ax.barh(positions, results["actual_loss_percentage"],
                              height=height, color="lightblue", label="Actual", alpha=0.85)
        _label_bars(ax, bars_actual, results["actual_loss_percentage"])

    bars_pred = ax.barh(positions + height, results["predicted_loss_percentage"],
                        height=height, color="lightcoral", label="Predicted", alpha=0.85)
    _label_bars(ax, bars_pred, results["predicted_loss_percentage"])

    span = _finite_span(results["actual_loss_percentage"], results["predicted_loss_percentage"])
    ax.set_xlim(*( (span[0] - 2, span[1] + 10) if span else (0, 100) ))

    context = ", ".join(f"{v}" for v in fixed.values())
    title_year = f" (Year: {year})" if year is not None else ""
    ax.set_title(f"Actual vs Predicted Food Loss by {_AXIS_LABELS[vary]}\n"
                 f"{context}{title_year}", fontsize=14)
    ax.set_xlabel("Loss Percentage (%)", fontsize=12)
    ax.set_ylabel(_AXIS_LABELS[vary], fontsize=12)
    ax.set_yticks(positions + height / 2)
    ax.set_yticklabels(results[vary], fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    return results


def _label_bars(ax, bars, values):
    """Annotate horizontal bars, skipping missing values."""
    for bar, value in zip(bars, values):
        if pd.notna(value):
            ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}%", va="center", ha="left", fontsize=8)


# --------------------------------------------------------------------------
# Residual diagnostics
# --------------------------------------------------------------------------
def _model_predictions(predictions_df, model_name):
    sub = predictions_df[predictions_df["model"] == model_name]
    if sub.empty:
        raise ValueError(f"No predictions recorded for {model_name!r}")
    return sub


def plot_predicted_vs_actual(predictions_df, model_name, ax=None, max_points=4000, seed=0):
    """Scatter predictions against observations with the identity line.

    A model that only ever predicts near the mean produces a horizontal
    cloud rather than a diagonal one, which a summary statistic hides.
    """
    sub = _model_predictions(predictions_df, model_name)
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=seed)

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5))

    ax.scatter(sub["y_true"], sub["y_pred"], s=8, alpha=0.25,
               color="steelblue", edgecolors="none")

    lo = float(min(sub["y_true"].min(), sub["y_pred"].min()))
    hi = float(max(sub["y_true"].max(), sub["y_pred"].max()))
    ax.plot([lo, hi], [lo, hi], "--", color="crimson", linewidth=1.2, label="Perfect prediction")

    full = _model_predictions(predictions_df, model_name)
    # Pooled over every held-out record. The per-split means reported in the
    # results table are a different aggregation of the same predictions and do
    # not coincide, so the basis is stated here.
    ax.set_title(f"Predicted vs Actual — {model_name}\n"
                 f"pooled over all held-out records: "
                 f"R² = {r2_score(full['y_true'], full['y_pred']):.3f}, "
                 f"MAE = {mean_absolute_error(full['y_true'], full['y_pred']):.3f}", fontsize=11)
    ax.set_xlabel("Actual loss percentage (%)", fontsize=11)
    ax.set_ylabel("Predicted loss percentage (%)", fontsize=11)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    return ax


def plot_residuals(predictions_df, model_name, ax=None, max_points=4000, seed=0):
    """Residuals against fitted values, with a zero reference line."""
    sub = _model_predictions(predictions_df, model_name)
    full_resid = sub["y_pred"] - sub["y_true"]
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=seed)
    resid = sub["y_pred"] - sub["y_true"]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(sub["y_pred"], resid, s=8, alpha=0.25, color="darkseagreen", edgecolors="none")
    ax.axhline(0, color="crimson", linestyle="--", linewidth=1.2)
    ax.set_title(f"Residuals vs Predicted — {model_name}\n"
                 f"pooled over all held-out records: "
                 f"mean residual = {full_resid.mean():+.3f}, "
                 f"sd = {full_resid.std():.3f}", fontsize=11)
    ax.set_xlabel("Predicted loss percentage (%)", fontsize=11)
    ax.set_ylabel("Residual (predicted − actual)", fontsize=11)
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    return ax


def residuals_by_group(predictions_df, model_name, group_col="country", top_n=15):
    """Error broken down by country, commodity, or stage.

    An aggregate score can hide a model that performs well on
    well-represented groups and poorly everywhere else, which is the
    question that matters for applying it globally.
    """
    sub = _model_predictions(predictions_df, model_name)
    if group_col not in sub.columns:
        raise ValueError(
            f"{group_col!r} not carried in the predictions; available: "
            f"{sorted(set(sub.columns) - {'y_true', 'y_pred', 'model'})}"
        )

    rows = []
    for name, g in sub.groupby(group_col):
        if len(g) < 2:
            continue
        rows.append({
            group_col: name,
            "n": len(g),
            "mae": mean_absolute_error(g["y_true"], g["y_pred"]),
            "mse": mean_squared_error(g["y_true"], g["y_pred"]),
            "mean_residual": float((g["y_pred"] - g["y_true"]).mean()),
            "actual_mean": float(g["y_true"].mean()),
        })

    out = pd.DataFrame(rows).sort_values("mae", ascending=False).reset_index(drop=True)
    return out.head(top_n) if top_n else out


def plot_raw_data_overview(data, top_n=20):
    """Descriptive plots of the observed data: loss by stage, country, commodity."""
    figures = []
    for col, kind, colour, title in (
        ("food_supply_stage", "v", "skyblue", "Average Food Loss Percentage by Supply Chain Stage"),
        ("country", "v", "skyblue", f"Top {top_n} Countries by Average Food Loss Percentage"),
        ("commodity", "h", "lightgreen", f"Top {top_n} Commodities by Average Food Loss Percentage"),
    ):
        agg = data.groupby(col)[TARGET].mean().sort_values(ascending=False)
        if col != "food_supply_stage":
            agg = agg.head(top_n)
        fig, ax = plt.subplots(figsize=(12, 8) if kind == "h" else (11, 6))
        if kind == "h":
            ax.barh(agg.index, agg.to_numpy(), color=colour, height=0.6)
            ax.set_xlabel("Average Loss Percentage")
            ax.set_ylabel(_AXIS_LABELS[col])
        else:
            ax.bar(agg.index, agg.to_numpy(), color=colour)
            ax.set_xlabel(_AXIS_LABELS[col])
            ax.set_ylabel("Average Loss Percentage")
            ax.tick_params(axis="x", rotation=90 if col == "country" else 45)
        ax.set_title(title)
        plt.tight_layout()
        figures.append((agg, fig))
    return figures


def error_by_loss_band(predictions_df, model_name, bands=(0, 2, 5, 10, 20, 30, 100)):
    """Break error down by how large the observed loss actually was.

    An aggregate score is dominated by the common case. Here more than four
    fifths of observations fall below 5% loss, so a model can look accurate
    overall while being systematically wrong about severe losses -- which
    are the observations an intervention would be aimed at. Reporting error
    within bands of the observed value makes that visible.
    """
    sub = _model_predictions(predictions_df, model_name).copy()
    sub["residual"] = sub["y_pred"] - sub["y_true"]
    sub["band"] = pd.cut(sub["y_true"], bins=list(bands), right=False)

    out = (
        sub.groupby("band", observed=True)
        .agg(
            n=("y_true", "size"),
            actual_mean=("y_true", "mean"),
            predicted_mean=("y_pred", "mean"),
            mean_residual=("residual", "mean"),
            mae=("residual", lambda s: s.abs().mean()),
            pct_underestimated=("residual", lambda s: 100.0 * (s < 0).mean()),
        )
        .reset_index()
    )
    out["pct_of_rows"] = (100.0 * out["n"] / len(sub)).round(2)
    return out.round(3)
