"""Prediction helpers and plots for the fitted models.

The public entry points take an optional ``year``. Models trained without a
temporal feature are called with ``year=None``; models trained with one are
given a year, which may fall outside the observed range. A single code path
serves both, so the no-observed-data case is handled in one place rather
than being re-derived per model.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_prep import TARGET


def _build_input(commodity, country, food_supply_stage, year=None):
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
    """Min and max across every finite value in the given series.

    Returns ``None`` when nothing finite is present. Guarding here is what
    keeps the axis-limit calculation working when a requested year has no
    observed data at all and every actual value is missing -- the case that
    previously raised ``ValueError: min() arg is an empty sequence``.
    """
    values = []
    for s in series:
        arr = pd.to_numeric(pd.Series(s), errors="coerce").to_numpy(dtype=float)
        values.append(arr[np.isfinite(arr)])

    stacked = np.concatenate(values) if values else np.array([])
    if stacked.size == 0:
        return None
    return float(stacked.min()), float(stacked.max())


def _apply_span(set_lim, span, pad_low=2.0, pad_high=10.0, fallback=(0.0, 100.0)):
    """Apply padded axis limits, falling back when no finite data exists."""
    if span is None:
        set_lim(*fallback)
    else:
        set_lim(span[0] - pad_low, span[1] + pad_high)


def predict_food_loss(model, commodity, country, food_supply_stage, year=None, verbose=True):
    """Predict the loss percentage for a single commodity/country/stage."""
    prediction = float(
        model.predict(_build_input(commodity, country, food_supply_stage, year))[0]
    )

    if verbose:
        where = f"commodity '{commodity}', country '{country}', stage '{food_supply_stage}'"
        when = f", year {year}" if year is not None else ""
        print(f"Predicted food loss percentage for {where}{when}: {prediction:.2f}%")

    return prediction


def _actual_mean(data, mask, year=None):
    """Mean observed loss for a subset, restricted to ``year`` when given.

    Returns ``np.nan`` when the subset is empty, so a missing observation
    propagates as a gap in the plot rather than an exception.
    """
    subset = data[mask]
    if year is not None:
        subset = subset[subset["year"] == year]
    return float(subset[TARGET].mean()) if len(subset) else np.nan


def visualize_food_loss_by_stage(commodity, country, model, data, year=None, ax=None):
    """Actual vs predicted loss across the supply-chain stages of one commodity.

    Stages come from the rows observed for this commodity and country. When
    ``year`` is given but never observed for them, actual values are absent
    and only the predictions are drawn.
    """
    relevant = data[(data["commodity"] == commodity) & (data["country"] == country)]
    if relevant.empty:
        print(f"No data found for commodity '{commodity}' in country '{country}'.")
        return None

    stages = relevant["food_supply_stage"].unique()
    year_observed = year is None or year in set(relevant["year"].unique())
    if year is not None:
        print(
            f"Using observed data from {year}."
            if year_observed
            else f"Year {year} not observed; showing predictions only."
        )

    rows = []
    for stage in stages:
        rows.append(
            {
                "food_supply_stage": stage,
                "actual_loss_percentage": _actual_mean(
                    relevant,
                    relevant["food_supply_stage"] == stage,
                    year if year_observed else None,
                )
                if year_observed
                else np.nan,
                "predicted_loss_percentage": predict_food_loss(
                    model, commodity, country, stage, year, verbose=False
                ),
            }
        )

    results = pd.DataFrame(rows).sort_values("predicted_loss_percentage")

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 8))

    positions = np.arange(len(results))
    bar_width = 0.35
    has_actual = results["actual_loss_percentage"].notna().any()

    if has_actual:
        actual_bars = ax.barh(
            positions, results["actual_loss_percentage"], height=bar_width,
            color="lightblue", label="Actual", alpha=0.7,
        )
    predicted_bars = ax.barh(
        positions + bar_width, results["predicted_loss_percentage"], height=bar_width,
        color="lightcoral", label="Predicted", alpha=0.7,
    )

    title_year = f" (Year: {year})" if year is not None else ""
    ax.set_title(
        f"Actual vs Predicted Food Loss by Supply Chain Stage\n"
        f"{commodity} in {country}{title_year}", fontsize=15,
    )
    ax.set_xlabel("Loss Percentage (%)", fontsize=13)
    ax.set_ylabel("Supply Chain Stage", fontsize=13)
    ax.set_yticks(positions + bar_width / 2)
    ax.set_yticklabels(results["food_supply_stage"], fontsize=11)

    _apply_span(
        ax.set_xlim,
        _finite_span(results["actual_loss_percentage"], results["predicted_loss_percentage"]),
    )

    _label_bars(ax, predicted_bars, results["predicted_loss_percentage"])
    if has_actual:
        _label_bars(ax, actual_bars, results["actual_loss_percentage"])

    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.legend()
    plt.tight_layout()
    return results


def visualize_food_loss_comparison(country, food_supply_stage, model, data, year=None, ax=None):
    """Actual vs predicted loss across every commodity in one country and stage."""
    relevant = data[
        (data["country"] == country) & (data["food_supply_stage"] == food_supply_stage)
    ]
    if relevant.empty:
        print(f"No data found for country '{country}' in stage '{food_supply_stage}'.")
        return None

    commodities = relevant["commodity"].unique()
    year_observed = year is None or year in set(relevant["year"].unique())
    if year is not None:
        print(
            f"Using observed data from {year}."
            if year_observed
            else f"Year {year} not observed; showing predictions only."
        )

    rows = []
    for commodity in commodities:
        rows.append(
            {
                "commodity": commodity,
                "actual_loss_percentage": _actual_mean(
                    relevant, relevant["commodity"] == commodity, year if year_observed else None
                )
                if year_observed
                else np.nan,
                "predicted_loss_percentage": predict_food_loss(
                    model, commodity, country, food_supply_stage, year, verbose=False
                ),
            }
        )

    results = pd.DataFrame(rows)

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 8))

    positions = np.arange(len(results))
    bar_width = 0.35
    has_actual = results["actual_loss_percentage"].notna().any()

    if has_actual:
        ax.bar(
            positions - bar_width / 2, results["actual_loss_percentage"], bar_width,
            label="Actual", color="lightblue", alpha=0.7,
        )
    ax.bar(
        positions + bar_width / 2, results["predicted_loss_percentage"], bar_width,
        label="Predicted", color="lightcoral", alpha=0.7,
    )

    title_year = f" (Year: {year})" if year is not None else ""
    ax.set_title(
        f"Actual vs Predicted Food Loss by Commodity\n"
        f"{country}, {food_supply_stage} stage{title_year}", fontsize=15,
    )
    ax.set_xlabel("Commodity", fontsize=13)
    ax.set_ylabel("Loss Percentage (%)", fontsize=13)
    ax.set_xticks(positions)
    ax.set_xticklabels(results["commodity"], rotation=90, fontsize=10)

    _apply_span(
        ax.set_ylim,
        _finite_span(results["actual_loss_percentage"], results["predicted_loss_percentage"]),
        pad_low=2.0, pad_high=5.0,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend()
    plt.tight_layout()
    return results


def _label_bars(ax, bars, values):
    """Annotate horizontal bars, skipping missing values."""
    for bar, value in zip(bars, values):
        if pd.notna(value):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}%",
                va="center", ha="left", fontsize=9,
            )


def plot_raw_data_overview(data, top_n=20):
    """Descriptive plots of the observed data: loss by stage, country, commodity."""
    by_stage = (
        data.groupby("food_supply_stage")[TARGET].mean()
        .sort_values(ascending=False).reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(by_stage["food_supply_stage"], by_stage[TARGET], color="skyblue")
    ax.set_title("Average Food Loss Percentage by Supply Chain Stage")
    ax.set_xlabel("Supply Chain Stage")
    ax.set_ylabel("Average Loss Percentage")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    by_country = (
        data.groupby("country")[TARGET].mean()
        .sort_values(ascending=False).head(top_n).reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(by_country["country"], by_country[TARGET], color="skyblue")
    ax.set_title(f"Top {top_n} Countries by Average Food Loss Percentage")
    ax.set_xlabel("Country")
    ax.set_ylabel("Average Loss Percentage")
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()

    by_commodity = (
        data.groupby("commodity")[TARGET].mean()
        .sort_values(ascending=False).head(top_n).reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(by_commodity["commodity"], by_commodity[TARGET], color="lightgreen", height=0.6)
    ax.set_title(f"Top {top_n} Commodities by Average Food Loss Percentage")
    ax.set_xlabel("Average Loss Percentage")
    ax.set_ylabel("Commodity")
    plt.tight_layout()

    return by_stage, by_country, by_commodity
