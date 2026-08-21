"""Naive forecasting baselines the learned models are measured against.

A regression score is only interpretable next to what a trivial rule would
have achieved on the same split. Both baselines forecast within a
``(country, commodity, food_supply_stage)`` group and report the rate at
which they had to fall back to a global rule, so the reader can tell how
much of the score is real group-level forecasting.
"""

import numpy as np
import pandas as pd

from data_prep import GROUP_COLS, TARGET


def persistence_baseline(train_df, test_df):
    """Predict each test row using the group's most recent training value.

    The classic "no change" forecast: whatever this country/commodity/stage
    last reported inside the training window is what we predict for every
    later year. Groups absent from training fall back to the training-set
    global mean.

    Returns
    -------
    (predictions, fallback_rate)
        ``fallback_rate`` is the share of test rows whose group never
        appeared in training and therefore received the global mean.
    """
    # Take the group's most recent training year, then average across every
    # row recorded for it. Roughly a third of rows share their full feature
    # tuple with another row, so a group's latest year commonly holds
    # several observations; sorting and taking the last one would pick an
    # arbitrary member of that tie and make the baseline depend on row
    # order rather than on the data.
    latest_year = train_df.groupby(GROUP_COLS, as_index=False)["year"].max()
    last_obs = (
        train_df.merge(latest_year, on=GROUP_COLS + ["year"], how="inner")
        .groupby(GROUP_COLS, as_index=False)[TARGET]
        .mean()
        .rename(columns={TARGET: "pred"})
    )

    merged = test_df.merge(last_obs, on=GROUP_COLS, how="left")
    fallback_rate = float(merged["pred"].isna().mean())
    merged["pred"] = merged["pred"].fillna(train_df[TARGET].mean())

    return merged["pred"].to_numpy(), fallback_rate


def linear_trend_baseline(train_df, test_df):
    """Extrapolate a per-group least-squares time trend to the test years.

    Fits ``loss_percentage ~ year`` separately within each group using only
    that group's training rows. Groups with fewer than two distinct training
    years cannot support a slope and fall back to a single trend fit on the
    whole training set.

    Returns
    -------
    (predictions, fallback_rate)
    """
    overall_fit = np.polyfit(train_df["year"], train_df[TARGET], deg=1)

    def fit_group(g):
        if g["year"].nunique() >= 2:
            slope, intercept = np.polyfit(g["year"], g[TARGET], deg=1)
            return pd.Series({"slope": slope, "intercept": intercept})
        return pd.Series({"slope": np.nan, "intercept": np.nan})

    group_fits = train_df.groupby(GROUP_COLS)[["year", TARGET]].apply(fit_group).reset_index()
    merged = test_df.merge(group_fits, on=GROUP_COLS, how="left")

    fallback_mask = merged["slope"].isna()
    fallback_rate = float(fallback_mask.mean())

    preds = np.where(
        fallback_mask,
        overall_fit[0] * merged["year"] + overall_fit[1],
        merged["slope"] * merged["year"] + merged["intercept"],
    )

    return preds, fallback_rate


BASELINES = {
    "Persistence baseline": persistence_baseline,
    "Linear trend baseline": linear_trend_baseline,
}
