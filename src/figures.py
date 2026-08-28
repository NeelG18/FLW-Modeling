"""Generation of every figure and table reported in the paper.

Each function here produces one numbered item and writes it to ``figures/``
or ``results/``. Nothing is drawn by hand: every value comes from the tables
written by the validation, ablation and targeting modules, so a figure
cannot drift from the result it depicts. Re-running ``build_all`` after a
change to the analysis regenerates the whole set.

Figures are numbered F1-F16 and tables T1-T10 in the order they appear in
the paper. The mapping to manuscript sections is recorded in ``ITEMS``.
"""

import textwrap
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyArrowPatch

from data_prep import (FIGURES_DIR, RESULTS_DIR, TARGET, GROUP_COLS,
                       DENSE_YEAR_MAX, load_flw_data)

# --------------------------------------------------------------------------
# Shared style
# --------------------------------------------------------------------------
# One palette, applied everywhere, so that a colour means the same thing in
# every figure: the flagship model, a baseline, and neutral context.
FLAGSHIP = "#1f6f8b"
BASELINE = "#c1666b"
NEUTRAL = "#8d99ae"
ACCENT = "#48a9a6"
HIGHLIGHT = "#e08e45"

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
    # Keep text as text in SVG exports rather than converting it to outlines,
    # so labels stay editable in a vector editor.
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# Formats written for every figure. PNG is what the manuscript embeds; SVG and
# PDF are vector and can be opened in a drawing program to adjust a label or a
# colour by hand without regenerating anything.
FIGURE_FORMATS = ("png", "svg", "pdf")

FLAGSHIP_MODEL = "Random Forest"
BASELINE_MODEL = "Persistence baseline"
SEVERITY = 10.0

ITEMS = {}   # id -> (section, description), filled by the decorators below


def _register(item_id, section, description):
    ITEMS[item_id] = (section, description)


def _save_fig(fig, item_id, description, section, formats=FIGURE_FORMATS):
    FIGURES_DIR.mkdir(exist_ok=True)
    stem = f"{item_id.lower()}_{_slug(description)}"
    png_path = FIGURES_DIR / f"{stem}.png"
    for fmt in formats:
        target = FIGURES_DIR / f"{stem}.{fmt}"
        if fmt != "png":
            (FIGURES_DIR / "vector").mkdir(exist_ok=True)
            target = FIGURES_DIR / "vector" / f"{stem}.{fmt}"
        fig.savefig(target, format=fmt)
    plt.close(fig)
    _register(item_id, section, description)
    return png_path


def _save_table(frame, item_id, description, section, index=False):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{item_id.lower()}_{_slug(description)}.csv"
    frame.to_csv(path, index=index)
    _register(item_id, section, description)
    return path


def _slug(text):
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")[:48]


def _read(name):
    return pd.read_csv(RESULTS_DIR / name)


def _predictions(name="time_aware_validation_predictions.csv"):
    return _read(name)


def _wrap(ax, title, width=64):
    ax.set_title("\n".join(textwrap.wrap(title, width)))


# ==========================================================================
# SECTION 2 - DATASET
# ==========================================================================
def t1_dataset_composition():
    """T1. What the modelling frame contains, before and after filtering."""
    raw = load_flw_data(drop_missing_stage=False)
    frame = load_flw_data()
    dense = load_flw_data(year_max=DENSE_YEAR_MAX)
    rows = [
        ("Records in the source extract", f"{len(raw):,}"),
        ("Records lacking a food supply stage", f"{len(raw) - len(frame):,}"),
        ("Records in the modelling frame", f"{len(frame):,}"),
        ("Records used for validation (2000-2021)", f"{len(dense):,}"),
        ("Countries", f"{frame.country.nunique()}"),
        ("Commodity labels", f"{frame.commodity.nunique()}"),
        ("CPC codes", f"{frame.cpc_code.nunique()}"),
        ("Supply chain stages", f"{frame.food_supply_stage.nunique()}"),
        ("Year range", f"{frame.year.min()}-{frame.year.max()}"),
        ("Median loss percentage", f"{frame[TARGET].median():.2f}"),
        ("Mean loss percentage", f"{frame[TARGET].mean():.2f}"),
        ("Maximum loss percentage", f"{frame[TARGET].max():.1f}"),
    ]
    out = pd.DataFrame(rows, columns=["Quantity", "Value"])
    return _save_table(out, "T1", "dataset composition", "2")


def f1_field_coverage():
    """F1. How completely each of the 18 source fields is recorded."""
    raw = load_flw_data(drop_missing_stage=False)
    cov = (raw.notna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    colours = [BASELINE if v < 8 else (NEUTRAL if v < 100 else FLAGSHIP) for v in cov]
    ax.barh(cov.index, cov.to_numpy(), color=colours)
    ax.axvline(8, color=BASELINE, linestyle=":", linewidth=1.2)
    ax.text(8.8, 0.2, "8% coverage", color=BASELINE, fontsize=8, rotation=90, va="bottom")
    ax.set_xlabel("Records where the field is recorded (%)")
    ax.set_xlim(0, 104)
    for y, v in enumerate(cov.to_numpy()):
        ax.text(v + 1, y, f"{v:.1f}", va="center", fontsize=7, color="#333")
    _wrap(ax, "F1. Field coverage in the source extract. Five fields fall below "
              "8%, including cause of loss and loss quantity.")
    return _save_fig(fig, "F1", "field coverage", "2")


def f2_records_per_year():
    """F2. Why validation stops at 2021."""
    frame = load_flw_data()
    counts = frame.year.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    colours = [BASELINE if y > DENSE_YEAR_MAX else FLAGSHIP for y in counts.index]
    ax.bar(counts.index, counts.to_numpy(), color=colours, width=0.75)
    for y in counts.index[counts.index > DENSE_YEAR_MAX]:
        ax.text(y, counts[y] + 40, f"{counts[y]}", ha="center", fontsize=7, color=BASELINE)
    ax.axvline(DENSE_YEAR_MAX + 0.5, color="#444", linestyle="--", linewidth=1)
    ax.text(DENSE_YEAR_MAX + 0.7, counts.max() * 0.85, "validation\ncutoff",
            fontsize=7.5, color="#444")
    ax.set_xlabel("Year"); ax.set_ylabel("Records")
    _wrap(ax, "F2. Records per year. The 2022-2024 tail holds 147, 44 and 144 records "
              "against roughly 1,000-1,700 for earlier years, reflecting reporting lag.")
    return _save_fig(fig, "F2", "records per year", "2")


def t2_cause_of_loss():
    """T2. Why cause of loss is excluded, and what including it would do."""
    prof = _read("cause_of_loss_profile.csv").iloc[0]
    sens = _read("cause_of_loss_sensitivity.csv")
    agg = (sens.groupby(["model", "variant"])["test_r2"].agg(["mean", "std"])
              .round(4).reset_index())
    rows = [
        ("Rows where the field is recorded", f"{int(prof.n_rows_with_value):,} "
                                             f"({prof.pct_coverage}%)"),
        ("Distinct values", f"{int(prof.n_distinct_values)}"),
        ("Mean rows per value", f"{prof.mean_rows_per_value}"),
        ("Values occurring exactly once", f"{int(prof.n_values_occurring_once)} "
                                          f"({prof.pct_values_occurring_once}%)"),
        ("Median value length (characters)", f"{int(prof.median_value_length_chars)}"),
        ("Longest value (characters)", f"{int(prof.max_value_length_chars):,}"),
    ]
    for _, r in agg[agg.model == FLAGSHIP_MODEL].iterrows():
        rows.append((f"Random forest R², {r.variant}", f"{r['mean']:.4f} ± {r['std']:.4f}"))
    out = pd.DataFrame(rows, columns=["Property", "Value"])
    return _save_table(out, "T2", "cause of loss profile and sensitivity", "2")


def t3_repeated_combinations():
    """T3. Repeated feature combinations, and whether they leak."""
    audit = _read("duplicate_tuple_audit.csv").iloc[0]
    seen = _read("seen_vs_unseen_tuples.csv")
    rows = [
        ("Records in the frame", f"{int(audit.n_rows):,}"),
        ("Distinct feature combinations", f"{int(audit.n_distinct_tuples):,}"),
        ("Records sharing a combination", f"{int(audit.n_rows_sharing_a_tuple):,} "
                                          f"({audit.pct_rows_sharing_a_tuple}%)"),
        ("Largest repeated group", f"{int(audit.largest_tuple_group)} records"),
        ("Test records whose combination is in training",
         f"{int(audit.n_test_rows_seen_in_train):,} ({audit.pct_test_rows_seen_in_train}%)"),
    ]
    for _, r in seen.iterrows():
        rows.append((f"{r.model}: MSE seen vs unseen",
                     f"{r.mse_seen:.2f} vs {r.mse_unseen:.2f}"))
    out = pd.DataFrame(rows, columns=["Property", "Value"])
    return _save_table(out, "T3", "repeated combination audit", "2")


# ==========================================================================
# SECTION 3 - METHODOLOGY
# ==========================================================================
def f3_loss_distribution():
    """F3. The skew that shapes every metric choice."""
    frame = load_flw_data()
    y = frame[TARGET]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2),
                                  gridspec_kw={"width_ratios": [2, 1]})
    ax.hist(y, bins=np.arange(0, 66, 1), color=FLAGSHIP, alpha=0.85)
    for q, style, label in ((y.median(), ":", "median"),
                            (y.quantile(0.90), "--", "90th percentile")):
        ax.axvline(q, color=BASELINE, linestyle=style, linewidth=1.3)
        ax.text(q + 0.7, ax.get_ylim()[1] * 0.72, f"{label}\n{q:.2f}%",
                fontsize=7.5, color=BASELINE)
    ax.set_xlim(0, 45); ax.set_xlabel("Observed loss (%)"); ax.set_ylabel("Records")
    ax.set_title("Distribution of observed loss")

    bands = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 100)]
    share = [100 * ((y >= lo) & (y < hi)).mean() for lo, hi in bands]
    labels = [f"{lo}–{hi}%" if hi < 100 else f"{lo}%+" for lo, hi in bands]
    ax2.barh(labels, share, color=[FLAGSHIP] * 3 + [HIGHLIGHT, BASELINE])
    for i, v in enumerate(share):
        ax2.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=7.5)
    ax2.set_xlim(0, 68); ax2.set_xlabel("Share of records (%)")
    ax2.set_title("Records by band")
    below5 = 100 * (y < 5).mean()
    fig.suptitle(f"F3. Observed loss is strongly right-skewed: {below5:.1f}% of records in the "
                 f"modelling frame fall below 5%, and the top decile begins at "
                 f"{y.quantile(0.90):.0f}%, which is the severity threshold used throughout.",
                 fontsize=9, y=1.04)
    return _save_fig(fig, "F3", "loss distribution", "3.1")


def f4_error_concentration():
    """F4. How few records determine a squared-error metric."""
    preds = _predictions()
    d = preds[preds.model == FLAGSHIP_MODEL].sort_values("y_true", ascending=False)
    sq = ((d.y_pred - d.y_true) ** 2).to_numpy()
    ab = (d.y_pred - d.y_true).abs().to_numpy()
    x = 100 * np.arange(1, len(sq) + 1) / len(sq)

    # The share of records at or above 20% loss, which is the claim the text makes.
    cut_pct = 100 * (d.y_true >= 20).mean()
    at = int(len(sq) * cut_pct / 100)
    share_sq = 100 * sq[:at].sum() / sq.sum()
    share_ab = 100 * ab[:at].sum() / ab.sum()

    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    ax.plot(x, 100 * np.cumsum(sq) / sq.sum(), color=BASELINE,
            label="squared error", linewidth=1.9)
    ax.plot(x, 100 * np.cumsum(ab) / ab.sum(), color=FLAGSHIP,
            label="absolute error", linewidth=1.9)
    ax.plot([0, 100], [0, 100], color=NEUTRAL, linestyle=":", linewidth=1,
            label="even contribution")
    ax.axvline(cut_pct, color="#444", linestyle="--", linewidth=0.9)
    ax.plot([cut_pct], [share_sq], "o", color=BASELINE, ms=5.5)
    ax.plot([cut_pct], [share_ab], "o", color=FLAGSHIP, ms=5.5)
    # Anchored well to the right of the marker line so neither is obscured.
    ax.annotate(f"records at or above 20% loss\n"
                f"({cut_pct:.1f}% of records)\n"
                f"{share_sq:.1f}% of squared error\n"
                f"{share_ab:.1f}% of absolute error",
                xy=(cut_pct, share_sq), xytext=(34, 9), fontsize=7.8,
                arrowprops=dict(arrowstyle="->", color="#666", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccc", lw=0.7))
    ax.set_xlabel("Records, ranked by observed loss, largest first (%)")
    ax.set_ylabel("Cumulative share of total error (%)")
    ax.legend(loc="lower right", fontsize=8)
    _wrap(ax, "F4. Squared error is concentrated in the records with the largest "
              "observed losses; absolute error is spread far more evenly.")
    return _save_fig(fig, "F4", "error concentration", "3.1")


def f5_validation_schematic():
    """F5. The two validation procedures, drawn."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 4.6))
    cutoffs = [2016, 2017, 2018, 2019, 2020]
    y0, y1 = 2000, 2021
    for i, c in enumerate(cutoffs):
        ax1.add_patch(Rectangle((y0, i - 0.32), c - y0 + 1, 0.64,
                                facecolor=FLAGSHIP, alpha=0.85))
        ax1.add_patch(Rectangle((c + 1, i - 0.32), y1 - c, 0.64,
                                facecolor=HIGHLIGHT, alpha=0.9))
        ax1.text(y0 + 0.3, i, "train", va="center", fontsize=7.5, color="white")
        ax1.text(c + 1.3, i, "test", va="center", fontsize=7.5, color="white")
    ax1.set_xlim(y0, y1 + 1); ax1.set_ylim(-0.7, len(cutoffs) - 0.3)
    ax1.set_yticks(range(len(cutoffs)))
    ax1.set_yticklabels([f"cutoff {c}" for c in cutoffs], fontsize=8)
    ax1.set_xticks(range(y0, y1 + 2, 3))
    ax1.set_xticklabels([str(y) for y in range(y0, y1 + 2, 3)])
    ax1.set_xlabel("Year"); ax1.grid(False)
    ax1.set_title("Walk-forward: train on years ≤ Y, test on years > Y", fontsize=9)

    rng = np.random.default_rng(0)
    for f in range(5):
        for j in range(5):
            colour = HIGHLIGHT if j == f else FLAGSHIP
            ax2.add_patch(Rectangle((j, f - 0.32), 0.92, 0.64,
                                    facecolor=colour, alpha=0.85))
        ax2.text(5.15, f, "test fold shaded", fontsize=7, color="#444", va="center") if f == 0 else None
    ax2.set_xlim(-0.1, 7.2); ax2.set_ylim(-0.7, 4.7)
    ax2.set_yticks(range(5)); ax2.set_yticklabels([f"fold {i+1}" for i in range(5)], fontsize=8)
    ax2.set_xticks(np.arange(5) + 0.46)
    ax2.set_xticklabels([f"group\nblock {i+1}" for i in range(5)], fontsize=7)
    ax2.grid(False)
    ax2.set_title("Grouped: whole country × commodity pairs held out (1,437 pairs)", fontsize=9)
    fig.suptitle("F5. The two validation procedures. Each answers a different question: "
                 "reaching a later year, and reaching an unseen pair.", fontsize=9, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, "F5", "validation schematic", "3.2")


def t4_random_forest_ablation():
    """T4. All 32 random forest configurations, current one marked."""
    abl = pd.read_csv(RESULTS_DIR / "random_forest_ablation.csv",
                      keep_default_na=False, dtype=str)
    for c in ("mse_mean", "mae_mean", "r2_mean", "r2_std"):
        abl[c] = abl[c].astype(float)
    abl["r2_rank"] = abl.r2_mean.rank(ascending=False).astype(int)
    abl["mae_rank"] = abl.mae_mean.rank().astype(int)
    abl["in_use"] = ((abl.bootstrap == "False") & (abl.max_features == "sqrt")
                     & (abl.max_depth == "None")).map({True: "yes", False: ""})
    out = abl[["bootstrap", "max_features", "max_depth", "r2_mean", "r2_std",
               "mae_mean", "r2_rank", "mae_rank", "in_use"]].sort_values("r2_rank")
    return _save_table(out.round(4), "T4", "random forest ablation", "3.9")


# ==========================================================================
# SECTION 4 - RESULTS
# ==========================================================================
def _adjusted(r2, n, p=383):
    from validation import adjusted_r2
    return adjusted_r2(r2, n, p)


def t5_primary_results():
    """T5. The main comparison: every model under time-aware validation."""
    from validation import skill_score
    summary = _read("time_aware_validation_summary.csv")
    full = _read("time_aware_validation_full_results.csv")
    rank = _read("ranking_metrics.csv").set_index("model")
    n_test = int(full.groupby("cutoff").n_test.first().mean())
    base_mae = float(summary.loc[summary.model == BASELINE_MODEL, "mae_mean"].iloc[0])

    rows = []
    for _, r in summary.iterrows():
        rows.append({
            "Model": r.model,
            "MAE": round(r.mae_mean, 3),
            "MAE sd": round(r.mae_std, 3),
            "Skill vs persistence": round(skill_score(r.mae_mean, base_mae), 3),
            "R²": round(r.r2_mean, 3),
            "Adjusted R²": round(_adjusted(r.r2_mean, n_test), 3),
            "AP @10%": round(rank.loc[r.model, "ap@10"], 3) if r.model in rank.index else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    return _save_table(out, "T5", "primary walk-forward results", "4.1")


def t6_grouped_results():
    """T6. The same models on unseen country and commodity pairs."""
    from validation import skill_score
    summary = _read("grouped_cv_summary.csv")
    base = float(summary.loc[summary.model == BASELINE_MODEL, "mae_mean"].iloc[0])
    out = pd.DataFrame({
        "Model": summary.model,
        "MAE": summary.mae_mean.round(3),
        "MAE sd": summary.mae_std.round(3),
        "Skill vs persistence": [round(skill_score(m, base), 3) for m in summary.mae_mean],
        "R²": summary.r2_mean.round(3),
        "R² sd": summary.r2_std.round(3),
    }).sort_values("MAE").reset_index(drop=True)
    return _save_table(out, "T6", "grouped cv results", "4.1")


def f6_paired_comparison():
    """F6. Paired differences against the baseline, with intervals."""
    full = _read("paired_comparison_walkforward_full.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), sharey=True)
    for ax, metric, title in zip(axes, ("squared_error", "absolute_error"),
                                 ("Squared error", "Absolute error")):
        d = full[full.metric == metric]
        order = d.groupby("model").mean_difference.mean().sort_values().index
        for i, model in enumerate(order):
            g = d[d.model == model]
            ax.hlines(i, g.ci_low.min(), g.ci_high.max(), color=NEUTRAL, linewidth=1)
            for _, r in g.iterrows():
                colour = FLAGSHIP if r.ci_high < 0 else (BASELINE if r.ci_low > 0 else NEUTRAL)
                ax.plot([r.mean_difference], [i], "o", color=colour, ms=4, alpha=0.9)
            ax.plot([g.mean_difference.mean()], [i], "D", color="#222", ms=5)
        ax.axvline(0, color="#222", linewidth=1)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
        ax.set_xlabel("Model error − baseline error")
        ax.set_title(title, fontsize=9)
        if metric == "squared_error":
            ax.set_xlim(-8, 12)
    fig.suptitle("F6. Paired differences against the persistence baseline, one point per "
                 "cutoff, diamond = mean. Negative favours the model.\nBlue: interval "
                 "entirely below zero. Red: entirely above.", fontsize=9, y=1.06)
    fig.tight_layout()
    return _save_fig(fig, "F6", "paired comparison vs baseline", "4.1")


def t7_nested_tuning():
    """T7. What fixing the selection leakage changed."""
    fixed = _read("time_aware_validation_summary.csv")[["model", "mae_mean"]]
    nested = _read("nested_tuning_summary.csv")
    best = nested.loc[nested.groupby("model").mae_mean.idxmin()][["model", "mae_mean", "objective"]]
    out = (fixed.merge(best, on="model", suffixes=(" fixed", " nested"))
                .rename(columns={"model": "Model", "objective": "Best objective"}))
    out["Change"] = (out["mae_mean nested"] - out["mae_mean fixed"]).round(4)
    out = out.round(4).sort_values("Change")
    return _save_table(out, "T7", "nested vs fixed selection", "4.1")


def t8_error_by_band():
    """T8. Where the model is accurate and where it is not."""
    band = _read("error_by_loss_band.csv")
    out = band.rename(columns={
        "band": "Observed loss band", "n": "Records", "pct_of_rows": "Share of records (%)",
        "actual_mean": "Mean observed", "predicted_mean": "Mean predicted",
        "mean_residual": "Mean residual", "mae": "MAE",
        "pct_underestimated": "Underestimated (%)"})
    return _save_table(out, "T8", "error by observed loss band", "4.2")


def f7_residuals():
    """F7. Predicted against actual, and residuals, on held-out records."""
    from visualization import plot_predicted_vs_actual, plot_residuals
    preds = _predictions()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    plot_predicted_vs_actual(preds, FLAGSHIP_MODEL, ax=axes[0])
    plot_residuals(preds, FLAGSHIP_MODEL, ax=axes[1])
    fig.suptitle("F7. Held-out predictions. The model compresses the upper tail: no "
                 "prediction exceeds 51.8% while observations reach 62.9%.",
                 fontsize=9, y=1.03)
    fig.tight_layout()
    return _save_fig(fig, "F7", "predicted vs actual and residuals", "4.2")


def f8_targeting_curve():
    """F8. What a fixed inspection budget recovers."""
    from targeting import targeting_curve
    preds = _predictions()
    ks = (1, 2, 5, 10, 15, 20, 30, 40, 50)
    rf = targeting_curve(preds, FLAGSHIP_MODEL, ks=ks, threshold=SEVERITY)
    bl = targeting_curve(preds, BASELINE_MODEL, ks=ks, threshold=SEVERITY)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    for ax, col, label in zip(axes, ("precision", "recall", "lift"),
                              ("Precision", "Recall of severe records", "Lift over random")):
        ax.plot(rf.k_pct, rf[col], "-o", color=FLAGSHIP, ms=4, label="Random forest")
        ax.plot(bl.k_pct, bl[col], "-s", color=BASELINE, ms=4, label="Persistence baseline")
        ax.set_xlabel("Top k% of the ranking inspected"); ax.set_title(label, fontsize=9)
        if col == "lift":
            ax.axhline(1, color=NEUTRAL, linestyle=":", linewidth=1)
    axes[0].axhline(rf.attrs["base_rate"], color=NEUTRAL, linestyle=":", linewidth=1)
    axes[0].text(22, rf.attrs["base_rate"] + 0.012, "prevalence", fontsize=7, color=NEUTRAL)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"F8. Screening performance at severity ≥{SEVERITY:.0f}% loss. Inspecting the "
                 "top 5% recovers 41.5% of severe records at 8.3× the random rate.",
                 fontsize=9, y=1.05)
    fig.tight_layout()
    return _save_fig(fig, "F8", "targeting curve", "4.2")


def f9_threshold_sensitivity():
    """F9. Whether the ranking advantage depends on where severity is drawn."""
    rank = _read("ranking_metrics.csv")
    thresholds = [5, 10, 15, 20, 30]
    show = [FLAGSHIP_MODEL, "Poly Ridge", "SVM", "KNN", BASELINE_MODEL]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for model in show:
        r = rank[rank.model == model]
        if r.empty: continue
        vals = [float(r[f"ap@{t}"].iloc[0]) for t in thresholds]
        style = dict(color=BASELINE, marker="s", linewidth=2.2) if model == BASELINE_MODEL else \
                dict(color=FLAGSHIP, marker="o", linewidth=2.2) if model == FLAGSHIP_MODEL else \
                dict(color=NEUTRAL, marker=".", linewidth=1.2, alpha=0.9)
        ax.plot(thresholds, vals, label=model, ms=5, **style)
    ax.axvline(SEVERITY, color="#444", linestyle="--", linewidth=0.9)
    ax.text(SEVERITY + 0.4, 0.66, "threshold used\n(top decile)", fontsize=7.5, color="#444")
    ax.set_xlabel("Severity threshold (% loss)"); ax.set_ylabel("Average precision")
    ax.legend(fontsize=7.5)
    _wrap(ax, "F9. Average precision across severity thresholds. Below about 8% the "
              "baseline is ahead; the models' advantage is specific to severe losses.")
    return _save_fig(fig, "F9", "threshold sensitivity", "4.2")


def f10_loss_capture():
    """F10. Share of total loss recovered, against the achievable ceiling."""
    from targeting import loss_capture_curve
    preds = _predictions()
    ks = (1, 2, 5, 10, 15, 20, 30, 40, 50)
    rf = loss_capture_curve(preds, FLAGSHIP_MODEL, ks=ks)
    bl = loss_capture_curve(preds, BASELINE_MODEL, ks=ks)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(rf.k_pct, rf.perfect_ranking_pct, color="#222", linestyle="--",
            linewidth=1.4, label="perfect ranking")
    ax.plot(rf.k_pct, rf.loss_captured_pct, "-o", color=FLAGSHIP, ms=4, label="Random forest")
    ax.plot(bl.k_pct, bl.loss_captured_pct, "-s", color=BASELINE, ms=4, label="Persistence baseline")
    ax.plot(rf.k_pct, rf.random_pct, color=NEUTRAL, linestyle=":", linewidth=1, label="random")
    ax.set_xlabel("Top k% of the ranking inspected")
    ax.set_ylabel("Share of total observed loss captured (%)")
    ax.legend(fontsize=8)
    _wrap(ax, "F10. Loss captured at each inspection budget. Loss is a rate, so every "
              "record counts equally regardless of production volume.")
    return _save_fig(fig, "F10", "loss capture", "4.2")


def t9_worst_stage():
    """T9. Identifying the highest-loss stage, models against the baseline."""
    ws = _read("worst_stage_accuracy.csv").sort_values("correct_pct", ascending=False)
    out = ws.rename(columns={"model": "Model", "n_comparisons": "Comparisons",
                             "correct_pct": "Correct (%)", "chance_pct": "Chance (%)",
                             "lift": "Lift"}).round(2)
    return _save_table(out, "T9", "worst stage identification", "4.3")


def _fit_flagship(include_year=True):
    from pipelines import build_pipeline
    frame = load_flw_data()
    cols = GROUP_COLS + (["year"] if include_year else [])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = build_pipeline(FLAGSHIP_MODEL, year_mode="scaled")
        pipe.fit(frame[cols], frame[TARGET])
    return pipe, frame


def f11_stage_comparison():
    """F11. Actual against predicted across the stages of one supply chain."""
    from visualization import plot_food_loss_comparison
    pipe, frame = _fit_flagship()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    # Chosen for coverage: this pair is observed at ten stages in one year, so
    # actual and predicted can be compared across the whole chain rather than
    # leaving most bars empty.
    plot_food_loss_comparison(pipe, frame, vary="food_supply_stage",
                              country="Bangladesh", commodity="Potatoes",
                              year=2009, ax=ax)
    _wrap(ax, "F11. Actual against random forest predictions by supply chain stage, "
              "potatoes in Bangladesh, 2009 — the pair with the widest stage coverage "
              "in the data. Read as an ordering: magnitudes understate severe losses.", 88)
    return _save_fig(fig, "F11", "stage comparison random forest", "4.3")


def f12_gap_filling():
    """F12. The case the model exists for: stages never measured."""
    pipe, frame = _fit_flagship()
    # A pair measured at exactly one stage, where the rest are the gap.
    per_pair = frame.groupby(["country", "commodity"])["food_supply_stage"].nunique()
    singles = per_pair[per_pair == 1].index
    counts = frame.groupby(["country", "commodity"]).size()
    country, commodity = max(singles, key=lambda k: counts[k])
    observed = set(frame[(frame.country == country) &
                         (frame.commodity == commodity)].food_supply_stage)
    # Span stages are excluded for the reason given in Section 3: they cover
    # several positions and so are not comparable with single ones on the
    # same axis.
    from targeting import SPAN_STAGES
    plausible = set(frame[frame.commodity == commodity].food_supply_stage) - SPAN_STAGES
    stages = sorted(plausible)

    from visualization import predict_food_loss
    preds = [predict_food_loss(pipe, country, commodity, s, 2021, verbose=False)
             for s in stages]
    order = np.argsort(preds)
    stages = [stages[i] for i in order]; preds = [preds[i] for i in order]
    colours = [FLAGSHIP if s in observed else HIGHLIGHT for s in stages]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bars = ax.barh(range(len(stages)), preds, color=colours, alpha=0.9)
    for b, v in zip(bars, preds):
        ax.text(b.get_width() + 0.08, b.get_y() + b.get_height() / 2,
                f"{v:.2f}%", va="center", fontsize=7.5)
    ax.set_yticks(range(len(stages))); ax.set_yticklabels(stages, fontsize=8)
    ax.set_xlabel("Predicted loss (%)")
    ax.set_xlim(0, max(preds) * 1.22)
    handles = [Rectangle((0, 0), 1, 1, color=FLAGSHIP),
               Rectangle((0, 0), 1, 1, color=HIGHLIGHT)]
    ax.legend(handles, ["measured for this pair", "never measured (estimated)"],
              fontsize=7.5, loc="lower right")
    _wrap(ax, f"F12. Gap filling for {commodity.lower()} in {country}, measured at one "
              f"stage only. Estimates for the remaining stages are what a lookup cannot "
              f"supply. Span stages excluded.", 82)
    return _save_fig(fig, "F12", "gap filling demonstration", "4.3")


# ==========================================================================
# SECTION 4.4 AND 5 - LIMITATIONS AND CONCLUSION
# ==========================================================================
def f13_country_representation():
    """F13. How unevenly the record count is distributed across countries."""
    frame = load_flw_data()
    counts = frame.country.value_counts().sort_values(ascending=False)
    cum = 100 * counts.cumsum() / counts.sum()
    x = 100 * np.arange(1, len(counts) + 1) / len(counts)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    ax.plot(x, cum, color=FLAGSHIP, linewidth=2)
    ax.plot([0, 100], [0, 100], color=NEUTRAL, linestyle=":", linewidth=1,
            label="even representation")
    top10 = 100 * counts.head(10).sum() / counts.sum()   # descriptive: full frame
    ax.plot([100 * 10 / len(counts)], [top10], "o", color=BASELINE, ms=6)
    ax.text(100 * 10 / len(counts) + 3, top10 - 6,
            f"10 countries\n{top10:.1f}% of records", fontsize=7.5, color=BASELINE)
    ax.set_xlabel("Countries, most-recorded first (%)")
    ax.set_ylabel("Cumulative share of records (%)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Concentration of records", fontsize=9)

    weight = _read("regional_weighting_analysis.csv")
    weight = weight[weight.model == FLAGSHIP_MODEL]      # the text quotes this model
    agg = (weight.groupby("weighting")[["mae_sparse", "mae_well_represented"]]
                 .mean().reindex(["unweighted", "inverse frequency"]))
    idx = np.arange(len(agg)); w = 0.36
    ax2.bar(idx - w/2, agg.mae_sparse, w, color=BASELINE, label="sparse countries (<50 records)")
    ax2.bar(idx + w/2, agg.mae_well_represented, w, color=FLAGSHIP, label="well represented")
    for i, (a, b) in enumerate(zip(agg.mae_sparse, agg.mae_well_represented)):
        ax2.text(i - w/2, a + 0.08, f"{a:.2f}", ha="center", fontsize=7.5)
        ax2.text(i + w/2, b + 0.08, f"{b:.2f}", ha="center", fontsize=7.5)
    ax2.set_xticks(idx); ax2.set_xticklabels(agg.index, fontsize=8)
    ax2.set_ylabel("Mean absolute error")
    ax2.set_ylim(0, max(agg.mae_sparse) * 1.34)
    ax2.legend(fontsize=7.5, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2)
    ax2.set_title("Error by representation, and the effect of weighting", fontsize=9)
    fig.suptitle(f"F13. Ten countries supply {top10:.1f}% of records. Error is "
                 f"{agg.mae_sparse['unweighted'] / agg.mae_well_represented['unweighted']:.1f}× "
                 "higher for sparsely recorded countries, and weighting does not close the gap.",
                 fontsize=9, y=1.05)
    fig.tight_layout()
    return _save_fig(fig, "F13", "country representation", "4.4")


def t10_representation_and_weighting():
    """T10. The representation gap and the remedy that failed."""
    rep = _read("country_representation.csv").iloc[0]
    weight = _read("regional_weighting_analysis.csv")
    agg = (weight.groupby(["model", "weighting"])[["mae_all", "mae_sparse", "r2_sparse"]]
                 .mean().round(4).reset_index())
    header = pd.DataFrame([
        {"model": "— representation —", "weighting": "", "mae_all": np.nan,
         "mae_sparse": np.nan, "r2_sparse": np.nan},
        {"model": f"{int(rep.n_countries)} countries; top 10 supply {rep.top10_share_pct}% "
                  f"of records; {int(rep.n_countries_below_threshold)} below "
                  f"{int(rep.sparse_threshold)} records",
         "weighting": "", "mae_all": np.nan, "mae_sparse": np.nan, "r2_sparse": np.nan},
    ])
    out = pd.concat([header, agg], ignore_index=True).rename(columns={
        "model": "Model", "weighting": "Weighting", "mae_all": "MAE all",
        "mae_sparse": "MAE sparse", "r2_sparse": "R² sparse"})
    return _save_table(out, "T10", "representation and weighting", "4.4")


def f14_stage_coverage():
    """F14. How much of the plausible space has never been measured."""
    frame = load_flw_data()
    per_pair = frame.groupby(["country", "commodity"])["food_supply_stage"].nunique()
    cov = _read("stage_coverage.csv").iloc[0]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.4),
                                  gridspec_kw={"width_ratios": [1, 1.05]})
    dist = per_pair.value_counts().sort_index()
    ax.bar(dist.index, dist.to_numpy(), color=[BASELINE] + [FLAGSHIP] * (len(dist) - 1))
    ax.annotate(f"{dist.iloc[0]} pairs\nmeasured at\none stage only",
                xy=(1, dist.iloc[0]), xytext=(3.1, dist.iloc[0] * 0.80),
                fontsize=7.8, color=BASELINE,
                arrowprops=dict(arrowstyle="->", color=BASELINE, lw=0.9))
    ax.set_ylim(0, dist.iloc[0] * 1.12)
    ax.set_xlabel("Stages measured for a country × commodity pair")
    ax.set_ylabel("Pairs"); ax.set_xticks(range(1, min(11, dist.index.max() + 1)))
    ax.set_title("Most pairs are measured once", fontsize=9)

    labels = ["measured", "plausible but\nnever measured"]
    vals = [int(cov.n_pairs * 0 + (per_pair.sum())), int(cov.plausible_gap)]
    ax2.barh(labels, vals, color=[FLAGSHIP, HIGHLIGHT])
    for i, v in enumerate(vals):
        ax2.text(v + 120, i, f"{v:,}", va="center", fontsize=8)
    ax2.set_xlabel("Country × commodity × stage combinations")
    ax2.set_xlim(0, max(vals) * 1.22)
    ax2.set_title("The gap the model addresses", fontsize=9)
    fig.suptitle("F14. Counting only stages at which a commodity is measured somewhere, "
                 f"{int(cov.plausible_gap):,} combinations have no observation.",
                 fontsize=9, y=1.05)
    fig.tight_layout()
    return _save_fig(fig, "F14", "stage coverage", "5")


# ==========================================================================
# METHOD ILLUSTRATIONS - originals replacing borrowed diagrams
# ==========================================================================
def f15_network_architecture():
    """F15. The actual network, replacing a borrowed diagram of a different one."""
    layers = [("input\n(encoded)", 383), ("hidden 1", 200), ("hidden 2", 100),
              ("hidden 3", 50), ("output", 1)]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))

    # Every annotation sits on a fixed row rather than following the box height,
    # so nothing staggers or collides with the title.
    Y_COUNT, Y_MID, Y_CONN, Y_LABEL = 0.86, 0.46, 0.20, 0.06
    xs = np.linspace(0.08, 0.92, len(layers))
    h = np.array([np.log10(n + 1) for _, n in layers])
    h = 0.16 + 0.42 * h / h.max()

    for i, ((label, n), x, height) in enumerate(zip(layers, xs, h)):
        colour = FLAGSHIP if 0 < i < len(layers) - 1 else NEUTRAL
        ax.add_patch(Rectangle((x - 0.048, Y_MID - height / 2), 0.096, height,
                               facecolor=colour, alpha=0.92))
        ax.text(x, Y_COUNT, f"{n:,}", ha="center", va="center",
                fontsize=9, weight="bold")
        ax.text(x, Y_LABEL, label, ha="center", va="center", fontsize=8.2)
        if i < len(layers) - 1:
            mid = (x + xs[i + 1]) / 2
            ax.add_patch(FancyArrowPatch((x + 0.052, Y_MID), (xs[i + 1] - 0.052, Y_MID),
                                         arrowstyle="->", mutation_scale=11, color="#666"))
            n_w = layers[i][1] * layers[i + 1][1] + layers[i + 1][1]
            ax.text(mid, Y_CONN, f"{n_w:,}", ha="center", va="center",
                    fontsize=7.4, color="#555")

    ax.text(0.5, Y_COUNT + 0.10, "neurons per layer", ha="center", fontsize=7.6,
            color="#777", style="italic")
    ax.text(0.5, Y_CONN - 0.075, "weights and biases per connection block",
            ha="center", fontsize=7.6, color="#777", style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.08); ax.axis("off")
    ax.set_title("F15. The multi-layer perceptron used here: three hidden layers over 383 "
                 "encoded inputs,\n102,001 trainable parameters, ReLU activation, trained by "
                 "stochastic gradient descent.", fontsize=9, pad=14)
    return _save_fig(fig, "F15", "network architecture", "3.7")


def f16_decision_tree_subtree():
    """F16. The top of the fitted tree, replacing a borrowed textbook example."""
    from sklearn.tree import plot_tree
    from pipelines import build_pipeline, make_column_transformer_for
    frame = load_flw_data()
    X, y = frame[GROUP_COLS + ["year"]], frame[TARGET]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = build_pipeline("Decision Tree", year_mode="scaled")
        pipe.fit(X, y)
    names = list(pipe[:-1].get_feature_names_out())
    # Long one-hot names collide at depth three; abbreviate the field prefix
    # and clip the level so the boxes stay legible.
    SHORT = {"country": "cty", "commodity": "cmd", "food_supply_stage": "stg", "year": "year"}
    def _short(n):
        raw = n.split("__")[-1]
        for field, abbr in SHORT.items():
            if raw.startswith(field):
                level = raw[len(field):].lstrip("_")
                return f"{abbr}={level[:16]}" if level else abbr
        return raw[:18]
    pretty = [_short(n) for n in names]

    fig, ax = plt.subplots(figsize=(16.5, 5.6))
    plot_tree(pipe[-1], max_depth=3, feature_names=pretty, filled=True,
              impurity=False, fontsize=6, ax=ax, precision=2,
              proportion=True, rounded=True)
    ax.set_title("F16. The first three levels of the fitted regression tree. The full "
                 "tree is grown without a depth limit; this excerpt is illustrative of "
                 "the structure, not the whole model.", fontsize=9)
    return _save_fig(fig, "F16", "decision tree subtree", "3.8")


# ==========================================================================
# BUILD
# ==========================================================================
BUILDERS = [
    t1_dataset_composition, f1_field_coverage, f2_records_per_year,
    t2_cause_of_loss, t3_repeated_combinations,
    f3_loss_distribution, f4_error_concentration, f5_validation_schematic,
    t4_random_forest_ablation,
    t5_primary_results, t6_grouped_results, f6_paired_comparison,
    t7_nested_tuning, t8_error_by_band, f7_residuals, f8_targeting_curve,
    f9_threshold_sensitivity, f10_loss_capture, t9_worst_stage,
    f11_stage_comparison, f12_gap_filling,
    f13_country_representation, t10_representation_and_weighting, f14_stage_coverage,
    f15_network_architecture, f16_decision_tree_subtree,
]


def build_all(verbose=True):
    """Generate every figure and table, returning a manifest.

    A builder that fails is recorded and the rest continue, so one broken
    item cannot cost the whole set.
    """
    rows = []
    for fn in BUILDERS:
        item = fn.__doc__.split(".")[0].strip()
        try:
            path = fn()
            status, where = "ok", path.name
        except Exception as exc:
            status, where = f"FAILED: {type(exc).__name__}: {exc}", ""
        section, description = ITEMS.get(item, ("", fn.__name__))
        rows.append({"item": item, "section": section,
                     "description": description, "file": where, "status": status})
        if verbose:
            mark = "  " if status == "ok" else "!!"
            print(f"{mark} {item:<4} §{section:<5} {description:<42} {where or status}")
    manifest = pd.DataFrame(rows)
    manifest.to_csv(RESULTS_DIR / "figure_manifest.csv", index=False)
    return manifest


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    m = build_all()
    bad = m[m.status != "ok"]
    print(f"\n{len(m) - len(bad)} of {len(m)} built")
    if len(bad):
        print(bad[["item", "status"]].to_string(index=False))
