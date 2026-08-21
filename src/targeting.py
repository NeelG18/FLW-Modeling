"""Ranking and screening evaluation.

Regression metrics ask how close a predicted percentage is. Targeting asks
whether the cells that matter surface first. A model can be poor at the
first and useful at the second, which is the case here: predictions
understate severe losses badly while still ordering them correctly.

The metrics below are chosen for a rare-positive ranking problem:

* average precision rather than ROC-AUC, because AUC is dominated by the
  many easy negatives at ~2% prevalence and reads far more favourably than
  the problem warrants
* precision, recall and lift at k, because an inspection budget is a
  fraction of the list, not a probability threshold
* share of loss captured at k, which needs no threshold at all

The severity threshold defaults to the top decile of observed losses rather
than a round number, so it means "unusually high for this data".
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from data_prep import GROUP_COLS, TARGET, RESULTS_DIR

DEFAULT_KS = (1, 5, 10, 20)
DEFAULT_THRESHOLDS = (5, 10, 15, 20, 30)
SEVERITY_QUANTILE = 0.90

# Values of food_supply_stage that span several chain positions rather than
# naming one. Comparing their loss rates against single positions is not
# like for like: a span accumulates loss across everything it covers.
SPAN_STAGES = frozenset({"Whole supply chain", "Post-harvest", "Pre-harvest", "Farm"})


def severity_threshold(y, quantile=SEVERITY_QUANTILE):
    """Loss percentage marking the top decile of observations."""
    return float(pd.Series(y).quantile(quantile))


def ranking_metrics(predictions_df, thresholds=DEFAULT_THRESHOLDS, split_col=None):
    """Rank correlation, average precision, and AUC per model.

    Average precision is reported at several thresholds so the reader can
    see whether a model's advantage depends on where severity is drawn.
    """
    rows = []
    groups = ([(None, predictions_df)] if split_col is None
              else list(predictions_df.groupby(split_col)))

    for split_value, d in groups:
        for model, g in d.groupby("model"):
            entry = {"model": model, "n": len(g),
                     "spearman": float(stats.spearmanr(g.y_pred, g.y_true).statistic)}
            if split_col:
                entry[split_col] = split_value
            for thr in thresholds:
                y_bin = (g.y_true >= thr).astype(int)
                if y_bin.nunique() > 1:
                    entry[f"ap@{thr}"] = average_precision_score(y_bin, g.y_pred)
                    entry[f"auc@{thr}"] = roc_auc_score(y_bin, g.y_pred)
                    entry[f"prevalence@{thr}"] = float(y_bin.mean())
                else:
                    entry[f"ap@{thr}"] = np.nan
            rows.append(entry)
    return pd.DataFrame(rows).round(4)


def targeting_curve(predictions_df, model_name, ks=DEFAULT_KS, threshold=None):
    """Precision, recall and lift when inspecting the top k% of the ranking.

    This is the operational view: given capacity to examine k% of cells,
    how many severe cases does the ranking surface, and how much better is
    that than examining k% at random.
    """
    d = predictions_df[predictions_df.model == model_name]
    if d.empty:
        raise ValueError(f"No predictions for {model_name!r}")
    threshold = severity_threshold(d.y_true) if threshold is None else threshold

    severe = d.y_true >= threshold
    base_rate = float(severe.mean())
    total_severe = int(severe.sum())

    rows = []
    ordered = d.sort_values("y_pred", ascending=False)
    for k in ks:
        n = max(1, int(len(d) * k / 100))
        top = ordered.head(n)
        hits = int((top.y_true >= threshold).sum())
        rows.append({
            "k_pct": k, "n_inspected": n, "n_severe_found": hits,
            "precision": hits / n,
            "recall": hits / total_severe if total_severe else np.nan,
            "lift": (hits / n) / base_rate if base_rate else np.nan,
        })
    out = pd.DataFrame(rows).round(4)
    out.attrs["threshold"] = round(threshold, 3)
    out.attrs["base_rate"] = round(base_rate, 5)
    return out


def loss_capture_curve(predictions_df, model_name, ks=DEFAULT_KS):
    """Share of total observed loss sitting in the top k% of the ranking.

    Threshold-free, and comparable against the ceiling a perfect ranking
    would reach -- which matters, because loss is not concentrated enough
    for even perfect ordering to capture most of it in a small slice.

    Loss is a rate, not a mass. Every cell counts equally regardless of
    production volume, so this answers "where should the next survey go",
    not "where is the most tonnage lost".
    """
    d = predictions_df[predictions_df.model == model_name]
    total = d.y_true.sum()
    by_pred = d.sort_values("y_pred", ascending=False).y_true.to_numpy()
    by_true = np.sort(d.y_true.to_numpy())[::-1]

    rows = []
    for k in ks:
        n = max(1, int(len(d) * k / 100))
        rows.append({
            "k_pct": k,
            "loss_captured_pct": 100 * by_pred[:n].sum() / total,
            "perfect_ranking_pct": 100 * by_true[:n].sum() / total,
            "random_pct": k,
        })
    out = pd.DataFrame(rows)
    out["efficiency_vs_perfect"] = (out.loss_captured_pct / out.perfect_ranking_pct).round(4)
    return out.round(3)


def paired_ranking_bootstrap(predictions_df, model_name, baseline, threshold=None,
                             n_boot=2000, seed=0, alpha=0.05, split_col="cutoff"):
    """Bootstrap CI for the difference in average precision against a baseline.

    Resampling happens within each split so the class balance of that split
    is preserved; average precision is undefined without positives.
    """
    threshold = (severity_threshold(predictions_df.y_true)
                 if threshold is None else threshold)
    rng = np.random.default_rng(seed)
    diffs = []

    for _, d in predictions_df.groupby(split_col):
        m = d[d.model == model_name].reset_index(drop=True)
        b = d[d.model == baseline].reset_index(drop=True)
        if m.empty or b.empty or len(m) != len(b):
            continue
        y = (m.y_true.to_numpy() >= threshold).astype(int)
        if y.sum() == 0:
            continue
        pm, pb = m.y_pred.to_numpy(), b.y_pred.to_numpy()

        boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            if y[idx].sum() == 0:
                continue
            boot.append(average_precision_score(y[idx], pm[idx])
                        - average_precision_score(y[idx], pb[idx]))
        if boot:
            diffs.append({
                "split": _, "observed_difference":
                    average_precision_score(y, pm) - average_precision_score(y, pb),
                "ci_low": float(np.percentile(boot, 100 * alpha / 2)),
                "ci_high": float(np.percentile(boot, 100 * (1 - alpha / 2))),
            })

    out = pd.DataFrame(diffs)
    if not out.empty:
        out["beats_baseline"] = out.ci_low > 0
        out["model"] = model_name
    return out.round(4)


def worst_stage_accuracy(predictions_df, exclude_spans=True):
    """How often the highest-predicted stage is the highest-observed one.

    This is the task the paper's stage comparisons illustrate. Span stages
    are excluded by default: an aggregate accumulates loss over everything
    it covers, so it would win the comparison by construction.
    """
    rows = []
    for model, d in predictions_df.groupby("model"):
        if exclude_spans:
            d = d[~d.food_supply_stage.isin(SPAN_STAGES)]
        hits = tot = 0
        chance = []
        for _, g in d.groupby(["country", "commodity", "cutoff"] if "cutoff" in d
                              else ["country", "commodity"]):
            agg = g.groupby("food_supply_stage").agg(a=("y_true", "mean"), p=("y_pred", "mean"))
            if len(agg) < 2:
                continue
            tot += 1
            hits += int(agg.p.idxmax() == agg.a.idxmax())
            chance.append(1 / len(agg))
        if tot:
            rows.append({"model": model, "n_comparisons": tot,
                         "correct_pct": 100 * hits / tot,
                         "chance_pct": 100 * float(np.mean(chance)),
                         "lift": (hits / tot) / float(np.mean(chance))})
    return pd.DataFrame(rows).round(3)


def measurement_versus_loss(df, exclude_spans=True):
    """Measurement effort against observed loss, per stage.

    Span stages are excluded by default because they cover several
    positions and so accumulate more loss by construction, which would
    manufacture the inverse relationship this table is testing for.
    """
    d = df[~df.food_supply_stage.isin(SPAN_STAGES)] if exclude_spans else df
    out = (d.groupby("food_supply_stage")
             .agg(n_measurements=(TARGET, "size"), mean_loss=(TARGET, "mean"))
             .sort_values("n_measurements", ascending=False).reset_index())
    out["pct_of_measurements"] = (100 * out.n_measurements / len(d)).round(2)
    rho = stats.spearmanr(out.n_measurements, out.mean_loss)
    out.attrs["spearman"] = round(float(rho.statistic), 4)
    out.attrs["p_value"] = round(float(rho.pvalue), 4)
    return out.round(3)


def stage_coverage(df):
    """How much of the plausible country-commodity-stage space is measured.

    A stage counts as plausible for a commodity only if that commodity is
    measured at it somewhere in the world. Counting all stages for every
    commodity inflates the gap with combinations that do not arise.
    """
    plausible = df.groupby("commodity")["food_supply_stage"].apply(set)
    observed = df.groupby(["country", "commodity"])["food_supply_stage"].apply(set)
    n_stages = df.food_supply_stage.nunique()

    naive = sum(n_stages - len(s) for s in observed)
    real = sum(len(plausible[m] - s) for (_, m), s in observed.items())
    return pd.DataFrame([{
        "n_pairs": len(observed),
        "n_stages_in_data": n_stages,
        "pairs_measured_at_one_stage": int((observed.apply(len) == 1).sum()),
        "naive_gap": naive,
        "plausible_gap": real,
        "overstatement_factor": round(naive / real, 2) if real else np.nan,
    }])


def save_table(frame, filename):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / filename
    frame.to_csv(path, index=False)
    return path
