"""Stage 3 - leakage hunt and relationship EDA. TRAINING ROWS ONLY.

The bundled leak_screen.py is run separately; this script does the parts the gate also
requires: the record-timing verdict per column with its with/without ablation, the leak
types a column-by-column screen cannot see, and the relationship EDA.

The test and holdout rows are never loaded here. Looking at feature-target relationships
on held-out rows leaks them into feature choices.

Rerunnable end to end:  python code/03_leak_eda.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

SKILLS = pathlib.Path(r"C:\dev\important repo\Data Scientist Skill Set")
sys.path.insert(0, str(SKILLS / "ds-validation-design/scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_folds import load_split  # noqa: E402
from elapsed import elapsed_min  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
ART = ROOT / "artifacts"

CATS = ["Gender", "City_Type", "Current_Car_Type",
        "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramer's V (Bergsma 2013)."""
    ct = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    phi2 = chi2 / n
    r, k = ct.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(rc - 1, kc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0


def main() -> None:
    tr = pd.read_parquet(ART / "train_clean.parquet")
    te_cols = set(pd.read_parquet(ART / "test_clean.parquet").columns)
    s = load_split(str(ART / "split" / "split.csv"), ids=tr["id"])
    p = tr[s.train_mask].reset_index(drop=True)          # TRAINING ROWS ONLY
    y = (p["Will_Buy_EV"] == "Yes").astype(int)
    feats = [c for c in p.columns if c not in ("id", "Will_Buy_EV")]

    X = p[feats].copy()
    for c in CATS:
        X[c] = X[c].astype("category")

    out: dict[str, object] = {"rows_used": len(p),
                              "note": "test and holdout rows were not loaded"}

    # --- with / without ablation for every column ------------------------------
    # The skill requires the score with and without any presumed-leak column. Run it
    # for all 13 so the leak table has a marginal contribution per column, not just a
    # univariate score.
    def cv_auc(cols):
        return float(cross_val_score(
            HistGradientBoostingClassifier(categorical_features="from_dtype",
                                           random_state=0),
            X[cols], y, cv=s.folds, scoring="roc_auc", n_jobs=5).mean())

    full = cv_auc(feats)
    ablate = {}
    for c in feats:
        rest = [f for f in feats if f != c]
        ablate[c] = round(full - cv_auc(rest), 6)
    out["cv_auc_all_features"] = round(full, 6)
    out["drop_one_delta"] = dict(sorted(ablate.items(), key=lambda kv: -kv[1]))

    # solo score of each DOMINANT column, to show "with and without" both ways
    dominant = ["Environmental_Concern_Level", "Subsidy_Available", "Annual_Income_USD"]
    out["solo_auc_of_dominant"] = {c: round(cv_auc([c]), 6) for c in dominant}
    out["cv_auc_without_all_three_dominant"] = round(
        cv_auc([f for f in feats if f not in dominant]), 6)

    # --- every column appears in the competition test file? --------------------
    # This is the decisive availability evidence: a column recorded after the outcome
    # could not be shipped in test.csv.
    out["present_in_competition_test_file"] = {c: (c in te_cols) for c in feats}

    # --- leak types a column screen cannot see ---------------------------------
    unseen: dict[str, object] = {}

    # duplicate / near-duplicate rows straddling the split
    key_all = tr[feats].astype(str).agg("|".join, axis=1)
    unseen["exact_duplicate_feature_vectors_in_full_train"] = int(
        (key_all.map(key_all.value_counts()) > 1).sum())

    # row order / id signal
    unseen["id_vs_target_spearman"] = round(float(
        stats.spearmanr(p["id"].astype(np.int64), y).statistic), 6)

    # pairwise association between features - independence is measured, not assumed
    cv_pairs = {}
    for a, b in itertools.combinations(CATS, 2):
        v = cramers_v(p[a], p[b])
        if v > 0.05:
            cv_pairs[f"{a} x {b}"] = round(v, 4)
    # categorical x binned numeric
    for a in CATS:
        for b in ["Annual_Income_USD", "Daily_Commute_km", "Age",
                  "Charging_Stations_Near_Home", "Charging_Stations_Near_Work"]:
            v = cramers_v(p[a], pd.qcut(p[b], 10, duplicates="drop"))
            if v > 0.05:
                cv_pairs[f"{a} x {b}(10q)"] = round(v, 4)
    unseen["cramers_v_above_0.05"] = dict(sorted(cv_pairs.items(), key=lambda kv: -kv[1]))

    # numeric redundancy
    sp = p[[c for c in feats if c not in CATS]].corr(method="spearman").abs()
    np.fill_diagonal(sp.values, 0)
    unseen["max_abs_spearman_between_numerics"] = round(float(sp.values.max()), 4)
    out["unseen_leak_checks"] = unseen

    # --- relationship EDA ------------------------------------------------------
    eda: dict[str, object] = {}
    for c in CATS:
        g = p.groupby(c, observed=True).apply(
            lambda d: pd.Series({"n": len(d), "rate": (d["Will_Buy_EV"] == "Yes").mean()}),
            include_groups=False)
        eda[c] = {str(k): [int(r["n"]), round(float(r["rate"]), 4)] for k, r in g.iterrows()}
    for c in [f for f in feats if f not in CATS]:
        try:
            b = pd.qcut(p[c], 10, duplicates="drop")
        except ValueError:
            b = p[c]
        g = p.groupby(b, observed=True).apply(
            lambda d: pd.Series({"n": len(d), "rate": (d["Will_Buy_EV"] == "Yes").mean()}),
            include_groups=False)
        eda[c] = {str(k): [int(r["n"]), round(float(r["rate"]), 4)] for k, r in g.iterrows()}
    out["target_rate_tables"] = eda

    (ART / "stage3_leak_eda.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "target_rate_tables"},
                     indent=2, default=str))
    print(f"\nelapsed {elapsed_min()} min")


if __name__ == "__main__":
    main()
