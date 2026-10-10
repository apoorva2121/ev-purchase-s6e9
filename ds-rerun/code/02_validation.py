"""Stage 2 - validation design, frozen test set, noise floor.

The split itself is carved by the bundled make_split.py (run separately, output in
artifacts/split/). This script does the four things the gate also requires:

  1. test-set precision estimate for ROC-AUC (Hanley-McNeil)
  2. the noise floor, measured in the primary metric's own units
  3. split sanity checks, including adversarial validation against the real Kaggle test set
  4. the grouped-by-income diagnostic Apoorva approved, to separate generator memory
     from transferable signal

Rerunnable end to end:  python code/02_validation.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GroupKFold, RepeatedStratifiedKFold,
                                     cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Standard error of ROC-AUC (Hanley & McNeil 1982)."""
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    var = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc**2)
           + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return float(np.sqrt(var))


def frames():
    tr = pd.read_parquet(ART / "train_clean.parquet")
    s = load_split(str(ART / "split" / "split.csv"), ids=tr["id"])
    feats = [c for c in tr.columns if c not in ("id", "Will_Buy_EV")]
    X = tr[feats].copy()
    for c in CATS:
        X[c] = X[c].astype(str)
    y = (tr["Will_Buy_EV"] == "Yes").astype(int)
    return tr, X, y, s, feats


def simple_pipe(nums, cats):
    """One simple pipeline, used only to measure the noise floor."""
    pre = ColumnTransformer([("n", StandardScaler(), nums),
                             ("c", OneHotEncoder(handle_unknown="ignore"), cats)])
    return Pipeline([("p", pre), ("m", LogisticRegression(max_iter=1000))])


def main() -> None:
    out: dict[str, object] = {}
    tr, X, y, s, feats = frames()
    nums = [c for c in feats if c not in CATS]
    Xtr, ytr = X[s.train_mask], y[s.train_mask]

    spec = json.loads((ART / "split" / "split_spec.json").read_text())
    out["split_spec"] = spec

    # --- 1. test-set precision -------------------------------------------------
    n_test = spec["sizes"]["test"]
    rate = float(spec["target_by_part"]["test"]["Yes"])
    n_pos = int(round(n_test * rate))
    n_neg = n_test - n_pos
    # evaluated at a plausible operating AUC, taken from the one-fold probe (0.937)
    se = hanley_mcneil_se(0.937, n_pos, n_neg)
    out["test_precision"] = {
        "n_test": n_test, "n_pos": n_pos, "n_neg": n_neg,
        "auc_assumed": 0.937,
        "hanley_mcneil_se": round(se, 6),
        "ci95_halfwidth": round(1.96 * se, 6),
    }

    # --- 2. noise floor, in ROC-AUC -------------------------------------------
    t0 = time.time()
    pipe = simple_pipe(nums, CATS)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
    scores = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=5)
    fold_sd = float(scores.std(ddof=1))
    repeat_means = scores.reshape(5, 5).mean(axis=1)
    repeat_sd = float(repeat_means.std(ddof=1))
    out["noise_floor"] = {
        "metric": "roc_auc",
        "pipeline": "StandardScaler+OneHot -> LogisticRegression(max_iter=1000)",
        "cv": "RepeatedStratifiedKFold(5x5, random_state=0)",
        "mean_auc": round(float(scores.mean()), 6),
        "fold_sd": round(fold_sd, 6),
        "repeat_sd_NOISE_FLOOR": round(repeat_sd, 6),
        "fold_sd_over_sqrt_k": round(fold_sd / np.sqrt(5), 6),
        "seconds": round(time.time() - t0, 1),
    }

    # --- 3. sanity checks ------------------------------------------------------
    checks: dict[str, object] = {}
    checks["target_rate_matches_across_parts"] = (
        len({round(float(spec["target_by_part"][p]["Yes"]), 4)
             for p in ("train", "holdout", "test")}) == 1)

    # adversarial validation: our training portion vs the REAL Kaggle test file.
    # Near 0.5 means the split and the data are sound; high means train/serving skew.
    te = pd.read_parquet(ART / "test_clean.parquet")
    Xte = te[feats].copy()
    for c in CATS:
        Xte[c] = Xte[c].astype(str)
    rng = np.random.default_rng(0)
    ia = rng.choice(len(Xtr), 60000, replace=False)
    ib = rng.choice(len(Xte), 60000, replace=False)
    Xadv = pd.concat([Xtr.iloc[ia], Xte.iloc[ib]], ignore_index=True)
    for c in CATS:
        Xadv[c] = Xadv[c].astype("category")
    d = np.r_[np.zeros(60000), np.ones(60000)]
    t0 = time.time()
    adv = cross_val_score(
        HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0),
        Xadv, d, cv=5, scoring="roc_auc", n_jobs=5).mean()
    checks["adversarial_train_vs_kaggle_test_auc"] = round(float(adv), 5)
    checks["adversarial_verdict"] = (
        "near 0.5 - train and Kaggle test are drawn alike" if adv < 0.55
        else "ELEVATED - inspect which features separate them")
    checks["adversarial_seconds"] = round(time.time() - t0, 1)
    out["sanity_checks"] = checks

    # --- 4. grouped-by-income diagnostic --------------------------------------
    # Approved by Apoorva 2026-09-27. NOT the primary split: it measures a condition
    # that describes only 0.6% of the real test set. Its job is to quantify how much
    # of the stratified score comes from memorising exact income values.
    t0 = time.time()
    groups = tr.loc[s.train_mask, "Annual_Income_USD"].values
    gkf = GroupKFold(n_splits=5)
    g_scores = cross_val_score(simple_pipe(nums, CATS), Xtr, ytr, groups=groups,
                               cv=gkf, scoring="roc_auc", n_jobs=5)
    strat_mean = float(scores.mean())
    out["grouped_diagnostic"] = {
        "scheme": "GroupKFold(5) on exact Annual_Income_USD",
        "mean_auc": round(float(g_scores.mean()), 6),
        "fold_sd": round(float(g_scores.std(ddof=1)), 6),
        "stratified_mean_auc": round(strat_mean, 6),
        "gap_attributable_to_exact_value_memory": round(strat_mean - float(g_scores.mean()), 6),
        "seconds": round(time.time() - t0, 1),
        "note": ("Diagnostic only. 99.4% of Kaggle test rows share an exact income value "
                 "with train, so the grouped condition is NOT the deployment condition."),
    }

    (ART / "stage2_validation.json").write_text(json.dumps(out, indent=2))

    # --- write the noise floor into search_space.json --------------------------
    ss = json.loads((ROOT / "search_space.json").read_text())
    ss["cv"] = {"scheme": "stratified (make_split.py), fixed 5 folds",
                "n_splits": 5, "n_repeats": 1, "groups": None, "seed": 42}
    ss["noise_floor_sd"] = round(repeat_sd, 6)
    ss["noise_floor_definition"] = (
        "sd of the CV mean across 5 repeats of RepeatedStratifiedKFold(5x5) on the simple "
        "logistic pipeline, measured in ROC-AUC. fold_sd="
        f"{fold_sd:.6f} recorded alongside for reference.")
    (ROOT / "search_space.json").write_text(json.dumps(ss, indent=2))

    print(json.dumps(out, indent=2))
    print(f"\nNOISE FLOOR (repeat_sd, ROC-AUC) = {repeat_sd:.6f}")
    print(f"elapsed {elapsed_min()} min")


if __name__ == "__main__":
    main()
