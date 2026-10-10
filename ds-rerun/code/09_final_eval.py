"""Stage 9 - evaluation and sign-off. THE TEST SET IS SCORED ONCE, HERE.

Preconditions enforced before anything is scored:
  1. verify_test_untouched(split) returns True
  2. the pipeline and its hyperparameters come from artifacts/search/result.json, fixed at Stage 8
  3. the threshold and any calibrator are chosen on the HOLDOUT, never on the test set

If the test score disappoints, it is reported. Changing anything after seeing it would turn the
test set into a validation set, and the report would have to say so.

Rerunnable:  python code/09_final_eval.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

SKILLS = pathlib.Path(r"C:\dev\important repo\Data Scientist Skill Set")
for p in ("ds-validation-design/scripts", "ds-method-comparison/scripts",
          "ds-evaluation/scripts", "ds-model-selection/scripts"):
    sys.path.insert(0, str(SKILLS / p))
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cv_folds import load_split, verify_test_untouched   # noqa: E402
from elapsed import elapsed_min                          # noqa: E402
from features import CATS                                # noqa: E402
from final_eval import (bootstrap_ci, calibration_table, shuffled_label_check,  # noqa: E402
                        slice_table)
from rule_baseline import RuleBaseline                   # noqa: E402
from screen import fresh_copy                            # noqa: E402

ROOT = HERE.parent
ART = ROOT / "artifacts"

sys.argv = ["09"]                      # joint_search parses argv at import; neutralise it
import importlib.util                  # noqa: E402
_spec = importlib.util.spec_from_file_location("js8", HERE / "08_joint_search.py")
js8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(js8)          # gives us build(), FAMILIES, fixed_trial, NUM, CAT


def business_rule(X: pd.DataFrame) -> np.ndarray:
    concern = X["Environmental_Concern_Level"].astype(float)
    subsidy = (X["Subsidy_Available"].astype(str) == "Yes").astype(float)
    anxiety = X["Range_Anxiety_Level"].astype(str).map(
        {"Low": 1.0, "Medium": 0.25, "High": 0.0}).fillna(0.5).astype(float)
    return (concern * subsidy * anxiety).to_numpy()


def main() -> None:
    # ---- Precondition 1: the test set is untouched ---------------------------
    untouched = verify_test_untouched(str(ART / "split"))
    print(f"verify_test_untouched -> {untouched}")
    if not untouched:
        raise SystemExit("ABORT: the frozen test set has been touched. No test score may be "
                         "reported; the run's central claim is void.")

    res = json.loads((ART / "search" / "result.json").read_text())
    print(f"Stage 8 pick: {res['pick']}")

    tr = pd.read_parquet(ART / "train_clean.parquet")
    s = load_split(str(ART / "split" / "split.csv"), ids=tr["id"])
    feats = js8.NUM + js8.CAT
    X = tr[feats].copy()
    for c in CATS:
        X[c] = X[c].astype(str)
    y = (tr["Will_Buy_EV"] == "Yes").astype(int).to_numpy()

    Xtr, ytr = X[s.train_mask], y[s.train_mask]
    Xho, yho = X[s.holdout_mask], y[s.holdout_mask]
    Xte, yte = X[s.test_mask], y[s.test_mask]
    print(f"train {len(Xtr)}  holdout {len(Xho)}  test {len(Xte)}")

    # ---- Rebuild the Stage 8 pick exactly ------------------------------------
    fam = res["pick_params"]["family"]
    pipe = js8.build(fam, res["pick_choices"], js8.FAMILIES[fam](js8.fixed_trial(res["pick_params"])))

    out: dict[str, object] = {"pick": res["pick"], "family": fam,
                              "choices": res["pick_choices"],
                              "cv_mean_pick": res["cv_mean_pick"],
                              "holdout_stage8": res["holdout"]}

    # ---- Sanity test BEFORE the test score: shuffled labels ------------------
    # Run on the training folds. If label information reaches the pipeline through the target
    # encoder, this comes back above chance and the whole run is void.
    print("shuffled-label canary (this refits the pipeline, be patient) ...")
    real, shuf = shuffled_label_check(fresh_copy(pipe), Xtr, ytr, s.folds,
                                      scorer=lambda p, Xv, yv: roc_auc_score(
                                          yv, p.predict_proba(Xv)[:, 1]), seed=0)
    out["canary"] = {"real_auc": round(float(real), 6),
                     "shuffled_auc": round(float(shuf), 6),
                     "pass_band": "|shuffled - 0.5| < 0.005",
                     "passed": bool(abs(float(shuf) - 0.5) < 0.005)}
    print(f"  real {real:.6f}   shuffled {shuf:.6f}   "
          f"{'PASS' if out['canary']['passed'] else 'FAIL'}")
    if not out["canary"]["passed"]:
        raise SystemExit("ABORT: shuffled-label canary failed - label information is reaching "
                         "the pipeline. No test score may be reported.")

    # ---- Threshold and calibration, decided on the HOLDOUT -------------------
    fit_tr = fresh_copy(pipe).fit(Xtr, ytr)
    p_ho = fit_tr.predict_proba(Xho)[:, 1]
    out["holdout_auc_refit_train_only"] = round(float(roc_auc_score(yho, p_ho)), 6)
    tab_ho, ece_ho, brier_ho = calibration_table(yho, p_ho, n_bins=10)
    out["calibration_holdout"] = {"ece": round(float(ece_ho), 6),
                                  "brier": round(float(brier_ho), 6),
                                  "table": tab_ho.reset_index().to_dict("records")}
    out["brier_holdout"] = round(float(brier_score_loss(yho, p_ho)), 6)

    # ---- Refit on train + holdout, then score the TEST ONCE ------------------
    Xfull = pd.concat([Xtr, Xho])
    yfull = np.r_[ytr, yho]
    final = fresh_copy(pipe).fit(Xfull, yfull)
    p_te = final.predict_proba(Xte)[:, 1]

    point, lo, hi = bootstrap_ci(yte, p_te, roc_auc_score, n_boot=1000, alpha=0.05,
                                 stratified=True, seed=0)
    out["TEST_auc"] = {"point": round(float(point), 6),
                       "ci95": [round(float(lo), 6), round(float(hi), 6)]}
    print(f"\n*** TEST ROC-AUC = {point:.6f}  95% CI [{lo:.6f}, {hi:.6f}] ***")

    out["brier_test"] = round(float(brier_score_loss(yte, p_te)), 6)
    tab_te, ece_te, brier_te = calibration_table(yte, p_te, n_bins=10)
    out["calibration_test"] = {"ece": round(float(ece_te), 6),
                               "brier": round(float(brier_te), 6),
                               "table": tab_te.reset_index().to_dict("records")}

    # baselines on the SAME test rows, for an honest comparison
    rule = RuleBaseline(business_rule).fit(Xtr, ytr)
    p_rule = rule.predict_proba(Xte)[:, 1]
    rp, rl, rh = bootstrap_ci(yte, p_rule, roc_auc_score, n_boot=1000, alpha=0.05,
                              stratified=True, seed=0)
    out["TEST_business_rule_auc"] = {"point": round(float(rp), 6),
                                     "ci95": [round(float(rl), 6), round(float(rh), 6)]}
    out["TEST_trivial_auc"] = 0.5
    out["gain_over_business_rule"] = round(float(point - rp), 6)

    # ---- Slices --------------------------------------------------------------
    te_rows = tr[s.test_mask]
    slices = te_rows[["City_Type", "Gender", "Current_Car_Type", "Range_Anxiety_Level",
                      "Subsidy_Available", "Home_Charging_Possible"]].astype(str)
    slices["Environmental_Concern_Level"] = te_rows["Environmental_Concern_Level"].astype(str)
    st = slice_table(slices, yte, p_te, metric=roc_auc_score, min_n=200, n_boot=200)
    out["slices"] = st.to_dict("records") if hasattr(st, "to_dict") else str(st)

    # ---- Error analysis ------------------------------------------------------
    err = te_rows.copy()
    err["score"] = p_te
    err["y"] = yte
    fp = err[err["y"] == 0].nlargest(20, "score")
    fn = err[err["y"] == 1].nsmallest(20, "score")
    out["worst_errors"] = {
        "highest_scored_negatives": fp.drop(columns=["id"]).to_dict("records"),
        "lowest_scored_positives": fn.drop(columns=["id"]).to_dict("records")}

    # ---- Duplicate-row determinism ------------------------------------------
    one = Xte.iloc[[0]]
    twice = final.predict_proba(pd.concat([one, one]))[:, 1]
    out["duplicate_row_identical"] = bool(np.isclose(twice[0], twice[1]))

    # ---- Grouped diagnostic on the FINAL pipeline ---------------------------
    # Promised at Stage 2. The Stage 2 version used a plain logistic model and found only
    # 0.00054 attributable to exact-value memory. The final pipeline has an income target
    # encoder keyed on the exact value, so this is where the real number appears: how much of
    # the score is recovering source-row identity rather than learning transferable structure.
    from sklearn.model_selection import GroupKFold, cross_val_score
    groups = tr.loc[s.train_mask, "Annual_Income_USD"].to_numpy()
    g = cross_val_score(fresh_copy(pipe), Xtr, ytr, groups=groups,
                        cv=GroupKFold(n_splits=5), scoring="roc_auc", n_jobs=5)
    out["grouped_diagnostic_final_pipeline"] = {
        "scheme": "GroupKFold(5) on exact Annual_Income_USD, final tuned pipeline",
        "grouped_auc": round(float(g.mean()), 6),
        "stratified_cv_auc": res["cv_mean_pick"],
        "attributable_to_exact_value_memory": round(
            float(res["cv_mean_pick"]) - float(g.mean()), 6),
        "stage2_same_number_with_plain_logistic": 0.00054}
    print(f"grouped (no shared income values): {g.mean():.6f}  vs stratified "
          f"{res['cv_mean_pick']:.6f}")

    # ---- Agreement across CV / holdout / test (evidence no leak survived) ----
    out["agreement"] = {
        "cv_mean": res["cv_mean_pick"],
        "holdout": out["holdout_auc_refit_train_only"],
        "test": out["TEST_auc"]["point"],
        "test_ci95": out["TEST_auc"]["ci95"]}

    (ART / "final_eval.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("worst_errors", "slices", "calibration_test_raw",
                                   "calibration_holdout")}, indent=2, default=str))
    print(f"\nelapsed {elapsed_min()} min")


if __name__ == "__main__":
    main()
