"""Stage 10 - handoff: production pipeline, raw-row tests, model card, Kaggle submission.

The saved pipeline's FIRST step is the Stage 1 stateless cleaning, wrapped in a
FunctionTransformer, so a raw production row gets exactly the cleaning the training data got.
A pipeline that assumes someone cleaned the input first is a pipeline that will be fed dirty
input eventually.

The Kaggle submission is produced here and only here, after Stage 9 has signed off. The pipeline
was chosen on local evidence alone; this step does not revisit that choice.

Rerunnable:  python code/10_handoff.py
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

SKILLS = pathlib.Path(r"C:\dev\important repo\Data Scientist Skill Set")
for p in ("ds-validation-design/scripts", "ds-method-comparison/scripts",
          "ds-evaluation/scripts"):
    sys.path.insert(0, str(SKILLS / p))
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from elapsed import elapsed_min   # noqa: E402
from features import CATS, RawPrep   # noqa: E402
from screen import fresh_copy     # noqa: E402

ROOT = HERE.parent
REPO = ROOT.parent
ART = ROOT / "artifacts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sys.argv = ["10"]
js8 = _load("js8", "08_joint_search.py")


def main() -> None:
    res = json.loads((ART / "search" / "result.json").read_text())
    ev = json.loads((ART / "final_eval.json").read_text())
    fam = res["pick_params"]["family"]
    feats = js8.NUM + js8.CAT

    # ---- build the production pipeline: cleaning FIRST ----------------------
    inner = js8.build(fam, res["pick_choices"],
                      js8.FAMILIES[fam](js8.fixed_trial(res["pick_params"])))

    from sklearn.pipeline import Pipeline
    production = Pipeline([("clean_raw", FunctionTransformer(RawPrep(feats), validate=False)),
                           ("model", inner)])

    # ---- fit on ALL labelled rows -------------------------------------------
    raw_train = pd.read_csv(REPO / "data/raw/train.csv")
    y_all = (raw_train["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
    t0 = time.time()
    production.fit(raw_train, y_all)
    fit_s = round(time.time() - t0, 1)
    print(f"fitted on all {len(raw_train):,} labelled rows in {fit_s}s")

    checks: dict[str, object] = {}

    # ---- raw rows end to end -------------------------------------------------
    raw_sample = raw_train.drop(columns=["Will_Buy_EV"]).head(5).copy()
    t0 = time.time()
    p_raw = production.predict_proba(raw_sample)[:, 1]
    checks["raw_rows_scored"] = [round(float(v), 6) for v in p_raw]
    checks["latency_ms_for_5_rows"] = round((time.time() - t0) * 1000, 1)

    # a deliberately dirty row: whitespace, wrong case, an impossible value
    dirty = raw_sample.head(1).copy()
    dirty.loc[:, "City_Type"] = "  Urban "
    dirty.loc[:, "Age"] = 999            # impossible: clean_raw must null it
    try:
        p_dirty = float(production.predict_proba(dirty)[:, 1][0])
        checks["dirty_row_scored"] = round(p_dirty, 6)
        checks["dirty_row_ok"] = True
    except Exception as e:                                     # noqa: BLE001
        checks["dirty_row_ok"] = False
        checks["dirty_row_error"] = f"{type(e).__name__}: {e}"

    # ---- duplicate determinism ----------------------------------------------
    one = raw_sample.head(1)
    twice = production.predict_proba(pd.concat([one, one]))[:, 1]
    checks["duplicate_row_identical"] = bool(np.isclose(twice[0], twice[1]))

    # ---- directional expectation --------------------------------------------
    # Stage 3 measured a monotone target rate 0.0057 -> 0.5183 across concern levels 1..5.
    probe = pd.concat([raw_sample.head(1)] * 5, ignore_index=True)
    probe["Environmental_Concern_Level"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    pr = production.predict_proba(probe)[:, 1]
    checks["concern_direction"] = [round(float(v), 6) for v in pr]
    checks["concern_monotone_increasing"] = bool(np.all(np.diff(pr) > 0))

    # ---- Kaggle submission ---------------------------------------------------
    raw_test = pd.read_csv(REPO / "data/raw/test.csv")
    t0 = time.time()
    p_test = production.predict_proba(raw_test)[:, 1]
    sub = pd.DataFrame({"id": raw_test["id"], "Will_Buy_EV": p_test})
    subdir = ROOT / "subs"
    subdir.mkdir(exist_ok=True)
    sub.to_csv(subdir / "submission_dsrerun.csv", index=False)
    checks["kaggle_rows"] = len(sub)
    checks["kaggle_scoring_seconds"] = round(time.time() - t0, 1)
    checks["kaggle_pred_summary"] = {
        "min": round(float(p_test.min()), 6), "mean": round(float(p_test.mean()), 6),
        "max": round(float(p_test.max()), 6),
        "train_base_rate": round(float(y_all.mean()), 6)}

    # ---- save ----------------------------------------------------------------
    import sklearn
    import lightgbm
    import xgboost
    meta = {
        "pick": res["pick"], "family": fam, "choices": res["pick_choices"],
        "params": res["pick_params"],
        "fitted_on_rows": int(len(raw_train)), "fit_seconds": fit_s,
        "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                     "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "seeds": {"split_seed": 42, "model_random_state": 0, "optuna_sampler_seed": 0},
        "test_auc": ev["TEST_auc"], "checks": checks,
    }
    joblib.dump({"pipeline": production, "meta": meta}, ART / "pipeline.joblib")
    (ART / "handoff.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(checks, indent=2, default=str))
    print(f"\nsubmission: {subdir / 'submission_dsrerun.csv'}  ({len(sub):,} rows)")
    print(f"elapsed {elapsed_min()} min")


if __name__ == "__main__":
    main()
