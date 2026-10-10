"""Stage 6c - how deep should the ladder go, and what else deserves a ladder?

Stage 6b established that the 3-rung income ladder (exact + //50 + //500) beats every single-key
scheme, and that hard-pooling the rare tail (adaptive, bin500, qbin200) actively hurts. Two
questions that leaves open, both measured here rather than assumed:

  1. Depth. If 3 rungs beat 1, do 5 rungs beat 3? Or does correlation between rungs start to cost?
  2. Scope. Income is the quasi-identifier, but Daily_Commute_km is also a censored continuous
     column with 805 distinct values. Does it deserve its own ladder? And does a composite key
     (exact income crossed with the strongest categorical) recover more per-source-row identity?

Rerunnable:  python code/06c_ladder_depth.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SKILLS = pathlib.Path(r"C:\dev\important repo\Data Scientist Skill Set")
for p in ("ds-validation-design/scripts", "ds-method-comparison/scripts"):
    sys.path.insert(0, str(SKILLS / p))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_folds import load_split  # noqa: E402
from elapsed import elapsed_min  # noqa: E402
from features import (CATS, MultiTE, _com, _inc, combo, floor_div,  # noqa: E402
                      ladder_specs, precise)
from screen import evaluate  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"


def main() -> None:
    tr = pd.read_parquet(ART / "train_clean.parquet")
    s = load_split(str(ART / "split" / "split.csv"), ids=tr["id"])
    p = tr[s.train_mask].reset_index(drop=True)
    y = (p["Will_Buy_EV"] == "Yes").astype(int).values
    feats = [c for c in p.columns if c not in ("id", "Will_Buy_EV")]
    nums = [c for c in feats if c not in CATS]

    Xl = p[feats].copy()
    for c in CATS:
        Xl[c] = Xl[c].astype(str)
    Xg = p[feats].copy()
    for c in CATS:
        Xg[c] = Xg[c].astype("category")

    inc_ladder3 = ladder_specs((50, 500))
    variants = {
        "ladder3_ref": inc_ladder3,
        "ladder2_50": ladder_specs((50,)),
        "ladder2_500": ladder_specs((500,)),
        "ladder4": ladder_specs((10, 100, 1000)),
        "ladder5": ladder_specs((10, 50, 200, 2000)),
        "ladder6": ladder_specs((5, 25, 100, 500, 5000)),
        "ladder3_plus_commute_exact": inc_ladder3 + [("com_exact", _com)],
        "ladder3_plus_commute_ladder": inc_ladder3 + [("com_exact", _com),
                                                      ("com_1", floor_div(_com, 1)),
                                                      ("com_5", floor_div(_com, 5))],
        "ladder3_plus_inc_x_concern": inc_ladder3 + [
            ("inc_x_concern", combo(_inc, "Environmental_Concern_Level"))],
        "ladder3_plus_inc_x_subsidy": inc_ladder3 + [
            ("inc_x_subsidy", combo(_inc, "Subsidy_Available"))],
        "ladder3_plus_inc_x_city": inc_ladder3 + [("inc_x_city", combo(_inc, "City_Type"))],
    }

    cache = ART / "stage6c_cache"
    cache.mkdir(exist_ok=True)

    def run(tag, pipes, X):
        f = cache / f"{tag}.parquet"
        if f.exists():
            print(f"### {tag} (cached)")
            return pd.read_parquet(f)
        print(f"### {tag}")
        long = evaluate(pipes, X, y, s.folds, scoring="roc_auc")
        long.to_parquet(f, index=False)
        return long

    gb = {n: Pipeline([("te", MultiTE(sp)),
                       ("model", HistGradientBoostingClassifier(
                           categorical_features="from_dtype", random_state=0))])
          for n, sp in variants.items()}
    res = {"gbdt": precise(run("ladder_gbdt", gb, Xg)).to_dict("records")}

    lin = {}
    for n, sp in variants.items():
        t = MultiTE(sp)
        lin[n] = Pipeline([
            ("te", t),
            ("pre", ColumnTransformer([("n", StandardScaler(), nums + t.new_cols()),
                                       ("c", OneHotEncoder(handle_unknown="ignore"), CATS)])),
            ("model", LogisticRegression(max_iter=1000))])
    res["linear"] = precise(run("ladder_linear", lin, Xl)).to_dict("records")

    (ART / "stage6c_ladder.json").write_text(json.dumps(res, indent=2, default=str))
    for k, v in res.items():
        print(f"\n=== ladder depth/scope, {k} proxy ===")
        print(pd.DataFrame(v).to_string(index=False))
    print(f"\nelapsed {elapsed_min()} min")


if __name__ == "__main__":
    main()
