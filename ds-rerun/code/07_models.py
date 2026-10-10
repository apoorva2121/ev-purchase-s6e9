"""Stage 7 - baselines, model families, imbalance, learnability verdict.

Order matters: baselines first, then a cheap family screen on the Stage 4-6 reference
preprocessing, then a learning curve on the best two. Nothing is tuned here; tuning is the
Stage 8 joint search.

Reference preprocessing = the Stage 6 winner (income target-encoding ladder: exact + //50 + //500),
because the skill requires families to be screened on the preprocessing they will actually use.
Screening families on raw columns would rank them on a pipeline nobody intends to ship.

Rerunnable:  python code/07_models.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SKILLS = pathlib.Path(r"C:\dev\important repo\Data Scientist Skill Set")
for p in ("ds-validation-design/scripts", "ds-method-comparison/scripts",
          "ds-model-selection/scripts"):
    sys.path.insert(0, str(SKILLS / p))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_folds import load_split           # noqa: E402
from elapsed import elapsed_min           # noqa: E402
from features import CATS, MultiTE, ladder_specs, precise   # noqa: E402
from rule_baseline import RuleBaseline    # noqa: E402
from screen import evaluate               # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
LADDER = ladder_specs((50, 500))


def business_rule(X: pd.DataFrame) -> np.ndarray:
    """Two-line rule read off the Stage 3 target-rate tables. Fits nothing, sees no labels.

        Environmental_Concern_Level  0.0057 / 0.0214 / 0.1109 / 0.2488 / 0.5183  (monotone)
        Subsidy_Available            No 0.0058  vs  Yes 0.2747
        Range_Anxiety_Level          High 0.0014, Medium 0.0417, Low 0.1890

    The three multiply because each of them has a near-deterministic "no" level: a buyer with
    no subsidy, or the lowest concern, or high range anxiety, essentially never buys.
    """
    concern = X["Environmental_Concern_Level"].astype(float)
    subsidy = (X["Subsidy_Available"].astype(str) == "Yes").astype(float)
    anxiety = X["Range_Anxiety_Level"].astype(str).map(
        {"Low": 1.0, "Medium": 0.25, "High": 0.0}).fillna(0.5).astype(float)
    return (concern * subsidy * anxiety).to_numpy()


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

    te_cols = MultiTE(LADDER).new_cols()

    def lin_ref():
        """Linear reference pipeline: ladder TE -> scale numerics -> one-hot categoricals."""
        return [("te", MultiTE(LADDER)),
                ("pre", ColumnTransformer([
                    ("n", StandardScaler(), nums + te_cols),
                    ("c", OneHotEncoder(handle_unknown="ignore"), CATS)]))]

    cache = ART / "stage7_cache"
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

    out: dict[str, object] = {}

    # ---------------- Step 1: baselines, all four ------------------------------
    plain_pre = ColumnTransformer([("n", StandardScaler(), nums),
                                   ("c", OneHotEncoder(handle_unknown="ignore"), CATS)])
    base = {
        "b1_trivial_prior": DummyClassifier(strategy="prior"),
        "b2_business_rule": RuleBaseline(business_rule),
        "b3_logistic_plain": Pipeline([("pre", plain_pre),
                                       ("m", LogisticRegression(max_iter=1000))]),
        "b4_tree_depth3": Pipeline([("pre", plain_pre),
                                    ("m", DecisionTreeClassifier(max_depth=3, random_state=0))]),
    }
    out["baselines"] = precise(run("baselines", base, Xl)).to_dict("records")

    # ---------------- Step 2: family screen on the reference preprocessing -----
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb

    fams = {
        "logistic": Pipeline([*lin_ref(), ("m", LogisticRegression(max_iter=1000))]),
        "hgb_sklearn": Pipeline([("te", MultiTE(LADDER)),
                                 ("m", HistGradientBoostingClassifier(
                                     categorical_features="from_dtype", random_state=0))]),
        "lightgbm": Pipeline([("te", MultiTE(LADDER)),
                              ("m", lgb.LGBMClassifier(random_state=0, verbose=-1, n_jobs=4))]),
        "xgboost": Pipeline([("te", MultiTE(LADDER)),
                             ("m", xgb.XGBClassifier(enable_categorical=True, tree_method="hist",
                                                     random_state=0, n_jobs=4,
                                                     eval_metric="auc"))]),
        "catboost": Pipeline([("te", MultiTE(LADDER)),
                              ("m", cb.CatBoostClassifier(cat_features=CATS, random_seed=0,
                                                          verbose=0,
                                                          allow_writing_files=False))]),
    }
    out["families"] = precise(run("families", fams, Xg)).to_dict("records")

    # ---------------- Step 3: imbalance ----------------------------------------
    # Positive rate 17.46% - moderate, not rare. The skill's step 1 is "change nothing but the
    # metric and threshold"; ROC-AUC is threshold-free and insensitive to class ratio, so no
    # change is expected. Screened anyway rather than assumed.
    imb = {
        "default": Pipeline([("te", MultiTE(LADDER)),
                             ("m", lgb.LGBMClassifier(random_state=0, verbose=-1, n_jobs=4))]),
        "class_weight_balanced": Pipeline([("te", MultiTE(LADDER)),
                                           ("m", lgb.LGBMClassifier(class_weight="balanced",
                                                                    random_state=0, verbose=-1,
                                                                    n_jobs=4))]),
        "scale_pos_weight": Pipeline([("te", MultiTE(LADDER)),
                                      ("m", lgb.LGBMClassifier(
                                          scale_pos_weight=float((y == 0).sum() / (y == 1).sum()),
                                          random_state=0, verbose=-1, n_jobs=4))]),
    }
    out["imbalance"] = precise(run("imbalance", imb, Xg)).to_dict("records")

    # ---------------- Step 4: learning curves on the best two ------------------
    lc_file = ART / "stage7_learning_curves.json"
    if lc_file.exists():
        lc = json.loads(lc_file.read_text())
    else:
        lc = {}
        for name, pipe, X in [("lightgbm", fams["lightgbm"], Xg),
                              ("logistic", fams["logistic"], Xl)]:
            t0 = time.time()
            sizes, trs, vas = learning_curve(
                pipe, X, y, cv=s.folds, scoring="roc_auc",
                train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], n_jobs=5)
            lc[name] = {"sizes": [int(v) for v in sizes],
                        "train_auc": [round(float(v), 6) for v in trs.mean(axis=1)],
                        "val_auc": [round(float(v), 6) for v in vas.mean(axis=1)],
                        "seconds": round(time.time() - t0, 1)}
            print(f"learning curve {name}: val {lc[name]['val_auc']}")
        lc_file.write_text(json.dumps(lc, indent=2))
    out["learning_curves"] = lc

    (ART / "stage7_models.json").write_text(json.dumps(out, indent=2, default=str))
    for k in ["baselines", "families", "imbalance"]:
        print(f"\n=== {k} ===")
        print(pd.DataFrame(out[k]).to_string(index=False))
    print("\n=== learning curves ===")
    for k, v in lc.items():
        print(f"{k}: sizes {v['sizes']}")
        print(f"    train {v['train_auc']}")
        print(f"    val   {v['val_auc']}")
    print(f"\nelapsed {elapsed_min()} min")


if __name__ == "__main__":
    main()
