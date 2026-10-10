"""Stages 4 and 5 - missing values and outliers, screened with two proxy models.

This data has ZERO missing values, so Stage 4 is inverted: the live question is whether the
two measured floors (income 30000, commute 5.0) should be TURNED INTO missing values, since
Stage 1 established they are censoring rather than measurements. That is screened, not assumed.

Both stages use the bundled ds-method-comparison/scripts/screen.py for the fits and the paired
Nadeau-Bengio SE. Its summarize() rounds to 4 decimals, which erases differences of the size this
problem turns on (operative noise floor 0.00050), so precise unrounded statistics are computed
alongside. The bundled keep-rule is preserved; only the printed precision is extended.

Rerunnable:  python code/04_05_missing_outliers.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SKILLS = pathlib.Path(r"C:\dev\important repo\Data Scientist Skill Set")
for p in ("ds-validation-design/scripts", "ds-method-comparison/scripts", "ds-outliers/scripts"):
    sys.path.insert(0, str(SKILLS / p))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_folds import load_split            # noqa: E402
from elapsed import elapsed_min            # noqa: E402
from outlier_transformers import AnomalyScore, RobustClipper, Winsorizer  # noqa: E402
from screen import evaluate, summarize     # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

CATS = ["Gender", "City_Type", "Current_Car_Type",
        "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
FLOORS = {"Annual_Income_USD": 30000.0, "Daily_Commute_km": 5.0}
NOISE = 0.00050


class FloorsToNaN(BaseEstimator, TransformerMixin):
    """Replace the measured censoring floors with NaN. Stateless in effect (the floor values
    are constants established at Stage 1), but kept as a pipeline step so that every variant
    is fitted and applied identically inside the fold."""

    def __init__(self, floors=None):
        self.floors = floors or FLOORS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c, v in self.floors.items():
            if c in X.columns:
                X[c] = X[c].mask(X[c] == v)
        return X


class AppendAnomalyScore(BaseEstimator, TransformerMixin):
    """DataFrame-preserving replacement for the bundled outlier_transformers.AnomalyScore.

    Why replaced: the bundled version is a ColumnTransformer-style transformer that returns a
    numpy array, which destroys the pandas `category` dtypes that HistGradientBoosting relies on
    via categorical_features="from_dtype" (it then receives the string 'Male' and raises).

    Checks covered: identical - an IsolationForest fitted on the training fold only, its
    score_samples appended as one extra feature. Same estimator, same in-fold fitting, same
    single appended column. The only difference is that the frame and its dtypes survive.
    """

    def __init__(self, nums, n_estimators=100, random_state=0):
        self.nums = nums
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y=None):
        from sklearn.ensemble import IsolationForest
        self.iso_ = IsolationForest(n_estimators=self.n_estimators,
                                    random_state=self.random_state,
                                    n_jobs=-1).fit(X[self.nums])
        return self

    def transform(self, X):
        X = X.copy()
        X["anomaly_score"] = self.iso_.score_samples(X[self.nums])
        return X


class FloorIndicator(BaseEstimator, TransformerMixin):
    """Keep the floor value AND append a binary is-at-floor column per censored feature."""

    def __init__(self, floors=None):
        self.floors = floors or FLOORS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c, v in self.floors.items():
            if c in X.columns:
                X[f"{c}__at_floor"] = (X[c] == v).astype(float)
        return X


def precise(long: pd.DataFrame) -> pd.DataFrame:
    """Unrounded paired statistics. Same maths as screen.summarize, more decimals."""
    wide = long.groupby(["fold", "name"])["score"].mean().unstack()
    ratio = float((long["n_val"] / long["n_train"]).mean())
    means = wide.mean()
    best = means.idxmax()
    rows = []
    for name in wide.columns:
        d = wide[name] - wide[best]
        n = int(d.notna().sum())
        var = d.var(ddof=1) if n > 1 else 0.0
        se = float(np.sqrt((1 / n + ratio) * var)) if n > 1 else float("nan")
        rows.append({"name": name, "mean_auc": round(float(means[name]), 6),
                     "diff_vs_best": round(float(d.mean()), 6),
                     "paired_se": round(se, 6),
                     "keep_paired": bool(name == best or float(d.mean()) >= -se),
                     "beats_noise_floor": bool(abs(float(d.mean())) > NOISE)})
    return pd.DataFrame(rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)


def main() -> None:
    tr = pd.read_parquet(ART / "train_clean.parquet")
    s = load_split(str(ART / "split" / "split.csv"), ids=tr["id"])
    p = tr[s.train_mask].reset_index(drop=True)
    y = (p["Will_Buy_EV"] == "Yes").astype(int).values
    feats = [c for c in p.columns if c not in ("id", "Will_Buy_EV")]
    nums = [c for c in feats if c not in CATS]

    Xl = p[feats].copy()                       # linear proxy view: categoricals as str
    for c in CATS:
        Xl[c] = Xl[c].astype(str)
    Xg = p[feats].copy()                       # gbdt proxy view: native categoricals
    for c in CATS:
        Xg[c] = Xg[c].astype("category")

    def lin(pre_steps, extra_nums=()):
        n = list(nums) + list(extra_nums)
        ct = ColumnTransformer([
            ("n", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                            ("sc", StandardScaler())]), n),
            ("c", OneHotEncoder(handle_unknown="ignore"), CATS)])
        return Pipeline([*pre_steps, ("pre", ct),
                         ("model", LogisticRegression(max_iter=1000))])

    def gbdt(pre_steps):
        return Pipeline([*pre_steps,
                         ("model", HistGradientBoostingClassifier(
                             categorical_features="from_dtype", random_state=0))])

    # Per-block cache. A crash in a later block must not cost the earlier blocks' fits.
    cache = ART / "stage45_cache"
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

    results: dict[str, object] = {}

    # ---------------- Stage 4: missing values ---------------------------------
    s4_lin = {
        "none": lin([]),
        "floors_nan_median_indicator": lin([("f", FloorsToNaN())]),
        "floor_indicator_kept_value": lin([("f", FloorIndicator())],
                                          extra_nums=[f"{c}__at_floor" for c in FLOORS]),
    }
    l4l = run("stage4_linear", s4_lin, Xl)
    s4_gb = {
        "none": gbdt([]),
        "floors_nan_native": gbdt([("f", FloorsToNaN())]),
        "floor_indicator_kept_value": gbdt([("f", FloorIndicator())]),
    }
    l4g = run("stage4_gbdt", s4_gb, Xg)

    results["stage4_linear"] = precise(l4l).to_dict("records")
    results["stage4_gbdt"] = precise(l4g).to_dict("records")
    results["stage4_linear_bundled"] = summarize(l4l, True, NOISE).to_dict("records")
    results["stage4_gbdt_bundled"] = summarize(l4g, True, NOISE).to_dict("records")

    # ---------------- Stage 5: outliers ---------------------------------------
    # Trees are invariant to monotone feature transforms, so capping is screened on the
    # linear proxy only; the gbdt side screens `none` against an appended anomaly score.
    s5_lin = {
        "none": lin([]),
        "winsor_01_99": Pipeline([
            ("pre", ColumnTransformer([
                ("n", Pipeline([("w", Winsorizer(0.01, 0.99)), ("sc", StandardScaler())]), nums),
                ("c", OneHotEncoder(handle_unknown="ignore"), CATS)])),
            ("model", LogisticRegression(max_iter=1000))]),
        "robust_clip_3.5": Pipeline([
            ("pre", ColumnTransformer([
                ("n", Pipeline([("r", RobustClipper(3.5)), ("sc", StandardScaler())]), nums),
                ("c", OneHotEncoder(handle_unknown="ignore"), CATS)])),
            ("model", LogisticRegression(max_iter=1000))]),
    }
    l5l = run("stage5_linear", s5_lin, Xl)

    s5_gb = {
        "none": gbdt([]),
        "anomaly_score_feature": gbdt([("a", AppendAnomalyScore(nums))]),
    }
    l5g = run("stage5_gbdt", s5_gb, Xg)

    results["stage5_linear"] = precise(l5l).to_dict("records")
    results["stage5_gbdt"] = precise(l5g).to_dict("records")

    (ART / "stage45_screen.json").write_text(json.dumps(results, indent=2, default=str))
    for k in ["stage4_linear", "stage4_gbdt", "stage5_linear", "stage5_gbdt"]:
        print(f"\n=== {k} ===")
        print(pd.DataFrame(results[k]).to_string(index=False))
    print(f"\nelapsed {elapsed_min()} min")


if __name__ == "__main__":
    main()
