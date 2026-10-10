"""Stage 6 - feature engineering, screened with two proxy models.

The candidate menu here comes from this run's own measurements, not from a generic list:

  - Stage 1/2 found that `Annual_Income_USD` is a quasi-identifier back to the 10,000-row
    source table (100% of values with >=200 rows exist there; 3.40x label overdispersion per
    exact value, 1.93x even within income bins). So an encoder keyed on the EXACT value can
    recover per-source-row propensity. That is the single highest-value candidate.
  - Stage 3 found four features with negative drop-one deltas (Gender, City_Type,
    Charging_Stations_Near_Work, Number_of_Cars_Owned) -> a feature-selection candidate.
  - Stage 3 found Environmental_Concern_Level steeply convex (0.0057 -> 0.5183 across 1-5),
    which a linear model cannot express from the raw ordinal -> a one-hot candidate for linear.
  - Stage 3 found City_Type x charging-station Cramer's V of 0.67 -> an interaction candidate.

Every encoder is fitted in-fold. sklearn's TargetEncoder cross-fits inside fit_transform
(verified on this machine, sklearn 1.7.2: 4 categories -> 20 distinct outputs), so training rows
never see their own target. AppendTE below overrides fit_transform explicitly, because
TransformerMixin's default would call fit().transform() and silently skip the cross-fitting.

Rerunnable:  python code/06_features.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

SKILLS = pathlib.Path(r"C:\dev\important repo\Data Scientist Skill Set")
for p in ("ds-validation-design/scripts", "ds-method-comparison/scripts"):
    sys.path.insert(0, str(SKILLS / p))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_folds import load_split   # noqa: E402
from elapsed import elapsed_min   # noqa: E402
from screen import evaluate       # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

CATS = ["Gender", "City_Type", "Current_Car_Type",
        "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
REDUNDANT = ["Gender", "City_Type", "Charging_Stations_Near_Work", "Number_of_Cars_Owned"]
NOISE = 0.00050


class AppendTE(BaseEstimator, TransformerMixin):
    """Append a cross-fitted target encoding of `cols` as new columns, keeping the originals.

    fit_transform is overridden deliberately: TargetEncoder only cross-fits inside its own
    fit_transform. TransformerMixin's default fit_transform is fit().transform(), which would
    give every training row the full-data encoding of its own value - a textbook target leak.
    Inside a sklearn Pipeline, .fit() calls fit_transform on intermediate steps, so training
    rows get cross-fitted values and prediction rows get the full-fold encoding. Correct.
    """

    def __init__(self, cols, keyfunc=None, random_state=0):
        self.cols = cols
        self.keyfunc = keyfunc
        self.random_state = random_state

    def _keys(self, X):
        K = X[self.cols].astype(float)
        return self.keyfunc(K) if self.keyfunc is not None else K

    def _new(self):
        return TargetEncoder(random_state=self.random_state, target_type="binary")

    def fit(self, X, y=None):
        K = self._keys(X)
        self.key_names_ = list(K.columns)
        self.te_ = self._new().fit(K, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        K = self._keys(X)
        self.key_names_ = list(K.columns)
        self.te_ = self._new()
        Z = self.te_.fit_transform(K, y)                  # cross-fitted
        return self._append(X, Z)

    def transform(self, X):
        return self._append(X, self.te_.transform(self._keys(X)))

    def _append(self, X, Z):
        # Column names come from the KEY frame, not self.cols: a keyfunc may expand one
        # source column into several keys (the income ladder emits three).
        X = X.copy()
        Z = np.asarray(Z)
        for i, name in enumerate(self.key_names_):
            X[f"{name}__te"] = Z[:, i]
        return X


class DropCols(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.cols if c in X.columns])


class CityChargingInteraction(BaseEstimator, TransformerMixin):
    """City_Type x charging-station interaction. Stage 3 measured Cramer's V 0.672 between
    City_Type and Charging_Stations_Near_Work, so their joint effect may not be additive."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        city = X["City_Type"].astype(str)
        for c in ["Charging_Stations_Near_Home", "Charging_Stations_Near_Work"]:
            X[f"{c}__x_city"] = (city + "|" + X[c].astype(int).astype(str)).astype("category")
        return X


# ===========================================================================
# APOORVA'S DECISION POINT - income target-encoding granularity
# ===========================================================================
def income_te_key(K: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw income column(s) into the KEY(S) the target encoder groups on.

    Context. `Annual_Income_USD` has 13,214 distinct values across 434,632 training rows, but
    the distribution is extremely uneven (measured at Stage 1):
        - 492 values carry >= 200 rows each; ALL 492 exist in the 10,000-row source table
        - 5,655 values carry <= 3 rows each; only 7.2% of those exist in the source table
        - the floor 30000 alone carries 61,605 rows
    So the exact value is informative where it is common and pure noise where it is rare.

    The trade-off:
      - Encoding on the EXACT value maximises generator-memory recovery but gives a
        near-random encoding to the ~5,655 values seen only once or twice per fold.
      - Encoding on a COARSER key (e.g. income // 500) pools rare values into stable buckets,
        losing exact-row identity but gaining variance.
      - A LADDER (several keys at once: exact, //50, //500) lets the model choose per region,
        at the cost of three correlated columns.

    Return a DataFrame whose columns are the keys to encode. One column = one encoder key.
    The default below is the exact value, which is the simplest thing that could work.
    """
    # Chosen by Apoorva, 2026-09-27: the ladder. Keeps exact-value identity where counts are
    # high, and pools the rare tail into coarser buckets where an exact encoding would be
    # fitted on 1-3 labels.
    inc = K["Annual_Income_USD"]
    return pd.DataFrame({
        "inc_exact": inc,
        "inc_50": (inc // 50) * 50,
        "inc_500": (inc // 500) * 500,
    }, index=K.index)


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

    def lin(steps, extra_nums=(), drop=()):
        n = [c for c in nums if c not in drop] + list(extra_nums)
        cc = [c for c in CATS if c not in drop]
        ct = ColumnTransformer([("n", StandardScaler(), n),
                                ("c", OneHotEncoder(handle_unknown="ignore"), cc)])
        return Pipeline([*steps, ("pre", ct), ("model", LogisticRegression(max_iter=1000))])

    def gb(steps):
        return Pipeline([*steps, ("model", HistGradientBoostingClassifier(
            categorical_features="from_dtype", random_state=0))])

    cache = ART / "stage6_cache"
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

    def precise(long):
        wide = long.groupby(["fold", "name"])["score"].mean().unstack()
        ratio = float((long["n_val"] / long["n_train"]).mean())
        means = wide.mean()
        best = means.idxmax()
        rows = []
        for name in wide.columns:
            d = wide[name] - wide[best]
            n = int(d.notna().sum())
            se = float(np.sqrt((1 / n + ratio) * d.var(ddof=1))) if n > 1 else float("nan")
            rows.append({"name": name, "mean_auc": round(float(means[name]), 6),
                         "diff_vs_best": round(float(d.mean()), 6),
                         "paired_se": round(se, 6),
                         "keep_paired": bool(name == best or float(d.mean()) >= -se),
                         "beats_noise_floor": bool(abs(float(d.mean())) > NOISE)})
        return pd.DataFrame(rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)

    INC, COM = ["Annual_Income_USD"], ["Annual_Income_USD", "Daily_Commute_km"]
    results = {}

    gb_pipes = {
        "base": gb([]),
        "te_income_exact": gb([("te", AppendTE(INC))]),
        "te_income_keyed": gb([("te", AppendTE(INC, keyfunc=income_te_key))]),
        "te_income_commute": gb([("te", AppendTE(COM))]),
        "drop_redundant4": gb([("d", DropCols(REDUNDANT))]),
        "city_charging_interaction": gb([("i", CityChargingInteraction())]),
    }
    results["gbdt"] = precise(run("stage6_gbdt", gb_pipes, Xg)).to_dict("records")

    lin_pipes = {
        "base": lin([]),
        "te_income_exact": lin([("te", AppendTE(INC))],
                               extra_nums=["Annual_Income_USD__te"]),
        "te_income_keyed": lin([("te", AppendTE(INC, keyfunc=income_te_key))],
                               extra_nums=["inc_exact__te", "inc_50__te", "inc_500__te"]),
        "concern_onehot": Pipeline([
            ("pre", ColumnTransformer([
                ("n", StandardScaler(), [c for c in nums if c != "Environmental_Concern_Level"]),
                ("c", OneHotEncoder(handle_unknown="ignore"),
                 CATS + ["Environmental_Concern_Level"])])),
            ("model", LogisticRegression(max_iter=1000))]),
        "drop_redundant4": lin([("d", DropCols(REDUNDANT))], drop=REDUNDANT),
    }
    results["linear"] = precise(run("stage6_linear", lin_pipes, Xl)).to_dict("records")

    (ART / "stage6_screen.json").write_text(json.dumps(results, indent=2, default=str))
    for k, v in results.items():
        print(f"\n=== stage6 {k} ===")
        print(pd.DataFrame(v).to_string(index=False))
    print(f"\nelapsed {elapsed_min()} min")


if __name__ == "__main__":
    main()
