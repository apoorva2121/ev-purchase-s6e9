"""Stage 6b - income target-encoding granularity, measured rather than assumed.

Apoorva's instruction (2026-09-27): "Test out best method for income column encoding, dont just
assume." Stage 6 only compared exact-value encoding against the ladder. This screens the whole
space on the same Stage 2 folds, with both proxies.

Candidates:
  none          raw income only, no target encoding                      (reference)
  exact         one category per distinct income value (13,214 levels)
  bin50         income // 50                                             (coarse)
  bin500        income // 500                                            (coarser)
  qbin200       200 equal-frequency bins                                 (density-adaptive width)
  ladder        exact + //50 + //500 together                            (Apoorva's pick, to test)
  adaptive      exact where the value is common in-fold, else //500      (count-conditional)
  exact_smooth2 / exact_smooth50   exact value, fixed shrinkage instead of sklearn's "auto"

Leakage discipline: every encoder is fitted inside the fold. The adaptive key's frequency
threshold is computed on the training fold only (from X, never y) inside fit(). sklearn's
TargetEncoder cross-fits within fit_transform; AppendTE overrides fit_transform so that path is
actually taken.

Rerunnable:  python code/06b_income_encoding.py
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
INC = "Annual_Income_USD"
NOISE = 0.00050


class IncomeTE(BaseEstimator, TransformerMixin):
    """Append target-encoded income column(s) under a chosen key scheme.

    `scheme` picks the key(s). `adaptive` learns its frequency table on the training fold only,
    from X alone - no label is involved in forming the key, and the encoding itself is
    cross-fitted by TargetEncoder.
    """

    def __init__(self, scheme="exact", smooth="auto", min_count=200, random_state=0):
        self.scheme = scheme
        self.smooth = smooth
        self.min_count = min_count
        self.random_state = random_state

    def _keys(self, X: pd.DataFrame) -> pd.DataFrame:
        inc = X[INC].astype(float)
        s = self.scheme
        if s == "exact":
            return pd.DataFrame({"inc_exact": inc}, index=X.index)
        if s == "bin50":
            return pd.DataFrame({"inc_50": (inc // 50) * 50}, index=X.index)
        if s == "bin500":
            return pd.DataFrame({"inc_500": (inc // 500) * 500}, index=X.index)
        if s == "qbin200":
            return pd.DataFrame(
                {"inc_q200": np.searchsorted(self.qedges_, inc.values)}, index=X.index)
        if s == "ladder":
            return pd.DataFrame({"inc_exact": inc,
                                 "inc_50": (inc // 50) * 50,
                                 "inc_500": (inc // 500) * 500}, index=X.index)
        if s == "adaptive":
            common = inc.isin(self.common_)
            return pd.DataFrame(
                {"inc_adaptive": np.where(common, inc, (inc // 500) * 500)}, index=X.index)
        raise ValueError(self.scheme)

    def _prepare(self, X, y):
        """Anything the key scheme must learn - from X only, on the training fold only."""
        if self.scheme == "qbin200":
            self.qedges_ = np.quantile(X[INC].astype(float).values,
                                       np.linspace(0, 1, 201)[1:-1])
        if self.scheme == "adaptive":
            vc = X[INC].astype(float).value_counts()
            self.common_ = set(vc[vc >= self.min_count].index)

    def _new(self):
        return TargetEncoder(random_state=self.random_state, target_type="binary",
                             smooth=self.smooth)

    def fit(self, X, y=None):
        self._prepare(X, y)
        K = self._keys(X)
        self.key_names_ = list(K.columns)
        self.te_ = self._new().fit(K, y)
        return self

    def fit_transform(self, X, y=None, **fp):
        self._prepare(X, y)
        K = self._keys(X)
        self.key_names_ = list(K.columns)
        self.te_ = self._new()
        return self._append(X, self.te_.fit_transform(K, y))   # cross-fitted

    def transform(self, X):
        return self._append(X, self.te_.transform(self._keys(X)))

    def _append(self, X, Z):
        X = X.copy()
        Z = np.asarray(Z)
        for i, n in enumerate(self.key_names_):
            X[f"{n}__te"] = Z[:, i]
        return X

    def new_cols(self):
        return {"exact": ["inc_exact__te"], "bin50": ["inc_50__te"], "bin500": ["inc_500__te"],
                "qbin200": ["inc_q200__te"], "adaptive": ["inc_adaptive__te"],
                "ladder": ["inc_exact__te", "inc_50__te", "inc_500__te"]}[self.scheme]


def precise(long, noise=NOISE):
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
                     "sd": round(float(wide[name].std(ddof=1)), 6),
                     "diff_vs_best": round(float(d.mean()), 6),
                     "paired_se": round(se, 6),
                     "keep_paired": bool(name == best or float(d.mean()) >= -se),
                     "beats_noise_floor": bool(abs(float(d.mean())) > noise)})
    return pd.DataFrame(rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)


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

    schemes = {
        "exact": dict(scheme="exact"),
        "bin50": dict(scheme="bin50"),
        "bin500": dict(scheme="bin500"),
        "qbin200": dict(scheme="qbin200"),
        "ladder": dict(scheme="ladder"),
        "adaptive": dict(scheme="adaptive", min_count=200),
        "exact_smooth2": dict(scheme="exact", smooth=2.0),
        "exact_smooth50": dict(scheme="exact", smooth=50.0),
    }

    cache = ART / "stage6b_cache"
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

    gb = {"none": Pipeline([("model", HistGradientBoostingClassifier(
        categorical_features="from_dtype", random_state=0))])}
    for name, kw in schemes.items():
        gb[name] = Pipeline([("te", IncomeTE(**kw)),
                             ("model", HistGradientBoostingClassifier(
                                 categorical_features="from_dtype", random_state=0))])
    res = {"gbdt": precise(run("income_gbdt", gb, Xg)).to_dict("records")}

    lin = {"none": Pipeline([
        ("pre", ColumnTransformer([("n", StandardScaler(), nums),
                                   ("c", OneHotEncoder(handle_unknown="ignore"), CATS)])),
        ("model", LogisticRegression(max_iter=1000))])}
    for name, kw in schemes.items():
        t = IncomeTE(**kw)
        lin[name] = Pipeline([
            ("te", t),
            ("pre", ColumnTransformer([("n", StandardScaler(), nums + t.new_cols()),
                                       ("c", OneHotEncoder(handle_unknown="ignore"), CATS)])),
            ("model", LogisticRegression(max_iter=1000))])
    res["linear"] = precise(run("income_linear", lin, Xl)).to_dict("records")

    (ART / "stage6b_income_encoding.json").write_text(json.dumps(res, indent=2, default=str))
    for k, v in res.items():
        print(f"\n=== income encoding, {k} proxy ===")
        print(pd.DataFrame(v).to_string(index=False))
    print(f"\nelapsed {elapsed_min()} min")


if __name__ == "__main__":
    main()
