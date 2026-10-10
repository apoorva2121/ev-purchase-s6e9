"""Shared, in-fold-fitted transformers for stages 6 to 10.

One module so the Stage 8 search, the Stage 9 evaluation and the Stage 10 production pipeline
all use the identical objects that were screened at Stage 6. Re-implementing them per stage is
how a pipeline silently stops being the thing that was measured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import TargetEncoder

CATS = ["Gender", "City_Type", "Current_Car_Type",
        "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
NUMS = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
        "Environmental_Concern_Level"]
FLOORS = {"Annual_Income_USD": 30000.0, "Daily_Commute_km": 5.0}
NOISE = 0.00050

# Stage 1 stateless cleaning lives HERE, as the single canonical definition, because it is used
# in three places that must not drift apart: the Stage 1 audit, and the Stage 10 production
# pipeline, which pickles it. 01_audit_clean.py imports it from here.
RANGE_ANXIETY_ORDER = ["Low", "Medium", "High"]
RANGE_RULES = {
    "Age": (16, 110),
    "Annual_Income_USD": (0, 10_000_000),
    "Daily_Commute_km": (0, 1_000),
    "Number_of_Cars_Owned": (0, 20),
    "Charging_Stations_Near_Home": (0, 500),
    "Charging_Stations_Near_Work": (0, 500),
    "Environmental_Concern_Level": (1, 5),
}


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless, row-wise cleaning. Learns nothing from the data: every rule is a constant.

    Reused verbatim as the first step of the Stage 10 production pipeline, so raw production
    rows get exactly the cleaning the training data got.
    """
    df = df.copy()
    for c in CATS:                       # no-op on this data (0 variants); production hardening
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
    for c, (lo, hi) in RANGE_RULES.items():     # both ends; 0 fired on this data
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            df[c] = v.where((v >= lo) & (v <= hi))
    if "Range_Anxiety_Level" in df.columns:
        df["Range_Anxiety_Level"] = pd.Categorical(
            df["Range_Anxiety_Level"], categories=RANGE_ANXIETY_ORDER, ordered=True)
    if "id" in df.columns:
        df["id"] = df["id"].astype("string")
    return df


class RawPrep:
    """Production entry point: stateless cleaning, then the exact columns and dtypes the model
    expects. A class rather than a closure, and it carries `feats` as instance state, so the
    pickled pipeline reloads in a fresh process. Two Stage 10 runs failed to pickle at all, and
    a third pickled but could not be reloaded, before this shape was reached."""

    def __init__(self, feats, cats=None):
        self.feats = list(feats)
        self.cats = list(cats) if cats is not None else list(CATS)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        d = clean_raw(df)
        d = d[self.feats].copy()
        for c in self.cats:
            d[c] = d[c].astype(str)
        return d


class MultiTE(BaseEstimator, TransformerMixin):
    """Append cross-fitted target encodings for an arbitrary set of derived keys.

    `specs` is a list of (name, callable(X) -> Series). Each becomes one encoder key and one
    appended column `<name>__te`.

    fit_transform is overridden on purpose. sklearn's TargetEncoder only cross-fits inside its
    own fit_transform; TransformerMixin's default fit_transform is fit().transform(), which
    would hand every training row the full-data encoding of its own value - a target leak.
    Inside a Pipeline, .fit() calls fit_transform on intermediate steps, so training rows get
    cross-fitted values and scoring rows get the full-fold encoding. That is the correct pairing.
    """

    def __init__(self, specs, smooth="auto", random_state=0):
        self.specs = specs
        self.smooth = smooth
        self.random_state = random_state

    def _keys(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({n: f(X) for n, f in self.specs}, index=X.index)

    def _new(self):
        return TargetEncoder(random_state=self.random_state, target_type="binary",
                             smooth=self.smooth)

    def fit(self, X, y=None):
        self.te_ = self._new().fit(self._keys(X), y)
        return self

    def fit_transform(self, X, y=None, **fp):
        self.te_ = self._new()
        return self._append(X, self.te_.fit_transform(self._keys(X), y))

    def transform(self, X):
        return self._append(X, self.te_.transform(self._keys(X)))

    def _append(self, X, Z):
        X = X.copy()
        Z = np.asarray(Z)
        for i, (n, _) in enumerate(self.specs):
            X[f"{n}__te"] = Z[:, i]
        return X

    def new_cols(self):
        return [f"{n}__te" for n, _ in self.specs]


# ---- key builders ---------------------------------------------------------
# These are module-level callables, NOT closures or lambdas, because the Stage 10 production
# pipeline is pickled with joblib and a closure cannot be serialised. Two Stage 10 runs failed
# on exactly this before it was fixed; a pipeline that cannot be saved is not a handoff.

def _inc(X):
    return X["Annual_Income_USD"].astype(float)


def _com(X):
    return X["Daily_Commute_km"].astype(float)


class FloorDivKey:
    """(value // d) * d, as a picklable callable."""

    def __init__(self, col_fn, d):
        self.col_fn = col_fn
        self.d = d

    def __call__(self, X):
        return (self.col_fn(X) // self.d) * self.d

    def __repr__(self):
        return f"FloorDivKey({getattr(self.col_fn, '__name__', self.col_fn)}, {self.d})"


class ComboKey:
    """An exact numeric value crossed with a categorical or ordinal column. Picklable."""

    def __init__(self, col_fn, other):
        self.col_fn = col_fn
        self.other = other

    def __call__(self, X):
        return self.col_fn(X).astype(str) + "|" + X[self.other].astype(str)

    def __repr__(self):
        return f"ComboKey({getattr(self.col_fn, '__name__', self.col_fn)}, {self.other})"


def floor_div(col_fn, d):
    return FloorDivKey(col_fn, d)


def combo(col_fn, other):
    return ComboKey(col_fn, other)


def ladder_specs(divisors=(50, 500), with_exact=True, col="income"):
    fn = _inc if col == "income" else _com
    tag = "inc" if col == "income" else "com"
    specs = [(f"{tag}_exact", fn)] if with_exact else []
    specs += [(f"{tag}_{d}", FloorDivKey(fn, d)) for d in divisors]
    return specs


class FloorIndicator(BaseEstimator, TransformerMixin):
    """Keep the censored floor value AND append a binary is-at-floor column per feature."""

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

    def new_cols(self):
        return [f"{c}__at_floor" for c in self.floors]


class FloorsToNaN(BaseEstimator, TransformerMixin):
    """Replace the measured censoring floors with NaN (for native-NaN model families)."""

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


def precise(long: pd.DataFrame, noise: float = NOISE) -> pd.DataFrame:
    """Unrounded paired statistics, same maths as screen.summarize with more decimals.

    `beats_noise_floor` uses the Stage 2 bootstrap sampling noise (0.00050), which answers
    "would this replicate on fresh data". `keep_paired` uses the Nadeau-Bengio paired SE,
    which answers "is this consistent across these folds". Both are reported because on
    434k rows they routinely disagree.
    """
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
