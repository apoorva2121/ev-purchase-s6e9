"""Post-mortem: why is this run's CV below the prior work's OOF, and what was missed?

IMPORTANT - this is DIAGNOSTIC ONLY, and it USES THE FROZEN TEST ROWS: the 5-fold CV below runs over
all 668,665 rows, test included. The headline 0.945514 was scored once and is not revised here, but
every number this script produces has seen test rows, and the seed-42 test set is compromised for
any future run on this competition. Nothing measured below may revise the reported 0.945514; doing so would turn the
test set into a validation set retroactively. Anything that wins here is an input to a FUTURE
run with a fresh test set, not a new headline.

Everything is measured on the prior work's own protocol so the numbers are comparable:
StratifiedKFold(5, shuffle=True, random_state=42) over the FULL 668,665 training rows.

Candidates:
  A  this run's final pipeline, on the full data            -> isolates the data-volume effect
  B  A + frequency encoding of income and commute            (prior had it; this run never screened it)
  C  A + distance to nearest source-table income             (prior had it; this run never built it)
  D  A + commute TE ladder                                   (this run screened it, tie on gbdt)
  E  smoothing ladder m=5/20/100 on income+commute           (prior's actual scheme)
  F  E + B + C                                               (closest reconstruction of the prior)
  G  F + the granularity ladder as well                      (both ladders at once)

Rerunnable:  python code/11_postmortem.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from elapsed import elapsed_min                                      # noqa: E402
from features import CATS, FloorsToNaN, MultiTE, _com, _inc, clean_raw, ladder_specs  # noqa: E402

ROOT = HERE.parent
REPO = ROOT.parent
ART = ROOT / "artifacts"
SEED = 42
TE_COLS = ["Annual_Income_USD", "Daily_Commute_km"]


class SmoothLadderTE(BaseEstimator, TransformerMixin):
    """The prior work's scheme: the SAME exact-value key encoded at SEVERAL smoothing strengths,
    each as its own feature. This run instead varied GRANULARITY at one smoothing - a different
    axis. The prior code's docstring argues sklearn's smooth="auto" degenerates to m~=1 on a
    binary target, leaving singleton categories nearly unsmoothed; that claim is tested here.
    """

    def __init__(self, cols=None, smooths=(5.0, 20.0, 100.0), random_state=0):
        # Stored EXACTLY as passed: sklearn's clone() refuses an estimator whose __init__
        # modifies its own parameters. Normalisation happens in _cols()/_smooths() instead.
        self.cols = cols
        self.smooths = smooths
        self.random_state = random_state

    def _cols(self):
        return list(self.cols) if self.cols else list(TE_COLS)

    def _smooths(self):
        return tuple(self.smooths)

    def _new(self, m):
        return TargetEncoder(random_state=self.random_state, target_type="binary", smooth=m)

    def fit(self, X, y=None):
        self.encs_ = {m: self._new(m).fit(X[self._cols()].astype(float), y)
                      for m in self._smooths()}
        return self

    def fit_transform(self, X, y=None, **fp):
        self.encs_, out = {}, {}
        for m in self._smooths():
            e = self._new(m)
            Z = e.fit_transform(X[self._cols()].astype(float), y)      # cross-fitted
            self.encs_[m] = e
            out[m] = np.asarray(Z)
        return self._append(X, out)

    def transform(self, X):
        out = {m: np.asarray(e.transform(X[self._cols()].astype(float)))
               for m, e in self.encs_.items()}
        return self._append(X, out)

    def _append(self, X, out):
        X = X.copy()
        for m, Z in out.items():
            for i, c in enumerate(self._cols()):
                X[f"te{int(m)}_{c}"] = Z[:, i]
        return X

    def new_cols(self):
        return [f"te{int(m)}_{c}" for m in self._smooths() for c in self._cols()]


class AddStaticCols(BaseEstimator, TransformerMixin):
    """Append precomputed, label-free columns by row id. Frequency encoding and
    distance-to-source are functions of the feature values only - no label is involved - so
    computing them once outside the fold is not leakage."""

    def __init__(self, table):
        self.table = table

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.join(self.table, how="left")


def build_static(tr_raw, te_raw, src):
    """Frequency encodings (over train+test, label-free) and distance to nearest source income."""
    both = pd.concat([tr_raw[TE_COLS], te_raw[TE_COLS]], ignore_index=True)
    freq = {c: both[c].value_counts() for c in TE_COLS}
    ogv = np.sort(src["Annual_Income_USD"].dropna().unique())

    def nearest_gap(v):
        i = np.clip(np.searchsorted(ogv, v), 1, len(ogv) - 1)
        return np.minimum(np.abs(v - ogv[i - 1]), np.abs(v - ogv[i]))

    def make(df):
        out = pd.DataFrame(index=df.index)
        for c in TE_COLS:
            out[f"freq_{c}"] = df[c].map(freq[c]).astype(float).values
        out["dist_to_orig_income"] = nearest_gap(df["Annual_Income_USD"].values)
        return out

    return make


def xgb_model():
    import xgboost as xgb
    # The Stage 8 winning hyperparameters, held fixed so every candidate differs only by features.
    return xgb.XGBClassifier(
        n_estimators=700, learning_rate=0.04746484403206429, max_depth=4,
        min_child_weight=1.8946637305151008, subsample=0.5836550391393641,
        colsample_bytree=0.5370211535436904, reg_lambda=0.00795133590998436,
        tree_method="hist", eval_metric="auc", n_jobs=8, random_state=0)


def main() -> None:
    tr_raw = pd.read_csv(REPO / "data/raw/train.csv")
    te_raw = pd.read_csv(REPO / "data/raw/test.csv")
    src = pd.read_csv(REPO / "data/raw/EV_Adoption_and_Range_Anxiety_Dataset.csv")

    tr = clean_raw(tr_raw)
    y = (tr_raw["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
    feats = [c for c in tr.columns if c not in ("id", "Will_Buy_EV")]
    X = tr[feats].copy()
    for c in CATS:
        X[c] = X[c].astype(str)

    make_static = build_static(tr_raw, te_raw, src)
    static = make_static(tr[feats])
    freq_only = AddStaticCols(static[[f"freq_{c}" for c in TE_COLS]])
    dist_only = AddStaticCols(static[["dist_to_orig_income"]])
    both_static = AddStaticCols(static)

    LAD = ladder_specs((50, 500))
    LAD_COM = LAD + [("com_exact", _com)]

    # Replicates the Stage 8 winning pipeline structure exactly: frame steps that ADD columns,
    # then FloorsToNaN, then a ColumnTransformer (numerics passthrough, categoricals one-hot with
    # min_frequency=20 - the template's "onehot" default), then the model.
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    NUM = [c for c in feats if c not in CATS]

    def pipe(*steps, extra_num=(), cat_encoding="onehot"):
        enc = (OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=20)
               if cat_encoding == "onehot" else TargetEncoder(random_state=0))
        pre = ColumnTransformer([("num", "passthrough", NUM + list(extra_num)),
                                 ("cat", enc, CATS)])
        return Pipeline([*steps, ("floors", FloorsToNaN()), ("pre", pre),
                         ("model", xgb_model())])

    lad_cols = MultiTE(LAD).new_cols()
    ladcom_cols = MultiTE(LAD_COM).new_cols()
    sm_cols = SmoothLadderTE().new_cols()
    freq_cols = [f"freq_{c}" for c in TE_COLS]
    dist_cols = ["dist_to_orig_income"]

    cands = {
        "A_this_run_final": pipe(("te", MultiTE(LAD)), extra_num=lad_cols),
        "B_plus_frequency": pipe(("te", MultiTE(LAD)), ("s", freq_only),
                                 extra_num=lad_cols + freq_cols),
        "C_plus_dist_to_source": pipe(("te", MultiTE(LAD)), ("s", dist_only),
                                      extra_num=lad_cols + dist_cols),
        "D_plus_commute_te": pipe(("te", MultiTE(LAD_COM)), extra_num=ladcom_cols),
        "E_smoothing_ladder_prior": pipe(("te", SmoothLadderTE()), extra_num=sm_cols),
        "F_prior_reconstruction": pipe(("te", SmoothLadderTE()), ("s", both_static),
                                       extra_num=sm_cols + freq_cols + dist_cols),
        "G_both_ladders_all": pipe(("te", MultiTE(LAD_COM)), ("te2", SmoothLadderTE()),
                                   ("s", both_static),
                                   extra_num=ladcom_cols + sm_cols + freq_cols + dist_cols),
        # H tests the slot that my Stage 8 search silently never sampled: target encoding of
        # the low-cardinality categoricals instead of one-hot.
        "H_cat_target_encoding": pipe(("te", MultiTE(LAD)), extra_num=lad_cols,
                                      cat_encoding="target"),
    }

    cache = ART / "postmortem_cache"
    cache.mkdir(exist_ok=True)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    results = {}
    for name, p in cands.items():
        f = cache / f"{name}.json"
        if f.exists():
            results[name] = json.loads(f.read_text())
            print(f"{name:26s} {results[name]['oof']:.6f} (cached)")
            continue
        t0 = time.time()
        oof = np.zeros(len(X))
        from sklearn.base import clone
        for ti, vi in folds:
            m = clone(p).fit(X.iloc[ti], y[ti])
            oof[vi] = m.predict_proba(X.iloc[vi])[:, 1]
        r = {"oof": float(roc_auc_score(y, oof)),
             "per_fold": [float(roc_auc_score(y[vi], oof[vi])) for _, vi in folds],
             "seconds": round(time.time() - t0, 1)}
        f.write_text(json.dumps(r))
        np.save(cache / f"oof_{name}.npy", oof)
        results[name] = r
        print(f"{name:26s} {r['oof']:.6f}  ({r['seconds']}s)", flush=True)

    base = results["A_this_run_final"]["oof"]
    table = []
    for k, v in results.items():
        table.append({"candidate": k, "oof_full_data": round(v["oof"], 6),
                      "delta_vs_A": round(v["oof"] - base, 6),
                      "beats_noise_0.0005": abs(v["oof"] - base) > 0.0005,
                      "seconds": v["seconds"]})
    df = pd.DataFrame(table).sort_values("oof_full_data", ascending=False)

    out = {"protocol": "StratifiedKFold(5, shuffle=True, random_state=42) on all 668,665 rows",
           "this_run_cv_on_65pct_data": 0.945206,
           "prior_work_oof_exp01": 0.94565,
           "prior_work_best_lb": 0.94615,
           "results": df.to_dict("records")}
    (ART / "postmortem.json").write_text(json.dumps(out, indent=2))
    print()
    print(df.to_string(index=False))
    print(f"\nthis run's Stage 8 CV on 65% of the data: 0.945206")
    print(f"prior work OOF (exp01, full data):          0.945650")
    print(f"elapsed {elapsed_min()} min")


if __name__ == "__main__":
    main()
