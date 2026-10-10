"""Experiment 01: nested per-exact-value target encoding.

Paired against baseline exp00 on the identical frozen split.
Smoothing is set EXPLICITLY (sklearn's smooth="auto" degenerates to m~=1 on a
binary target, which would leave singleton categories near-unsmoothed).
"""
import time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RAW = "C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
OUT = "C:/dev/important repo/ml-bench/ev-purchase-s6e9/artifacts"
SEED, NFOLD, NINNER = 42, 5, 5
TE_COLS = ["Annual_Income_USD", "Daily_Commute_km"]
SMOOTHS = [5, 20, 100]

tr = pd.read_csv(f"{RAW}/train.csv"); te = pd.read_csv(f"{RAW}/test.csv")
og = pd.read_csv(f"{RAW}/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
raw_feats = [c for c in tr.columns if c not in ("id", "Will_Buy_EV")]

def stats_of(keys, labels):
    """sum and count of y per exact key value."""
    d = pd.DataFrame({"k": keys, "y": labels}).groupby("k")["y"].agg(["sum", "count"])
    return d["sum"], d["count"]

def apply_te(keys, s, c, prior, m):
    """m-estimate smoothing: (sum_y + prior*m) / (count + m). Unseen -> prior."""
    ks = pd.Series(keys)
    return ((ks.map(s).fillna(0) + prior * m) / (ks.map(c).fillna(0) + m)).values

# ---- label-free features (transductive on train+test is safe: no labels used) ----
both = pd.concat([tr[raw_feats], te[raw_feats]], ignore_index=True)
def freq_map(col):
    return both[col].value_counts()

X = tr[raw_feats].copy(); Xte = te[raw_feats].copy()
for c in TE_COLS:
    fm = freq_map(c)
    X[f"freq_{c}"] = X[c].map(fm).values
    Xte[f"freq_{c}"] = Xte[c].map(fm).values

X["is_income_floor"]  = (X.Annual_Income_USD == 30000).astype(int)
Xte["is_income_floor"] = (Xte.Annual_Income_USD == 30000).astype(int)
X["is_commute_floor"]  = (X.Daily_Commute_km == 5.0).astype(int)
Xte["is_commute_floor"] = (Xte.Daily_Commute_km == 5.0).astype(int)

# signed 1-D distance to nearest ORIGINAL income value (cheap novelty signal;
# the expensive row-similarity variant measured negative in prior work)
ogv = np.sort(og.Annual_Income_USD.unique())
def nearest_gap(v):
    i = np.clip(np.searchsorted(ogv, v), 1, len(ogv) - 1)
    return np.minimum(np.abs(v - ogv[i - 1]), np.abs(v - ogv[i]))
X["dist_to_orig_income"]  = nearest_gap(X.Annual_Income_USD.values)
Xte["dist_to_orig_income"] = nearest_gap(Xte.Annual_Income_USD.values)

for c in raw_feats:
    if X[c].dtype == object:
        cats = sorted(set(X[c]) | set(Xte[c]))
        X[c] = pd.Categorical(X[c], categories=cats)
        Xte[c] = pd.Categorical(Xte[c], categories=cats)

te_names = [f"te{m}_{c}" for c in TE_COLS for m in SMOOTHS]
for n in te_names:
    X[n] = np.nan; Xte[n] = 0.0

params = dict(objective="binary", metric="auc", learning_rate=0.02,
              num_leaves=32, max_depth=5, feature_fraction=0.30,
              bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=10,
              lambda_l1=0.071, lambda_l2=2.03, max_bin=1024,
              verbose=-1, num_threads=-1, seed=SEED)

def run(labels, tag, early_stop=True):
    oof = np.zeros(len(X)); pred = np.zeros(len(Xte)); t0 = time.time()
    skf = StratifiedKFold(NFOLD, shuffle=True, random_state=SEED)
    for k, (ti, vi) in enumerate(skf.split(X, labels)):
        Xtr = X.iloc[ti].copy(); Xva = X.iloc[vi].copy(); Xt = Xte.copy()
        ytr = labels[ti]; prior = ytr.mean()
        # inner loop -> fold-safe encodings for the OUTER-TRAIN rows
        inner = StratifiedKFold(NINNER, shuffle=True, random_state=SEED + 1)
        for itr, iva in inner.split(Xtr, ytr):
            for c in TE_COLS:
                s, cnt = stats_of(Xtr[c].values[itr], ytr[itr])
                p = ytr[itr].mean()
                for m in SMOOTHS:
                    Xtr.iloc[iva, Xtr.columns.get_loc(f"te{m}_{c}")] = \
                        apply_te(Xtr[c].values[iva], s, cnt, p, m)
        # outer-val and test encoded from the FULL outer-train
        for c in TE_COLS:
            s, cnt = stats_of(Xtr[c].values, ytr)
            for m in SMOOTHS:
                Xva[f"te{m}_{c}"] = apply_te(Xva[c].values, s, cnt, prior, m)
                Xt[f"te{m}_{c}"]  = apply_te(Xt[c].values,  s, cnt, prior, m)
        cb = [lgb.early_stopping(500, verbose=False)] if early_stop else []
        vs = [lgb.Dataset(Xva, labels[vi])] if early_stop else None
        m_ = lgb.train(params, lgb.Dataset(Xtr, ytr),
                       num_boost_round=20000 if early_stop else 400,
                       valid_sets=vs, callbacks=cb)
        bi = m_.best_iteration if early_stop else None
        oof[vi] = m_.predict(Xva, num_iteration=bi)
        pred += m_.predict(Xt, num_iteration=bi) / NFOLD
        if early_stop:
            print(f"  fold {k} auc={roc_auc_score(labels[vi], oof[vi]):.5f} iters={bi}", flush=True)
    print(f"{tag} OOF AUC = {roc_auc_score(labels, oof):.5f}  ({time.time()-t0:.0f}s)", flush=True)
    return oof, pred

oof, pred = run(y, "EXP01 nested-TE")
np.save(f"{OUT}/oof_exp01.npy", oof); np.save(f"{OUT}/pred_exp01.npy", pred)

base = np.load(f"{OUT}/oof_baseline.npy")
print(f"\nbaseline  = {roc_auc_score(y, base):.5f}")
print(f"exp01     = {roc_auc_score(y, oof):.5f}")
print(f"DELTA     = {roc_auc_score(y, oof) - roc_auc_score(y, base):+.5f}")

# paired bootstrap on the full OOF vector
rs = np.random.RandomState(7); d = []
for _ in range(200):
    i = rs.randint(0, len(y), len(y))
    if y[i].min() == y[i].max(): continue
    d.append(roc_auc_score(y[i], oof[i]) - roc_auc_score(y[i], base[i]))
d = np.array(d)
print(f"paired bootstrap delta = {d.mean():+.5f}  sd={d.std():.5f}  "
      f"P(delta>0)={np.mean(d>0):.3f}")

# canary: early stopping OFF (it peeks at the val fold and inflates shuffled AUC)
ys = np.random.RandomState(0).permutation(y)
o2, _ = run(ys, "CANARY(shuffled)", early_stop=False)
a = roc_auc_score(ys, o2)
print(f"CANARY = {a:.5f}  pass band |a-0.5|<0.0019 -> {'PASS' if abs(a-0.5)<0.0019 else 'FAIL'}")
