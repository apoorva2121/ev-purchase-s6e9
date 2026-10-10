"""Honest reference baseline: raw features only, fold-safe, no target encoding.

Deliberately plain. Every later experiment is measured as a delta against this
number, so it must contain no cleverness and no leakage.
"""
import time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RAW = "C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
OUT = "C:/dev/important repo/ml-bench/ev-purchase-s6e9/artifacts"
SEED, NFOLD = 42, 5

tr = pd.read_csv(f"{RAW}/train.csv"); te = pd.read_csv(f"{RAW}/test.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
feats = [c for c in tr.columns if c not in ("id", "Will_Buy_EV")]
X, Xte = tr[feats].copy(), te[feats].copy()
for c in feats:
    if X[c].dtype == object:
        cats = sorted(set(X[c]) | set(Xte[c]))
        X[c] = pd.Categorical(X[c], categories=cats)
        Xte[c] = pd.Categorical(Xte[c], categories=cats)

params = dict(objective="binary", metric="auc", learning_rate=0.03,
              num_leaves=31, max_depth=5, feature_fraction=0.3,
              bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=100,
              lambda_l2=1.0, verbose=-1, num_threads=-1, seed=SEED)

oof = np.zeros(len(X)); pred = np.zeros(len(Xte)); t0 = time.time()
for k, (ti, vi) in enumerate(StratifiedKFold(NFOLD, shuffle=True, random_state=SEED).split(X, y)):
    m = lgb.train(params, lgb.Dataset(X.iloc[ti], y[ti]), num_boost_round=3000,
                  valid_sets=[lgb.Dataset(X.iloc[vi], y[vi])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[vi] = m.predict(X.iloc[vi], num_iteration=m.best_iteration)
    pred += m.predict(Xte, num_iteration=m.best_iteration) / NFOLD
    print(f"  fold {k} auc={roc_auc_score(y[vi], oof[vi]):.5f} iters={m.best_iteration}", flush=True)

auc = roc_auc_score(y, oof)
print(f"\nBASELINE OOF AUC = {auc:.5f}   ({time.time()-t0:.0f}s)")
np.save(f"{OUT}/oof_baseline.npy", oof); np.save(f"{OUT}/pred_baseline.npy", pred)

# Leak canary: shuffled labels must collapse to ~0.500. If not, the pipeline leaks.
ys = np.random.RandomState(0).permutation(y)
o2 = np.zeros(len(X))
for ti, vi in StratifiedKFold(NFOLD, shuffle=True, random_state=SEED).split(X, ys):
    m = lgb.train(params, lgb.Dataset(X.iloc[ti], ys[ti]), num_boost_round=200)
    o2[vi] = m.predict(X.iloc[vi])
print(f"LEAK CANARY (shuffled y) OOF AUC = {roc_auc_score(ys, o2):.5f}  <- must be ~0.500")
