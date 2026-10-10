"""WOE as a DIFFERENT SHRINKAGE ESTIMATOR, not as a transform.

Plain WOE is monotone in target encoding and therefore invisible to a GBDT.
What is not invisible: shrinking in log-odds space with a credibility weight
lambda = n/(n+m), which reorders categories relative to probability-space
m-estimate smoothing (loud low-count categories get pulled harder).

Anscombe continuity correction (+0.5) keeps zero-positive categories finite.
"""
import time, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RAW = "C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
OUT = "C:/dev/important repo/ml-bench/ev-purchase-s6e9/artifacts"
SEED, NFOLD, NINNER = 42, 5, 5
TE_COLS = ["Annual_Income_USD", "Daily_Commute_km"]
SMOOTHS = [5, 20, 100]

tr = pd.read_csv(f"{RAW}/train.csv"); te = pd.read_csv(f"{RAW}/test.csv")
og = pd.read_csv(f"{RAW}/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y = (tr.Will_Buy_EV == "Yes").astype(int).values
raw_feats = [c for c in tr.columns if c not in ("id", "Will_Buy_EV")]
both = pd.concat([tr[raw_feats], te[raw_feats]], ignore_index=True)
og_inc_cnt = og.Annual_Income_USD.value_counts(); og_cmt_cnt = og.Daily_Commute_km.value_counts()
ogv_inc = np.sort(og.Annual_Income_USD.dropna().unique())
ogv_cmt = np.sort(og.Daily_Commute_km.dropna().unique())

def gap(v, ref):
    i = np.clip(np.searchsorted(ref, v), 1, len(ref)-1)
    return np.minimum(np.abs(v-ref[i-1]), np.abs(v-ref[i]))

def build(df):
    X = df[raw_feats].copy()
    for c in TE_COLS:
        X[f"freq_{c}"] = X[c].map(both[c].value_counts()).values
    X["is_income_floor"] = (X.Annual_Income_USD == 30000).astype(int)
    X["is_commute_floor"] = (X.Daily_Commute_km == 5.0).astype(int)
    X["dist_to_orig_income"] = gap(X.Annual_Income_USD.values, ogv_inc)
    oc = X.Annual_Income_USD.map(og_inc_cnt).fillna(0).values
    X["orig_count_income"] = oc
    X["is_novel_income"] = (oc == 0).astype(int)
    X["lift_income"] = X[f"freq_{TE_COLS[0]}"].values / np.where(oc == 0, np.nan, oc)
    occ = X.Daily_Commute_km.map(og_cmt_cnt).fillna(0).values
    X["orig_count_commute"] = occ
    X["lift_commute"] = X[f"freq_{TE_COLS[1]}"].values / np.where(occ == 0, np.nan, occ)
    X["dist_to_orig_commute"] = gap(X.Daily_Commute_km.values, ogv_cmt)
    return X

def agg(k, l):
    d = pd.DataFrame({"k": k, "y": l}).groupby("k")["y"].agg(["sum", "count"])
    return d["sum"], d["count"]

def enc_prob(keys, s, c, prior, m):                      # probability-space m-estimate
    ks = pd.Series(keys)
    return ((ks.map(s).fillna(0) + prior*m) / (ks.map(c).fillna(0) + m)).values

def enc_woe(keys, s, c, Npos, Nneg, m):                  # logit-space credibility WOE
    ks = pd.Series(keys)
    pos = ks.map(s).fillna(0).values
    n   = ks.map(c).fillna(0).values
    neg = n - pos
    raw = np.log((pos + 0.5)/(Npos + 0.5)) - np.log((neg + 0.5)/(Nneg + 0.5))
    return (n/(n + m)) * raw                             # shrink toward 0 = prior odds

params = dict(objective="binary", metric="auc", learning_rate=0.02,
              num_leaves=32, max_depth=5, feature_fraction=0.30,
              bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=10,
              lambda_l1=0.071, lambda_l2=2.03, max_bin=1024,
              verbose=-1, num_threads=-1, seed=SEED)

def run(mode, tag):
    """mode: 'te' | 'woe' | 'both'"""
    X, Xte = build(tr), build(te)
    for c in X.columns:
        if X[c].dtype == object:
            cats = sorted(set(X[c]) | set(Xte[c]))
            X[c] = pd.Categorical(X[c], categories=cats); Xte[c] = pd.Categorical(Xte[c], categories=cats)
    kinds = {"te": ["te"], "woe": ["woe"], "both": ["te", "woe"]}[mode]
    cols = [f"{k}{m}_{c}" for c in TE_COLS for m in SMOOTHS for k in kinds]
    for n in cols: X[n] = np.nan; Xte[n] = 0.0
    oof = np.zeros(len(X)); pred = np.zeros(len(Xte)); t0 = time.time()
    for ti, vi in StratifiedKFold(NFOLD, shuffle=True, random_state=SEED).split(X, y):
        Xtr, Xva, Xt = X.iloc[ti].copy(), X.iloc[vi].copy(), Xte.copy()
        ytr = y[ti]
        for itr, iva in StratifiedKFold(NINNER, shuffle=True, random_state=SEED+1).split(Xtr, ytr):
            yi = ytr[itr]; p = yi.mean(); P, N = yi.sum(), len(yi)-yi.sum()
            for c in TE_COLS:
                kv = tr[c].values[ti]
                s, cn = agg(kv[itr], yi)
                for m in SMOOTHS:
                    if "te" in kinds:
                        Xtr.iloc[iva, Xtr.columns.get_loc(f"te{m}_{c}")] = enc_prob(kv[iva], s, cn, p, m)
                    if "woe" in kinds:
                        Xtr.iloc[iva, Xtr.columns.get_loc(f"woe{m}_{c}")] = enc_woe(kv[iva], s, cn, P, N, m)
        p = ytr.mean(); P, N = ytr.sum(), len(ytr)-ytr.sum()
        for c in TE_COLS:
            kv = tr[c].values[ti]; s, cn = agg(kv, ytr)
            for m in SMOOTHS:
                if "te" in kinds:
                    Xva[f"te{m}_{c}"] = enc_prob(tr[c].values[vi], s, cn, p, m)
                    Xt[f"te{m}_{c}"]  = enc_prob(te[c].values,     s, cn, p, m)
                if "woe" in kinds:
                    Xva[f"woe{m}_{c}"] = enc_woe(tr[c].values[vi], s, cn, P, N, m)
                    Xt[f"woe{m}_{c}"]  = enc_woe(te[c].values,     s, cn, P, N, m)
        mdl = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=20000,
                        valid_sets=[lgb.Dataset(Xva, y[vi])],
                        callbacks=[lgb.early_stopping(500, verbose=False)])
        oof[vi] = mdl.predict(Xva, num_iteration=mdl.best_iteration)
        pred += mdl.predict(Xt, num_iteration=mdl.best_iteration)/NFOLD
    a = roc_auc_score(y, oof)
    print(f"{tag:<28} OOF={a:.5f}  nfeat={X.shape[1]}  ({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{OUT}/oof_{tag}.npy", oof); np.save(f"{OUT}/pred_{tag}.npy", pred)
    return a, oof

ref = np.load(f"{OUT}/oof_exp02_stats.npy") if __import__("os").path.exists(f"{OUT}/oof_exp02_stats.npy") \
      else np.load(f"{OUT}/oof_exp01.npy")
aref = roc_auc_score(y, ref)
print(f"{'reference':<28} OOF={aref:.5f}\n", flush=True)

res = {}
for mode, tag in [("woe", "exp07_woe_only"), ("both", "exp08_woe_plus_te")]:
    a, o = run(mode, tag)
    rs = np.random.RandomState(7); d = []
    for _ in range(200):
        i = rs.randint(0, len(y), len(y))
        d.append(roc_auc_score(y[i], o[i]) - roc_auc_score(y[i], ref[i]))
    d = np.array(d)
    print(f"    paired vs ref: {d.mean():+.5f}  sd={d.std():.5f}  P(>0)={np.mean(d>0):.3f}", flush=True)
    res[tag] = a
json.dump(res, open(f"{OUT}/woe_results.json","w"), indent=2)
