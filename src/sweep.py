"""Systematic variant sweep on the frozen split. Each variant is a paired
comparison against exp01. Negative results are kept.
"""
import time, sys, json, numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RAW = "C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
OUT = "C:/dev/important repo/ml-bench/ev-purchase-s6e9/artifacts"
SEED, NFOLD, NINNER = 42, 5, 5
TE_COLS = ["Annual_Income_USD", "Daily_Commute_km"]
SMOOTHS = [5, 20, 100]
BINS = [250, 1000, 5000]           # income bin widths for the binned-TE variant

tr = pd.read_csv(f"{RAW}/train.csv"); te = pd.read_csv(f"{RAW}/test.csv")
og = pd.read_csv(f"{RAW}/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y = (tr.Will_Buy_EV == "Yes").astype(int).values
raw_feats = [c for c in tr.columns if c not in ("id", "Will_Buy_EV")]
LOW_SIGNAL = ["Age","Gender","City_Type","Number_of_Cars_Owned","Current_Car_Type",
              "Charging_Stations_Near_Home","Charging_Stations_Near_Work"]

both = pd.concat([tr[raw_feats], te[raw_feats]], ignore_index=True)
og_inc_cnt = og.Annual_Income_USD.value_counts()
og_cmt_cnt = og.Daily_Commute_km.value_counts()
ogv_inc = np.sort(og.Annual_Income_USD.dropna().unique())
ogv_cmt = np.sort(og.Daily_Commute_km.dropna().unique())
PROBIT = dict(inc=1.4095, concern=0.7358, subsidy=2.3410,
              anx_med=-1.0172, anx_high=-2.9972, const=-6.7148)

def gap(v, ref):
    i = np.clip(np.searchsorted(ref, v), 1, len(ref) - 1)
    return np.minimum(np.abs(v - ref[i-1]), np.abs(v - ref[i]))

def build(df, flags):
    X = df[raw_feats].copy()
    for c in TE_COLS:
        X[f"freq_{c}"] = X[c].map(both[c].value_counts()).values
    X["is_income_floor"] = (X.Annual_Income_USD == 30000).astype(int)
    X["is_commute_floor"] = (X.Daily_Commute_km == 5.0).astype(int)
    X["dist_to_orig_income"] = gap(X.Annual_Income_USD.values, ogv_inc)
    if "stats" in flags:
        oc = X.Annual_Income_USD.map(og_inc_cnt).fillna(0).values
        X["orig_count_income"] = oc
        X["is_novel_income"] = (oc == 0).astype(int)
        X["lift_income"] = X[f"freq_{TE_COLS[0]}"].values / np.where(oc == 0, np.nan, oc)
        occ = X.Daily_Commute_km.map(og_cmt_cnt).fillna(0).values
        X["orig_count_commute"] = occ
        X["lift_commute"] = X[f"freq_{TE_COLS[1]}"].values / np.where(occ == 0, np.nan, occ)
        X["dist_to_orig_commute"] = gap(X.Daily_Commute_km.values, ogv_cmt)
    if "probit" in flags:
        s = (PROBIT["inc"] * df.Annual_Income_USD / 1e5
             + PROBIT["concern"] * df.Environmental_Concern_Level
             + PROBIT["subsidy"] * (df.Subsidy_Available == "Yes")
             + PROBIT["anx_med"] * (df.Range_Anxiety_Level == "Medium")
             + PROBIT["anx_high"] * (df.Range_Anxiety_Level == "High")
             + PROBIT["const"])
        X["probit_score"] = s.values
        X["probit_prob"] = norm.cdf(s.values)
    if "drop" in flags:
        X = X.drop(columns=LOW_SIGNAL)
    return X

def te_keys(df, flags):
    """(name, key-series) pairs to target-encode."""
    ks = []
    if "binned_only" not in flags:
        for c in TE_COLS:
            ks.append((c, df[c].values))
    if "binned" in flags or "binned_only" in flags:
        for b in BINS:
            ks.append((f"incbin{b}", (df.Annual_Income_USD.values // b).astype(np.int64)))
    return ks

def stats_of(k, l):
    d = pd.DataFrame({"k": k, "y": l}).groupby("k")["y"].agg(["sum", "count"])
    return d["sum"], d["count"]

def apply_te(k, s, c, prior, m):
    ks = pd.Series(k)
    return ((ks.map(s).fillna(0) + prior*m) / (ks.map(c).fillna(0) + m)).values

params = dict(objective="binary", metric="auc", learning_rate=0.02,
              num_leaves=32, max_depth=5, feature_fraction=0.30,
              bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=10,
              lambda_l1=0.071, lambda_l2=2.03, max_bin=1024,
              verbose=-1, num_threads=-1, seed=SEED)

def run(flags, tag):
    X, Xte = build(tr, flags), build(te, flags)
    for c in X.columns:
        if X[c].dtype == object:
            cats = sorted(set(X[c]) | set(Xte[c]))
            X[c] = pd.Categorical(X[c], categories=cats)
            Xte[c] = pd.Categorical(Xte[c], categories=cats)
    kt, kte = te_keys(tr, flags), te_keys(te, flags)
    for n, _ in kt:
        for m in SMOOTHS:
            X[f"te{m}_{n}"] = np.nan; Xte[f"te{m}_{n}"] = 0.0
    oof = np.zeros(len(X)); pred = np.zeros(len(Xte)); t0 = time.time()
    for ti, vi in StratifiedKFold(NFOLD, shuffle=True, random_state=SEED).split(X, y):
        Xtr, Xva, Xt = X.iloc[ti].copy(), X.iloc[vi].copy(), Xte.copy()
        ytr = y[ti]; prior = ytr.mean()
        for itr, iva in StratifiedKFold(NINNER, shuffle=True, random_state=SEED+1).split(Xtr, ytr):
            for n, kv in kt:
                s, c = stats_of(kv[ti][itr], ytr[itr]); p = ytr[itr].mean()
                for m in SMOOTHS:
                    Xtr.iloc[iva, Xtr.columns.get_loc(f"te{m}_{n}")] = \
                        apply_te(kv[ti][iva], s, c, p, m)
        for (n, kv), (_, kvte) in zip(kt, kte):
            s, c = stats_of(kv[ti], ytr)
            for m in SMOOTHS:
                Xva[f"te{m}_{n}"] = apply_te(kv[vi], s, c, prior, m)
                Xt[f"te{m}_{n}"] = apply_te(kvte, s, c, prior, m)
        mdl = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=20000,
                        valid_sets=[lgb.Dataset(Xva, y[vi])],
                        callbacks=[lgb.early_stopping(500, verbose=False)])
        oof[vi] = mdl.predict(Xva, num_iteration=mdl.best_iteration)
        pred += mdl.predict(Xt, num_iteration=mdl.best_iteration) / NFOLD
    a = roc_auc_score(y, oof)
    print(f"{tag:<34} OOF={a:.5f}  nfeat={X.shape[1]}  ({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{OUT}/oof_{tag}.npy", oof); np.save(f"{OUT}/pred_{tag}.npy", pred)
    return a

base01 = np.load(f"{OUT}/oof_exp01.npy")
a01 = roc_auc_score(y, base01)
print(f"{'exp01 (control)':<34} OOF={a01:.5f}\n", flush=True)

VARIANTS = [
    (["stats"],                     "exp02_stats"),
    (["stats","probit"],            "exp03_probit"),
    (["stats","drop"],              "exp04_droplowsig"),
    (["stats","binned"],            "exp05_binned_plus_exact"),
    (["stats","binned_only"],       "exp06_binned_only"),
]
res = {}
for flags, tag in VARIANTS:
    try:
        res[tag] = run(flags, tag)
    except Exception as e:
        print(f"{tag} FAILED: {e}", flush=True)

print("\n=== SWEEP SUMMARY (paired vs exp01) ===")
for t, a in sorted(res.items(), key=lambda kv: -kv[1]):
    print(f"  {t:<30} {a:.5f}   delta={a-a01:+.5f}")
json.dump(res, open(f"{OUT}/sweep_results.json","w"), indent=2)
