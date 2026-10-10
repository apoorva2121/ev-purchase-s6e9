"""Batch 2: the untried items from the tabular checklist that survive our own
measurements. Each is a paired change against exp05 (best so far, 0.94579).
"""
import time, json, os, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.cluster import MiniBatchKMeans

RAW = "C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
OUT = "C:/dev/important repo/ml-bench/ev-purchase-s6e9/artifacts"
SEED, NFOLD, NINNER = 42, 5, 5
TE_COLS = ["Annual_Income_USD", "Daily_Commute_km"]
SMOOTHS = [5, 20, 100]; BINS = [250, 1000, 5000]

tr = pd.read_csv(f"{RAW}/train.csv"); te = pd.read_csv(f"{RAW}/test.csv")
og = pd.read_csv(f"{RAW}/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y = (tr.Will_Buy_EV=="Yes").astype(int).values
raw = [c for c in tr.columns if c not in ("id","Will_Buy_EV")]
both = pd.concat([tr[raw], te[raw]], ignore_index=True)
ogv = np.sort(og.Annual_Income_USD.dropna().unique())
CATS = ["Gender","City_Type","Current_Car_Type","Home_Charging_Possible",
        "Subsidy_Available","Range_Anxiety_Level"]

def gap(v):
    i = np.clip(np.searchsorted(ogv, v), 1, len(ogv)-1)
    return np.minimum(np.abs(v-ogv[i-1]), np.abs(v-ogv[i]))

# group stats computed on train+test (label-free -> cannot leak)
GRP = {}
for c in CATS + ["Environmental_Concern_Level","Number_of_Cars_Owned"]:
    GRP[c] = both.groupby(c)[["Annual_Income_USD","Daily_Commute_km","Age"]].agg(["mean","std"])

def build(df, flags):
    X = df[raw].copy()
    for c in TE_COLS: X[f"freq_{c}"] = X[c].map(both[c].value_counts()).values
    X["is_income_floor"]=(X.Annual_Income_USD==30000).astype(int)
    X["is_commute_floor"]=(X.Daily_Commute_km==5.0).astype(int)
    X["dist_to_orig_income"]=gap(X.Annual_Income_USD.values)
    if "ratio" in flags:
        X["inc_per_km"]   = X.Annual_Income_USD/np.maximum(X.Daily_Commute_km,1e-9)
        X["inc_per_car"]  = X.Annual_Income_USD/np.maximum(X.Number_of_Cars_Owned,1)
        X["km_per_car"]   = X.Daily_Commute_km/np.maximum(X.Number_of_Cars_Owned,1)
        X["chg_total"]    = X.Charging_Stations_Near_Home + X.Charging_Stations_Near_Work
        X["chg_diff"]     = X.Charging_Stations_Near_Home - X.Charging_Stations_Near_Work
        X["inc_x_concern"]= X.Annual_Income_USD*X.Environmental_Concern_Level
        X["inc_per_age"]  = X.Annual_Income_USD/np.maximum(X.Age,1)
    if "group" in flags:
        for c, g in GRP.items():
            k = df[c].values
            for num in ["Annual_Income_USD","Daily_Commute_km"]:
                m = pd.Series(k).map(g[(num,"mean")]).values
                s = pd.Series(k).map(g[(num,"std")]).values
                X[f"dev_{num}_by_{c}"] = df[num].values - m
                X[f"z_{num}_by_{c}"]   = (df[num].values - m)/np.where(s==0,np.nan,s)
    if "kmeans" in flags:
        X["km_cluster"] = flags["kmeans"].predict(flags["kmscale"](df)).astype(int)
    return X

def keys(df):
    k = [(c, df[c].values) for c in TE_COLS]
    k += [(f"incbin{b}", (df.Annual_Income_USD.values//b).astype(np.int64)) for b in BINS]
    return k

def agg(k,l):
    d = pd.DataFrame({"k":k,"y":l}).groupby("k")["y"].agg(["sum","count"]); return d["sum"],d["count"]
def ap(k,s,c,p,m):
    ks=pd.Series(k); return ((ks.map(s).fillna(0)+p*m)/(ks.map(c).fillna(0)+m)).values

BASE = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=32,
            max_depth=5, feature_fraction=0.30, bagging_fraction=0.8, bagging_freq=1,
            min_data_in_leaf=10, lambda_l1=0.071, lambda_l2=2.03, max_bin=1024,
            verbose=-1, num_threads=-1, seed=SEED)

def run(flags, tag, params=None, label_encode=False, seeds=(SEED,)):
    P = dict(BASE); P.update(params or {})
    X, Xte = build(tr,flags), build(te,flags)
    for c in X.columns:
        if X[c].dtype==object:
            cs = sorted(set(X[c])|set(Xte[c]))
            if label_encode:
                mp = {v:i for i,v in enumerate(cs)}
                X[c]=X[c].map(mp).astype(int); Xte[c]=Xte[c].map(mp).astype(int)
            else:
                X[c]=pd.Categorical(X[c],categories=cs); Xte[c]=pd.Categorical(Xte[c],categories=cs)
    kt,kte = keys(tr), keys(te)
    for n,_ in kt:
        for m in SMOOTHS: X[f"te{m}_{n}"]=np.nan; Xte[f"te{m}_{n}"]=0.0
    oof=np.zeros(len(X)); pred=np.zeros(len(Xte)); t0=time.time()
    for ti,vi in StratifiedKFold(NFOLD,shuffle=True,random_state=SEED).split(X,y):
        Xtr,Xva,Xt = X.iloc[ti].copy(), X.iloc[vi].copy(), Xte.copy(); ytr=y[ti]
        for itr,iva in StratifiedKFold(NINNER,shuffle=True,random_state=SEED+1).split(Xtr,ytr):
            for n,kv in kt:
                s,c=agg(kv[ti][itr],ytr[itr]); p=ytr[itr].mean()
                for m in SMOOTHS:
                    Xtr.iloc[iva,Xtr.columns.get_loc(f"te{m}_{n}")]=ap(kv[ti][iva],s,c,p,m)
        pr=ytr.mean()
        for (n,kv),(_,kv2) in zip(kt,kte):
            s,c=agg(kv[ti],ytr)
            for m in SMOOTHS:
                Xva[f"te{m}_{n}"]=ap(kv[vi],s,c,pr,m); Xt[f"te{m}_{n}"]=ap(kv2,s,c,pr,m)
        fo=np.zeros(len(vi)); ft=np.zeros(len(Xt))
        for sd in seeds:
            Q=dict(P); Q["seed"]=sd; Q["bagging_seed"]=sd; Q["feature_fraction_seed"]=sd
            mdl=lgb.train(Q, lgb.Dataset(Xtr,ytr), num_boost_round=20000,
                          valid_sets=[lgb.Dataset(Xva,y[vi])],
                          callbacks=[lgb.early_stopping(500,verbose=False)])
            fo+=mdl.predict(Xva,num_iteration=mdl.best_iteration)/len(seeds)
            ft+=mdl.predict(Xt, num_iteration=mdl.best_iteration)/len(seeds)
        oof[vi]=fo; pred+=ft/NFOLD
    a=roc_auc_score(y,oof)
    print(f"{tag:<30} OOF={a:.5f}  nfeat={X.shape[1]}  ({time.time()-t0:.0f}s)",flush=True)
    np.save(f"{OUT}/oof_{tag}.npy",oof); np.save(f"{OUT}/pred_{tag}.npy",pred)
    return a

ref=np.load(f"{OUT}/oof_exp05_binned_plus_exact.npy"); aref=roc_auc_score(y,ref)
print(f"{'reference exp05':<30} OOF={aref:.5f}\n",flush=True)

res={}
res["exp09_ratios"]      = run({"ratio":1}, "exp09_ratios")
res["exp10_groupstats"]  = run({"group":1}, "exp10_groupstats")
res["exp11_ratio_group"] = run({"ratio":1,"group":1}, "exp11_ratio_group")
res["exp12_labelenc"]    = run({}, "exp12_labelenc", label_encode=True)
res["exp13_maxbin255"]   = run({}, "exp13_maxbin255", params=dict(max_bin=255))
res["exp14_seedavg5"]    = run({}, "exp14_seedavg5", seeds=(42,101,202,303,404))
print("\n=== BATCH2 (paired vs exp05) ===")
for t,a in sorted(res.items(), key=lambda kv:-kv[1]):
    print(f"  {t:<26} {a:.5f}  delta={a-aref:+.5f}")
json.dump(res, open(f"{OUT}/batch2.json","w"), indent=2)
