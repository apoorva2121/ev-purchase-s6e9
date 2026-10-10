"""Batch 3: every remaining untried item from the tabular checklist.
All paired against exp05 (0.94579) on the frozen seed-42 split.
"""
import time,json,numpy as np,pandas as pd,lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

RAW="C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
OUT="C:/dev/important repo/ml-bench/ev-purchase-s6e9/artifacts"
SEED,NFOLD,NINNER=42,5,5
TE_COLS=["Annual_Income_USD","Daily_Commute_km"];SMOOTHS=[5,20,100];BINS=[250,1000,5000]
ORD={"Range_Anxiety_Level":["Low","Medium","High"]}

tr=pd.read_csv(f"{RAW}/train.csv");te=pd.read_csv(f"{RAW}/test.csv")
og=pd.read_csv(f"{RAW}/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y=(tr.Will_Buy_EV=="Yes").astype(int).values
raw=[c for c in tr.columns if c not in ("id","Will_Buy_EV")]
both=pd.concat([tr[raw],te[raw]],ignore_index=True)
ogv=np.sort(og.Annual_Income_USD.dropna().unique())
CATC=[c for c in raw if tr[c].dtype==object]
# interaction pairs to concatenate then target-encode
PAIRS=[("Subsidy_Available","Range_Anxiety_Level"),
       ("Environmental_Concern_Level","Subsidy_Available"),
       ("Environmental_Concern_Level","Range_Anxiety_Level"),
       ("City_Type","Current_Car_Type"),
       ("Home_Charging_Possible","Range_Anxiety_Level")]

def gap(v):
    i=np.clip(np.searchsorted(ogv,v),1,len(ogv)-1)
    return np.minimum(np.abs(v-ogv[i-1]),np.abs(v-ogv[i]))

KMF=["Annual_Income_USD","Daily_Commute_km","Age","Environmental_Concern_Level",
     "Charging_Stations_Near_Home","Charging_Stations_Near_Work"]
_sc=StandardScaler().fit(both[KMF]); _km=MiniBatchKMeans(24,random_state=0,n_init=5).fit(_sc.transform(both[KMF]))

def build(df,flags):
    X=df[raw].copy()
    for c in TE_COLS: X[f"freq_{c}"]=X[c].map(both[c].value_counts()).values
    X["is_income_floor"]=(X.Annual_Income_USD==30000).astype(int)
    X["is_commute_floor"]=(X.Daily_Commute_km==5.0).astype(int)
    X["dist_to_orig_income"]=gap(X.Annual_Income_USD.values)
    if "kmeans" in flags:
        Z=_sc.transform(df[KMF]); X["km_id"]=_km.predict(Z)
        d=_km.transform(Z); X["km_d0"]=d.min(1); X["km_d1"]=np.sort(d,1)[:,1]
    if "onehot" in flags:
        for c in CATC:
            for v in sorted(both[c].unique())[1:]: X[f"oh_{c}_{v}"]=(df[c].values==v).astype(np.int8)
    if "dropgender" in flags: X=X.drop(columns=["Gender"])
    return X

def keys(df,flags):
    k=[(c,df[c].values) for c in TE_COLS]
    k+=[(f"incbin{b}",(df.Annual_Income_USD.values//b).astype(np.int64)) for b in BINS]
    if "pairte" in flags:
        for a,b in PAIRS:
            k.append((f"px_{a[:6]}_{b[:6]}", (df[a].astype(str)+"|"+df[b].astype(str)).values))
        for b in [1000,5000]:
            k.append((f"inc{b}_x_sub",((df.Annual_Income_USD.values//b).astype(str)+"|"+df.Subsidy_Available.values)))
            k.append((f"inc{b}_x_anx",((df.Annual_Income_USD.values//b).astype(str)+"|"+df.Range_Anxiety_Level.values)))
    return k

def agg(k,l):
    d=pd.DataFrame({"k":k,"y":l}).groupby("k")["y"].agg(["sum","count"]);return d["sum"],d["count"]
def ap(k,s,c,p,m):
    ks=pd.Series(k);return ((ks.map(s).fillna(0)+p*m)/(ks.map(c).fillna(0)+m)).values

BASE=dict(objective="binary",metric="auc",learning_rate=0.02,num_leaves=32,max_depth=5,
          feature_fraction=0.30,bagging_fraction=0.8,bagging_freq=1,min_data_in_leaf=10,
          lambda_l1=0.071,lambda_l2=2.03,max_bin=1024,verbose=-1,num_threads=8,seed=SEED)

def run(flags,tag,params=None,onehot_cats=False):
    P=dict(BASE);P.update(params or {})
    X,Xte=build(tr,flags),build(te,flags)
    for c in X.columns:
        if X[c].dtype==object:
            if onehot_cats: X=X.drop(columns=[c]);Xte=Xte.drop(columns=[c]);continue
            cs=ORD.get(c) or sorted(set(X[c])|set(Xte[c]))
            X[c]=pd.Categorical(X[c],categories=cs);Xte[c]=pd.Categorical(Xte[c],categories=cs)
    kt,kte=keys(tr,flags),keys(te,flags)
    for n,_ in kt:
        for m in SMOOTHS: X[f"te{m}_{n}"]=np.nan;Xte[f"te{m}_{n}"]=0.0
    if "icons" in flags:   # interaction constraints: semantics vs artifacts
        cols=list(X.columns)
        g1=[i for i,c in enumerate(cols) if c in raw]
        g2=[i for i,c in enumerate(cols) if c not in raw]
        P["interaction_constraints"]=[g1,g2]
    oof=np.zeros(len(X));pred=np.zeros(len(Xte));t0=time.time()
    for ti,vi in StratifiedKFold(NFOLD,shuffle=True,random_state=SEED).split(X,y):
        Xtr,Xva,Xt=X.iloc[ti].copy(),X.iloc[vi].copy(),Xte.copy();ytr=y[ti]
        for itr,iva in StratifiedKFold(NINNER,shuffle=True,random_state=SEED+1).split(Xtr,ytr):
            for n,kv in kt:
                s,c=agg(kv[ti][itr],ytr[itr]);p=ytr[itr].mean()
                for m in SMOOTHS: Xtr.iloc[iva,Xtr.columns.get_loc(f"te{m}_{n}")]=ap(kv[ti][iva],s,c,p,m)
        pr=ytr.mean()
        for (n,kv),(_,k2) in zip(kt,kte):
            s,c=agg(kv[ti],ytr)
            for m in SMOOTHS:
                Xva[f"te{m}_{n}"]=ap(kv[vi],s,c,pr,m);Xt[f"te{m}_{n}"]=ap(k2,s,c,pr,m)
        mdl=lgb.train(P,lgb.Dataset(Xtr,ytr),num_boost_round=20000,
                      valid_sets=[lgb.Dataset(Xva,y[vi])],
                      callbacks=[lgb.early_stopping(500,verbose=False)])
        oof[vi]=mdl.predict(Xva,num_iteration=mdl.best_iteration)
        pred+=mdl.predict(Xt,num_iteration=mdl.best_iteration)/NFOLD
    a=roc_auc_score(y,oof)
    print(f"{tag:<30} OOF={a:.5f} nfeat={X.shape[1]} ({time.time()-t0:.0f}s)",flush=True)
    np.save(f"{OUT}/oof_{tag}.npy",oof);np.save(f"{OUT}/pred_{tag}.npy",pred)
    return a

ref=np.load(f"{OUT}/oof_exp05_binned_plus_exact.npy");aref=roc_auc_score(y,ref)
print(f"{'reference exp05':<30} OOF={aref:.5f}\n",flush=True)
res={}
TESTS=[
 ({"pairte":1},        "exp15_pair_interact_te", None, False),
 ({},                  "exp16_interaction_cons", None, False),
 ({},                  "exp17_dart",             dict(boosting="dart",drop_rate=0.05), False),
 ({},                  "exp18_linear_tree",      dict(linear_tree=True,linear_lambda=1.0), False),
 ({},                  "exp19_goss",             dict(boosting="goss"), False),
 ({},                  "exp20_catparams",        dict(cat_smooth=50,min_data_per_group=200,max_cat_threshold=64), False),
 ({"kmeans":1},        "exp21_kmeans",           None, False),
 ({"onehot":1},        "exp22_onehot",           None, True),
 ({"dropgender":1},    "exp23_drop_gender",      None, False),
]
TESTS[1][0]["icons"]=1
for flags,tag,pp,oh in TESTS:
    try: res[tag]=run(flags,tag,pp,oh)
    except Exception as e: print(f"{tag} FAILED: {type(e).__name__}: {e}",flush=True)
print("\n=== BATCH3 (paired vs exp05) ===")
for t,a in sorted(res.items(),key=lambda kv:-kv[1]): print(f"  {t:<28} {a:.5f} delta={a-aref:+.5f}")
json.dump(res,open(f"{OUT}/batch3.json","w"),indent=2)
