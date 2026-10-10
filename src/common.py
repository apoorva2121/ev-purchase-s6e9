"""Shared feature pipeline. Backward-compatible with batch4/mlp; extended with
split-seed, smoothing-set and hierarchical-prior options (review B1/B2).
hier = list of (child_key, parent_key): child TE shrunk toward the parent TE of
the same row instead of the global mean, plus the residual as its own column."""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
RAW="C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
OUT="C:/dev/important repo/ml-bench/ev-purchase-s6e9/artifacts"
SEED,NFOLD,NINNER=42,5,5
TE_COLS=["Annual_Income_USD","Daily_Commute_km"];SMOOTHS=[5,20,100];BINS=[250,1000,5000]
ORD={"Range_Anxiety_Level":["Low","Medium","High"]}
tr=pd.read_csv(f"{RAW}/train.csv");te=pd.read_csv(f"{RAW}/test.csv")
og=pd.read_csv(f"{RAW}/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y=(tr.Will_Buy_EV=="Yes").astype(int).values
RAWF=[c for c in tr.columns if c not in ("id","Will_Buy_EV")]
both=pd.concat([tr[RAWF],te[RAWF]],ignore_index=True)
ogv=np.sort(og.Annual_Income_USD.dropna().unique())
def gap(v):
    i=np.clip(np.searchsorted(ogv,v),1,len(ogv)-1); return np.minimum(np.abs(v-ogv[i-1]),np.abs(v-ogv[i]))
def build(df):
    X=df[RAWF].copy()
    for c in TE_COLS: X[f"freq_{c}"]=X[c].map(both[c].value_counts()).values
    X["is_income_floor"]=(X.Annual_Income_USD==30000).astype(int)
    X["is_commute_floor"]=(X.Daily_Commute_km==5.0).astype(int)
    X["dist_to_orig_income"]=gap(X.Annual_Income_USD.values)
    return X
def keys(df):
    k=[(c,df[c].values) for c in TE_COLS]
    return k+[(f"incbin{b}",(df.Annual_Income_USD.values//b).astype(np.int64)) for b in BINS]
def keys_exact(df): return [(c,df[c].values) for c in TE_COLS]
def agg(k,l):
    d=pd.DataFrame({"k":k,"y":l}).groupby("k")["y"].agg(["sum","count"]);return d["sum"],d["count"]
def ap(k,s,c,p,m):
    ks=pd.Series(k);return (ks.map(s).fillna(0).values+p*m)/(ks.map(c).fillna(0).values+m)
def encode_cats(X,Xte,numeric=False,mode=None):
    mode=mode or ("numeric" if numeric else "native")
    if mode=="string": return X,Xte
    for c in list(X.columns):
        if X[c].dtype==object:
            cs=ORD.get(c) or sorted(set(X[c])|set(Xte[c]))
            if mode=="numeric":
                mp={v:i for i,v in enumerate(cs)};X[c]=X[c].map(mp).astype(int);Xte[c]=Xte[c].map(mp).astype(int)
            else:
                X[c]=pd.Categorical(X[c],categories=cs);Xte[c]=pd.Categorical(Xte[c],categories=cs)
    return X,Xte
def init_te(X,Xte,kt,smooths=SMOOTHS,hier=None):
    for n,_ in kt:
        for m in smooths: X[f"te{m}_{n}"]=np.nan;Xte[f"te{m}_{n}"]=0.0
    for child,parent in (hier or []):
        for m in smooths:
            for pre in ("hier","hres"): X[f"{pre}{m}_{child}"]=np.nan;Xte[f"{pre}{m}_{child}"]=0.0
    return X,Xte
def te_fold(Xtr,Xva,Xt,ti,vi,ytr,kt,kte,inner_seed=None,smooths=SMOOTHS,hier=None):
    inner_seed=SEED+1 if inner_seed is None else inner_seed
    col=Xtr.columns.get_loc; kd=dict(kt); kd2=dict(kte); hier=hier or []
    for itr,iva in StratifiedKFold(NINNER,shuffle=True,random_state=inner_seed).split(Xtr,ytr):
        p=ytr[itr].mean(); st={}
        for n,kv in kt:
            st[n]=agg(kv[ti][itr],ytr[itr])
            for m in smooths: Xtr.iloc[iva,col(f"te{m}_{n}")]=ap(kv[ti][iva],*st[n],p,m)
        for child,parent in hier:
            for m in smooths:
                prior=Xtr.iloc[iva,col(f"te{m}_{parent}")].values
                v=ap(kd[child][ti][iva],*st[child],prior,m)
                Xtr.iloc[iva,col(f"hier{m}_{child}")]=v; Xtr.iloc[iva,col(f"hres{m}_{child}")]=v-prior
    pr=ytr.mean(); st={}
    for n,kv in kt:
        st[n]=agg(kv[ti],ytr)
        for m in smooths: Xva[f"te{m}_{n}"]=ap(kv[vi],*st[n],pr,m);Xt[f"te{m}_{n}"]=ap(kd2[n],*st[n],pr,m)
    for child,parent in hier:
        for m in smooths:
            pv=Xva[f"te{m}_{parent}"].values; v=ap(kd[child][vi],*st[child],pv,m)
            Xva[f"hier{m}_{child}"]=v; Xva[f"hres{m}_{child}"]=v-pv
            pt=Xt[f"te{m}_{parent}"].values; v=ap(kd2[child],*st[child],pt,m)
            Xt[f"hier{m}_{child}"]=v; Xt[f"hres{m}_{child}"]=v-pt
    return Xtr,Xva,Xt
