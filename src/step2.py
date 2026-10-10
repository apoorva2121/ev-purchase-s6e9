"""Step 2: the remaining untried levers.

  d1  digit-power features  -- each decimal digit of each numeric as its own column
                               (from the top-38 notebook; we only ever tested digit COUNT)
  d2  quantisation ladder   -- TE keys at inc10/50/500 (finer than our 250/1000/5000)
  d3  semi-supervised TE    -- fold test rows into the TE counts with SOFT labels from an
                               existing model. Takes each income value from ~50 rows to ~72.
                               Uses no test labels (there are none) so it cannot leak them;
                               the risk is self-confirmation bias, hence the canary.
  d4  d1+d2+d3 combined
"""
import sys,time,numpy as np,pandas as pd,lightgbm as lgb
sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
from cv import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

NUMC=["Annual_Income_USD","Daily_Commute_km","Age","Charging_Stations_Near_Home",
      "Charging_Stations_Near_Work","Environmental_Concern_Level","Number_of_Cars_Owned"]
# decide which digit columns exist ONCE, from train+test, so both frames match
_DIG=[]
for _c in NUMC:
    _v=np.rint(both[_c].values).astype(np.int64)
    for _k in range(0,6):
        if len(np.unique((_v//(10**_k))%10))>1: _DIG.append((_c,_k))
def build_digits(df):
    X=build(df)
    for c,k in _DIG:
        v=np.rint(df[c].values).astype(np.int64)
        X[f"{c}_d{k}"]=((v//(10**k))%10).astype(np.int8)
    return X
def keys_ladder(df):
    inc=np.rint(df.Annual_Income_USD.values).astype(np.int64)
    km=np.rint(df.Daily_Commute_km.values).astype(np.int64)
    return [(c,df[c].values) for c in TE_COLS]+[
        ("lad_inc10",inc//10),("lad_inc50",inc//50),("lad_inc500",inc//500),
        ("lad_inc5000",inc//5000),("lad_km5",km//5)]

# ---- semi-supervised TE: test rows carry soft labels from an existing model ----
SOFT=np.load(f"{OUT}/pred_all12.npy")          # rank-space in [0,1]; rescale to the prior
SOFT=np.clip(SOFT*(y.mean()/SOFT.mean()),0,1)
def agg_semi(k_tr,l_tr,k_te,w_te):
    a=pd.DataFrame({"k":k_tr,"y":l_tr}).groupby("k")["y"].agg(["sum","count"])
    b=pd.DataFrame({"k":k_te,"y":w_te}).groupby("k")["y"].agg(["sum","count"])
    s=a["sum"].add(b["sum"],fill_value=0.0); c=a["count"].add(b["count"],fill_value=0.0)
    return s,c

def run_semi(tag,key_fn,build_fn=build,split_seed=42,seed=42,labels=None):
    yy=y if labels is None else labels
    X,Xte=build_fn(tr),build_fn(te); X,Xte=encode_cats(X,Xte)
    kt,kte=key_fn(tr),key_fn(te); X,Xte=init_te(X,Xte,kt)
    kd2=dict(kte)
    oof=np.zeros(len(X));pred=np.zeros(len(Xte));t0=time.time()
    for ti,vi in StratifiedKFold(NFOLD,shuffle=True,random_state=split_seed).split(X,yy):
        Xtr,Xva,Xt=X.iloc[ti].copy(),X.iloc[vi].copy(),Xte.copy(); ytr=yy[ti]
        col=Xtr.columns.get_loc
        for itr,iva in StratifiedKFold(NINNER,shuffle=True,random_state=split_seed+1).split(Xtr,ytr):
            p=ytr[itr].mean()
            for n,kv in kt:
                s,c=agg_semi(kv[ti][itr],ytr[itr],kd2[n],SOFT)
                for m in SMOOTHS: Xtr.iloc[iva,col(f"te{m}_{n}")]=ap(kv[ti][iva],s,c,p,m)
        pr=ytr.mean()
        for n,kv in kt:
            s,c=agg_semi(kv[ti],ytr,kd2[n],SOFT)
            for m in SMOOTHS:
                Xva[f"te{m}_{n}"]=ap(kv[vi],s,c,pr,m); Xt[f"te{m}_{n}"]=ap(kd2[n],s,c,pr,m)
        o,t=fit_lgb(BASE,seed)(Xtr,ytr,Xva,Xt); oof[vi]=o; pred+=t/NFOLD
    a=roc_auc_score(yy,oof)
    print(f"{tag:<28} OOF={a:.5f} nfeat={X.shape[1]} ({time.time()-t0:.0f}s)",flush=True)
    np.save(f"{OUT}/oof_{tag}.npy",oof);np.save(f"{OUT}/pred_{tag}.npy",pred); return a

if __name__=="__main__":
    print("reference lgb42 OOF=0.94574\n",flush=True)
    R={}
    R["d1_digits"]  = run_cv("d1_digits",build_digits,keys,fit_lgb(BASE,42),42)[0]
    R["d2_ladder"]  = run_cv("d2_ladder",build,keys_ladder,fit_lgb(BASE,42),42)[0]
    R["d3_semite"]  = run_semi("d3_semite",keys)
    R["d4_all"]     = run_semi("d4_all",keys_ladder,build_digits)
    ys=np.random.RandomState(0).permutation(y)
    ca=run_semi("d3_canary",keys,labels=ys)
    print(f"  SEMI-TE CANARY = {ca:.5f} -> {'PASS' if abs(ca-0.5)<0.0019 else 'FAIL'}",flush=True)
    print("\n=== STEP 2 (vs lgb42 0.94574) ===")
    for k,v in sorted(R.items(),key=lambda kv:-kv[1]): print(f"  {k:<14} {v:.5f}  {v-0.94574:+.5f}")
