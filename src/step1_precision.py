"""Step 1: cheap precision levers on the known-good feature set.
All paired on split A (seed 42) against the same honest-ES reference.

  a) test-side TE fitted on 100% of train, vs the current average of 5x80% fits
  b) finer income bins (25/50/100) filling the gap between exact and 250
"""
import sys,time,numpy as np,pandas as pd,lightgbm as lgb
sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
from cv import *
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import roc_auc_score

def keys_fine(df):
    k=[(c,df[c].values) for c in TE_COLS]
    return k+[(f"incbin{b}",(df.Annual_Income_USD.values//b).astype(np.int64)) for b in (25,50,100,250,1000,5000)]

def run_full_test_te(tag,key_fn,split_seed=42,seed=42):
    """Identical to run_cv except the TEST encoding is fit once on ALL train rows."""
    X,Xte=build(tr),build(te); X,Xte=encode_cats(X,Xte)
    kt,kte=key_fn(tr),key_fn(te); X,Xte=init_te(X,Xte,kt)
    # test encoding from 100% of train, computed once
    prior=y.mean()
    for (n,kv),(_,k2) in zip(kt,kte):
        s,c=agg(kv,y)
        for m in SMOOTHS: Xte[f"te{m}_{n}"]=ap(k2,s,c,prior,m)
    oof=np.zeros(len(X));pred=np.zeros(len(Xte));t0=time.time()
    for ti,vi in StratifiedKFold(NFOLD,shuffle=True,random_state=split_seed).split(X,y):
        Xtr,Xva=X.iloc[ti].copy(),X.iloc[vi].copy(); Xt=Xte.copy(); ytr=y[ti]
        dummy=Xte.iloc[:1].copy()
        Xtr,Xva,_=te_fold(Xtr,Xva,dummy,ti,vi,ytr,kt,kte,inner_seed=split_seed+1)
        o,t=fit_lgb(BASE,seed)(Xtr,ytr,Xva,Xt); oof[vi]=o; pred+=t/NFOLD
    a=roc_auc_score(y,oof)
    print(f"{tag:<32} OOF={a:.5f} nfeat={X.shape[1]} ({time.time()-t0:.0f}s)",flush=True)
    np.save(f"{OUT}/oof_{tag}.npy",oof);np.save(f"{OUT}/pred_{tag}.npy",pred); return a

print("reference lgb42 (honest ES, 5x80% test TE) OOF=0.94574\n",flush=True)
R={}
R["s1a_fulltest_te"] = run_full_test_te("s1a_fulltest_te",keys)
R["s1b_finebins"]    = run_cv("s1b_finebins",build,keys_fine,fit_lgb(BASE,42),42)[0]
R["s1c_both"]        = run_full_test_te("s1c_both",keys_fine)
print("\n=== STEP 1 (paired vs lgb42 = 0.94574) ===")
for k,v in sorted(R.items(),key=lambda kv:-kv[1]): print(f"  {k:<22} {v:.5f}  {v-0.94574:+.5f}")
