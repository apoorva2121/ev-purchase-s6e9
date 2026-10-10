"""Step 5: isolate the 10-fold effect from the window-encoding effect.

w3 (windows + exact TE, 10-fold) scored 0.94599, +0.00025 over the 5-fold
reference -- but it changed two things at once. At 10 folds each model trains on
90% of the data instead of 80%, so part of that gain is the fold count, not the
features. These controls separate the two, on both splits.
"""
import sys,time,numpy as np,pandas as pd
sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
from cv import *
from window_enc import run_window
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def run_nfold(tag,key_fn,split_seed=42,seed=42,nfold=10,build_fn=build):
    X,Xte=build_fn(tr),build_fn(te); X,Xte=encode_cats(X,Xte)
    kt,kte=key_fn(tr),key_fn(te); X,Xte=init_te(X,Xte,kt)
    oof=np.zeros(len(X)); pred=np.zeros(len(Xte)); t0=time.time()
    for ti,vi in StratifiedKFold(nfold,shuffle=True,random_state=split_seed).split(X,y):
        Xtr,Xva,Xt=X.iloc[ti].copy(),X.iloc[vi].copy(),Xte.copy(); ytr=y[ti]
        Xtr,Xva,Xt=te_fold(Xtr,Xva,Xt,ti,vi,ytr,kt,kte,inner_seed=split_seed+1)
        o,t=fit_lgb(BASE,seed)(Xtr,ytr,Xva,Xt); oof[vi]=o; pred+=t/nfold
    a=roc_auc_score(y,oof)
    print(f"{tag:<26} nfold={nfold:<3} split={split_seed} OOF={a:.5f} ({time.time()-t0:.0f}s)",flush=True)
    np.save(f"{OUT}/oof_{tag}.npy",oof); np.save(f"{OUT}/pred_{tag}.npy",pred); return a

if __name__=="__main__":
    print("5-fold refs: split A lgb42=0.94574   split B exp05=0.94576",flush=True)
    print("w3 (windows+TE, 10-fold, split A) was 0.94599\n",flush=True)
    R={}
    R["g1_exp05_10fold_A"] = run_nfold("g1_exp05_10fold_A",keys,42)            # control: fold count alone
    R["g2_exp05_10fold_B"] = run_nfold("g2_exp05_10fold_B",keys,7)             # same on split B
    R["g3_win_10fold_B"]   = run_window("g3_win_10fold_B",True,7,42,10)        # windows+TE, 10-fold, split B
    R["g4_exp05_20fold_A"] = run_nfold("g4_exp05_20fold_A",keys,42,nfold=20)   # do more folds keep helping?
    print("\n=== STEP 5 ===")
    for k,v in sorted(R.items(),key=lambda kv:-kv[1]): print(f"  {k:<22} {v:.5f}")
