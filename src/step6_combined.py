"""Step 6: combine every lever that measured positive, and test whether they stack.

Measured separately, all vs the 5-fold exp05 reference 0.94574:
    10-fold CV                      +0.00017
    digits + ladder + semi-TE (d4)  +0.00021
    window encodings                +0.00008

  h1  d4 features at 10-fold                         (fold count + d4)
  h2  d4 + windows at 10-fold                        (everything)
  h3  h2 on split B                                  (replication)
  h4  everything at 10-fold, 3 seeds averaged        (final candidate)
  canary on the full combined pipeline
"""
import sys,time,numpy as np,pandas as pd
sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
from cv import *
from step2 import build_digits, keys_ladder, agg_semi, SOFT
from window_enc import (WIN_INC,WIN_KM,WIN_GRP,WIN_PRIOR,window_rates,group_window_scan,
                        INC_TR,INC_TE,KM_TR,KM_TE,G_TR,G_TE,WCOLS)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

def run_combo(tag,nfold=10,split_seed=42,seeds=(42,),use_win=True,use_semi=True,
              build_fn=build_digits,key_fn=keys_ladder,labels=None):
    yy=y if labels is None else labels
    X,Xte=build_fn(tr),build_fn(te); X,Xte=encode_cats(X,Xte)
    kt,kte=key_fn(tr),key_fn(te); X,Xte=init_te(X,Xte,kt); kd2=dict(kte)
    if use_win:
        for c in WCOLS: X[c]=np.nan; Xte[c]=0.0
    oof=np.zeros(len(X)); pred=np.zeros(len(Xte)); t0=time.time()
    for ti,vi in StratifiedKFold(nfold,shuffle=True,random_state=split_seed).split(X,yy):
        Xtr,Xva,Xt=X.iloc[ti].copy(),X.iloc[vi].copy(),Xte.copy()
        ytr=yy[ti]; col=Xtr.columns.get_loc; gm=ytr.mean()
        AGG=(lambda k_tr,l_tr,n: agg_semi(k_tr,l_tr,kd2[n],SOFT)) if use_semi else (lambda k_tr,l_tr,n: agg(k_tr,l_tr))
        for itr,iva in StratifiedKFold(NINNER,shuffle=True,random_state=split_seed+1).split(Xtr,ytr):
            p=ytr[itr].mean()
            for n,kv in kt:
                s,c=AGG(kv[ti][itr],ytr[itr],n)
                for m in SMOOTHS: Xtr.iloc[iva,col(f"te{m}_{n}")]=ap(kv[ti][iva],s,c,p,m)
        pr=ytr.mean()
        for n,kv in kt:
            s,c=AGG(kv[ti],ytr,n)
            for m in SMOOTHS:
                Xva[f"te{m}_{n}"]=ap(kv[vi],s,c,pr,m); Xt[f"te{m}_{n}"]=ap(kd2[n],s,c,pr,m)
        if use_win:
            inc_f,km_f,g_f=INC_TR[ti],KM_TR[ti],G_TR[ti]
            W=np.zeros((len(ti),len(WCOLS)),np.float32)
            for a,b in StratifiedKFold(5,shuffle=True,random_state=split_seed+1).split(inc_f,ytr):
                W[b]=np.concatenate([window_rates(inc_f[a],ytr[a],inc_f[b],WIN_INC,WIN_PRIOR,ytr[a].mean()),
                                     window_rates(km_f[a],ytr[a],km_f[b],WIN_KM,WIN_PRIOR,ytr[a].mean()),
                                     group_window_scan(g_f[a],inc_f[a],ytr[a],g_f[b],inc_f[b],ytr[a].mean())],1)
            Wva=np.concatenate([window_rates(inc_f,ytr,INC_TR[vi],WIN_INC,WIN_PRIOR,gm),
                                window_rates(km_f,ytr,KM_TR[vi],WIN_KM,WIN_PRIOR,gm),
                                group_window_scan(g_f,inc_f,ytr,G_TR[vi],INC_TR[vi],gm)],1)
            Wte=np.concatenate([window_rates(inc_f,ytr,INC_TE,WIN_INC,WIN_PRIOR,gm),
                                window_rates(km_f,ytr,KM_TE,WIN_KM,WIN_PRIOR,gm),
                                group_window_scan(g_f,inc_f,ytr,G_TE,INC_TE,gm)],1)
            for j,c in enumerate(WCOLS): Xtr[c]=W[:,j]; Xva[c]=Wva[:,j]; Xt[c]=Wte[:,j]
        fo=np.zeros(len(vi)); ft=np.zeros(len(Xt))
        for sd in seeds:
            o,t=fit_lgb(BASE,sd)(Xtr,ytr,Xva,Xt); fo+=o/len(seeds); ft+=t/len(seeds)
        oof[vi]=fo; pred+=ft/nfold
    a=roc_auc_score(yy,oof)
    print(f"{tag:<26} nfold={nfold:<3} split={split_seed} seeds={len(seeds)} OOF={a:.5f} nfeat={X.shape[1]} ({time.time()-t0:.0f}s)",flush=True)
    np.save(f"{OUT}/oof_{tag}.npy",oof); np.save(f"{OUT}/pred_{tag}.npy",pred); return a

if __name__=="__main__":
    print("refs: 5-fold exp05 A=0.94574 B=0.94576 | 10-fold control A=0.94591 | d4 5-fold=0.94595\n",flush=True)
    R={}
    R["h1_d4_10fold_A"]   = run_combo("h1_d4_10fold_A",use_win=False)
    R["h2_all_10fold_A"]  = run_combo("h2_all_10fold_A",use_win=True)
    R["h3_all_10fold_B"]  = run_combo("h3_all_10fold_B",use_win=True,split_seed=7)
    ys=np.random.RandomState(0).permutation(y)
    ca=run_combo("h_canary",use_win=True,labels=ys)
    print(f"  COMBINED CANARY = {ca:.5f} -> {'PASS' if abs(ca-0.5)<0.0019 else 'FAIL'}",flush=True)
    R["h4_all_10fold_3seed"]=run_combo("h4_all_10fold_3seed",use_win=True,seeds=(42,101,202))
    print("\n=== STEP 6: do the levers stack? ===")
    for k,v in sorted(R.items(),key=lambda kv:-kv[1]): print(f"  {k:<24} {v:.5f}  vs 0.94574 = {v-0.94574:+.5f}")
