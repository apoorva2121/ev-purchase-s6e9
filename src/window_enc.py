"""Window encodings (from the public XGBoost notebook) + our nested exact TE.

New idea vs everything we built: instead of encoding an exact value or a fixed
BIN, encode the target rate over a CENTRED WINDOW [v-w, v+w] in integer income
units, via prefix sums. No bin-edge artefacts, every value gets a symmetric
neighbourhood, and several widths give a multi-resolution view.

  WIN_INC = [2,5,10,25,50,200]   income windows (dollars)
  WIN_KM  = [1,3,10]             commute windows (km)
  WIN_GRP = [5,25]               income windows within City_Type x Current_Car_Type
  WIN_PRIOR = 10.0               m-estimate smoothing constant
Fold-safe exactly like our TE: inner StratifiedKFold(5) for the training rows,
full outer-train stats for the validation fold and test.
"""
import sys,time,numpy as np,pandas as pd,lightgbm as lgb
sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
from cv import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

WIN_INC=[2,5,10,25,50,200]; WIN_KM=[1,3,10]; WIN_GRP=[5,25]; WIN_PRIOR=10.0

def window_rates(v_fit,y_fit,v_query,widths,prior,gm):
    lo=int(min(v_fit.min(),v_query.min())); hi=int(max(v_fit.max(),v_query.max())); n=hi-lo+1
    cnt=np.bincount(v_fit-lo,minlength=n).astype(np.float64)
    s=np.bincount(v_fit-lo,weights=y_fit.astype(np.float64),minlength=n)
    ccnt=np.concatenate([[0.0],np.cumsum(cnt)]); cs=np.concatenate([[0.0],np.cumsum(s)])
    q=v_query-lo; out=[]
    for w in widths:
        a=np.clip(q-w,0,n); b=np.clip(q+w+1,0,n)
        out.append((((cs[b]-cs[a])+prior*gm)/((ccnt[b]-ccnt[a])+prior)).astype("float32"))
    return np.stack(out,1)

def group_window_scan(g_a,inc_a,y_a,g_q,inc_q,gm):
    out=np.full((len(inc_q),len(WIN_GRP)),gm,np.float32)
    for g in np.unique(g_q):
        ma,mq=g_a==g,g_q==g
        if ma.sum(): out[mq]=window_rates(inc_a[ma],y_a[ma],inc_q[mq],WIN_GRP,WIN_PRIOR,gm)
    return out

INC_TR=np.rint(tr.Annual_Income_USD.values).astype(np.int64)
INC_TE=np.rint(te.Annual_Income_USD.values).astype(np.int64)
KM_TR=np.rint(tr.Daily_Commute_km.values).astype(np.int64)
KM_TE=np.rint(te.Daily_Commute_km.values).astype(np.int64)
G_TR=(tr.City_Type.astype(str)+"|"+tr.Current_Car_Type.astype(str)).values
G_TE=(te.City_Type.astype(str)+"|"+te.Current_Car_Type.astype(str)).values

WCOLS=[f"win_inc_{w}" for w in WIN_INC]+[f"win_km_{w}" for w in WIN_KM]+[f"win_grp_{w}" for w in WIN_GRP]

def run_window(tag,use_exact_te=True,split_seed=42,seed=42,nfold=NFOLD):
    X,Xte=build(tr),build(te); X,Xte=encode_cats(X,Xte)
    kt,kte=(keys(tr),keys(te)) if use_exact_te else ([],[])
    X,Xte=init_te(X,Xte,kt)
    for c in WCOLS: X[c]=np.nan; Xte[c]=0.0
    oof=np.zeros(len(X)); pred=np.zeros(len(Xte)); t0=time.time()
    for ti,vi in StratifiedKFold(nfold,shuffle=True,random_state=split_seed).split(X,y):
        Xtr,Xva,Xt=X.iloc[ti].copy(),X.iloc[vi].copy(),Xte.copy(); ytr=y[ti]; gm=ytr.mean()
        if kt: Xtr,Xva,Xt=te_fold(Xtr,Xva,Xt,ti,vi,ytr,kt,kte,inner_seed=split_seed+1)
        inc_f,km_f,g_f=INC_TR[ti],KM_TR[ti],G_TR[ti]
        # training rows: cross-fitted inner folds
        Wtr=np.zeros((len(ti),len(WCOLS)),np.float32)
        for a,b in StratifiedKFold(5,shuffle=True,random_state=split_seed+1).split(inc_f,ytr):
            blocks=[window_rates(inc_f[a],ytr[a],inc_f[b],WIN_INC,WIN_PRIOR,ytr[a].mean()),
                    window_rates(km_f[a],ytr[a],km_f[b],WIN_KM,WIN_PRIOR,ytr[a].mean()),
                    group_window_scan(g_f[a],inc_f[a],ytr[a],g_f[b],inc_f[b],ytr[a].mean())]
            Wtr[b]=np.concatenate(blocks,1)
        # val + test: full outer-train stats
        Wva=np.concatenate([window_rates(inc_f,ytr,INC_TR[vi],WIN_INC,WIN_PRIOR,gm),
                            window_rates(km_f,ytr,KM_TR[vi],WIN_KM,WIN_PRIOR,gm),
                            group_window_scan(g_f,inc_f,ytr,G_TR[vi],INC_TR[vi],gm)],1)
        Wte=np.concatenate([window_rates(inc_f,ytr,INC_TE,WIN_INC,WIN_PRIOR,gm),
                            window_rates(km_f,ytr,KM_TE,WIN_KM,WIN_PRIOR,gm),
                            group_window_scan(g_f,inc_f,ytr,G_TE,INC_TE,gm)],1)
        for j,c in enumerate(WCOLS):
            Xtr[c]=Wtr[:,j]; Xva[c]=Wva[:,j]; Xt[c]=Wte[:,j]
        o,t=fit_lgb(BASE,seed)(Xtr,ytr,Xva,Xt); oof[vi]=o; pred+=t/nfold
    a=roc_auc_score(y,oof)
    print(f"{tag:<30} split={split_seed} nfold={nfold} OOF={a:.5f} nfeat={X.shape[1]} ({time.time()-t0:.0f}s)",flush=True)
    np.save(f"{OUT}/oof_{tag}.npy",oof); np.save(f"{OUT}/pred_{tag}.npy",pred); return a

if __name__=="__main__":
    print("reference lgb42 (exact+binned TE, honest ES) OOF=0.94574\n",flush=True)
    R={}
    R["w1_windows_only"]   = run_window("w1_windows_only",use_exact_te=False)
    R["w2_windows_plus_te"]= run_window("w2_windows_plus_te",use_exact_te=True)
    R["w3_wte_10fold"]     = run_window("w3_wte_10fold",use_exact_te=True,nfold=10)
    R["w4_wte_splitB"]     = run_window("w4_wte_splitB",use_exact_te=True,split_seed=7)
    print("\n=== WINDOW ENCODINGS (vs lgb42 0.94574) ===")
    for k,v in sorted(R.items(),key=lambda kv:-kv[1]): print(f"  {k:<22} {v:.5f}  {v-0.94574:+.5f}")
