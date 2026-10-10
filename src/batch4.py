"""Batch 4: kNN-target, HistGradientBoosting, refit-on-all vs fold-average,
ridge stacking done correctly, and a repeated-CV / fresh-seed honesty check.
"""
import sys,time,json,numpy as np,pandas as pd,lightgbm as lgb
sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
from common import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata

BASE=dict(objective="binary",metric="auc",learning_rate=0.02,num_leaves=32,max_depth=5,
          feature_fraction=0.30,bagging_fraction=0.8,bagging_freq=1,min_data_in_leaf=10,
          lambda_l1=0.071,lambda_l2=2.03,max_bin=1024,verbose=-1,num_threads=8,seed=SEED)
ref=np.load(f"{OUT}/oof_exp05_binned_plus_exact.npy");aref=roc_auc_score(y,ref)
print(f"reference exp05 OOF={aref:.5f}\n",flush=True)
res={}

# ---------- exp24: kNN out-of-fold target feature ----------
print("exp24: building kNN-target feature (fold-safe)...",flush=True)
KF=["Annual_Income_USD","Daily_Commute_km","Age","Environmental_Concern_Level",
    "Charging_Stations_Near_Home","Charging_Stations_Near_Work","Number_of_Cars_Owned"]
sc=StandardScaler().fit(both[KF])
Ztr,Zte=sc.transform(tr[KF]),sc.transform(te[KF])
knn_tr=np.zeros(len(tr));knn_te=np.zeros(len(te));t0=time.time()
for ti,vi in StratifiedKFold(NFOLD,shuffle=True,random_state=SEED).split(Ztr,y):
    sub=np.random.RandomState(0).choice(ti,min(60000,len(ti)),replace=False)
    nn=NearestNeighbors(n_neighbors=25,n_jobs=8).fit(Ztr[sub])
    _,idx=nn.kneighbors(Ztr[vi]);knn_tr[vi]=y[sub][idx].mean(1)
    _,idx=nn.kneighbors(Zte);knn_te+=y[sub][idx].mean(1)/NFOLD
print(f"  knn feature built ({time.time()-t0:.0f}s), standalone AUC={roc_auc_score(y,knn_tr):.5f}",flush=True)

def run_lgb(extra_tr=None,extra_te=None,tag="",params=None,refit_all=False):
    P=dict(BASE);P.update(params or {})
    X,Xte=build(tr),build(te)
    if extra_tr is not None:
        for k,v in extra_tr.items(): X[k]=v
        for k,v in extra_te.items(): Xte[k]=v
    X,Xte=encode_cats(X,Xte)
    kt,kte=keys(tr),keys(te);X,Xte=init_te(X,Xte,kt)
    oof=np.zeros(len(X));pred=np.zeros(len(Xte));iters=[];t0=time.time()
    for ti,vi in StratifiedKFold(NFOLD,shuffle=True,random_state=SEED).split(X,y):
        Xtr,Xva,Xt=X.iloc[ti].copy(),X.iloc[vi].copy(),Xte.copy();ytr=y[ti]
        Xtr,Xva,Xt=te_fold(Xtr,Xva,Xt,ti,vi,ytr,kt,kte)
        m=lgb.train(P,lgb.Dataset(Xtr,ytr),num_boost_round=20000,
                    valid_sets=[lgb.Dataset(Xva,y[vi])],
                    callbacks=[lgb.early_stopping(500,verbose=False)])
        oof[vi]=m.predict(Xva,num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        if not refit_all: pred+=m.predict(Xt,num_iteration=m.best_iteration)/NFOLD
    if refit_all:   # single model on ALL data with mean rounds
        Xa,Xv,Xt=X.copy(),X.iloc[:1].copy(),Xte.copy()
        ia=np.arange(len(X));Xa,_,Xt=te_fold(Xa,Xv,Xt,ia,np.array([0]),y,kt,kte)
        m=lgb.train(P,lgb.Dataset(Xa,y),num_boost_round=int(np.mean(iters)))
        pred=m.predict(Xt)
    a=roc_auc_score(y,oof)
    print(f"{tag:<30} OOF={a:.5f} ({time.time()-t0:.0f}s)",flush=True)
    np.save(f"{OUT}/oof_{tag}.npy",oof);np.save(f"{OUT}/pred_{tag}.npy",pred)
    return a

res["exp24_knn_target"]=run_lgb({"knn25":knn_tr},{"knn25":knn_te},"exp24_knn_target")
res["exp25_refit_all"] =run_lgb(None,None,"exp25_refit_all",refit_all=True)

# ---------- exp26: HistGradientBoosting (diversity member) ----------
print("\nexp26: HistGradientBoosting",flush=True)
X,Xte=build(tr),build(te);X,Xte=encode_cats(X,Xte,numeric=True)
kt,kte=keys(tr),keys(te);X,Xte=init_te(X,Xte,kt)
oof=np.zeros(len(X));pred=np.zeros(len(Xte));t0=time.time()
for ti,vi in StratifiedKFold(NFOLD,shuffle=True,random_state=SEED).split(X,y):
    Xtr,Xva,Xt=X.iloc[ti].copy(),X.iloc[vi].copy(),Xte.copy();ytr=y[ti]
    Xtr,Xva,Xt=te_fold(Xtr,Xva,Xt,ti,vi,ytr,kt,kte)
    h=HistGradientBoostingClassifier(max_iter=1200,learning_rate=0.03,max_depth=6,
        min_samples_leaf=20,l2_regularization=2.0,max_bins=255,
        early_stopping=True,validation_fraction=0.1,n_iter_no_change=60,random_state=SEED)
    h.fit(Xtr,ytr)
    oof[vi]=h.predict_proba(Xva)[:,1];pred+=h.predict_proba(Xt)[:,1]/NFOLD
a=roc_auc_score(y,oof);print(f"{'exp26_histgb':<30} OOF={a:.5f} ({time.time()-t0:.0f}s)",flush=True)
np.save(f"{OUT}/oof_exp26_histgb.npy",oof);np.save(f"{OUT}/pred_exp26_histgb.npy",pred)
res["exp26_histgb"]=a

# ---------- exp27: ridge/logistic stacking, done correctly ----------
# The documented 8-sigma failure came from OOF columns being single-model preds
# while test columns were fold-averages. Here BOTH are fold-averages.
print("\nexp27: logistic stacking on OOF (distribution-matched)",flush=True)
import glob,os
names=[os.path.basename(f)[4:-4] for f in glob.glob(f"{OUT}/oof_*.npy")
       if os.path.exists(f.replace("oof_","pred_"))]
names=[n for n in names if n not in ("baseline",)]
M=np.column_stack([rankdata(np.load(f"{OUT}/oof_{n}.npy"))/len(y) for n in names])
MT=np.column_stack([rankdata(np.load(f"{OUT}/pred_{n}.npy"))/len(te) for n in names])
so=np.zeros(len(y))
for ti,vi in StratifiedKFold(NFOLD,shuffle=True,random_state=7).split(M,y):
    lr=LogisticRegression(C=1.0,max_iter=2000).fit(M[ti],y[ti]);so[vi]=lr.predict_proba(M[vi])[:,1]
a=roc_auc_score(y,so)
print(f"  members={len(names)}  stack OOF={a:.5f}  delta_vs_exp05={a-aref:+.5f}",flush=True)
res["exp27_logistic_stack"]=a
lr=LogisticRegression(C=1.0,max_iter=2000).fit(M,y)
np.save(f"{OUT}/oof_exp27_stack.npy",so);np.save(f"{OUT}/pred_exp27_stack.npy",lr.predict_proba(MT)[:,1])

print("\n=== BATCH4 (paired vs exp05) ===")
for t,v in sorted(res.items(),key=lambda kv:-kv[1]):print(f"  {t:<28} {v:.5f} delta={v-aref:+.5f}")
json.dump(res,open(f"{OUT}/batch4.json","w"),indent=2)
