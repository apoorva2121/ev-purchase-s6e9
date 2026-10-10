"""Generic CV runner: any build/key/fit combination, any split seed, honest ES
(rounds chosen on a 10% slice of the TRAINING portion, x1.11, then refit)."""
import time,numpy as np,pandas as pd,lightgbm as lgb
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import roc_auc_score
from common import *
BASE=dict(objective="binary",metric="auc",learning_rate=0.02,num_leaves=32,max_depth=5,
          feature_fraction=0.30,bagging_fraction=0.8,bagging_freq=1,min_data_in_leaf=10,
          lambda_l1=0.071,lambda_l2=2.03,max_bin=1024,verbose=-1,num_threads=8)
def fit_lgb(P,seed):
    def f(Xtr,ytr,Xva,Xt):
        Xa,Xb,ya,yb=train_test_split(Xtr,ytr,test_size=0.10,random_state=seed,stratify=ytr)
        Q=dict(P,seed=seed,bagging_seed=seed,feature_fraction_seed=seed)
        m=lgb.train(Q,lgb.Dataset(Xa,ya),20000,valid_sets=[lgb.Dataset(Xb,yb)],callbacks=[lgb.early_stopping(300,verbose=False)])
        n=max(int(m.best_iteration/0.9),100)
        m=lgb.train(Q,lgb.Dataset(Xtr,ytr),n)
        return m.predict(Xva),m.predict(Xt)
    return f
def fit_cat_native(seed,cat_cols,iters=6000,lr=0.05):
    from catboost import CatBoostClassifier
    def f(Xtr,ytr,Xva,Xt):
        Xa,Xb,ya,yb=train_test_split(Xtr,ytr,test_size=0.10,random_state=seed,stratify=ytr)
        kw=dict(learning_rate=lr,depth=6,l2_leaf_reg=3.0,rsm=0.5,border_count=128,random_seed=seed,
                verbose=0,thread_count=8,cat_features=cat_cols)
        c=CatBoostClassifier(iterations=iters,eval_metric="AUC",early_stopping_rounds=200,**kw)
        c.fit(Xa,ya,eval_set=(Xb,yb)); n=max(int(c.get_best_iteration()/0.9),100)
        c=CatBoostClassifier(iterations=n,**kw); c.fit(Xtr,ytr)
        return c.predict_proba(Xva)[:,1],c.predict_proba(Xt)[:,1]
    return f
def run_cv(tag,build_fn,key_fn,fit_fn,split_seed=42,smooths=SMOOTHS,hier=None,cat_mode="native",extra_cat=(),labels=None):
    yy=y if labels is None else labels
    X,Xte=build_fn(tr),build_fn(te); X,Xte=encode_cats(X,Xte,mode=cat_mode)
    for c in extra_cat:
        cs=sorted(set(X[c])|set(Xte[c])); X[c]=pd.Categorical(X[c],categories=cs); Xte[c]=pd.Categorical(Xte[c],categories=cs)
    kt,kte=key_fn(tr),key_fn(te); X,Xte=init_te(X,Xte,kt,smooths,hier=hier)
    oof=np.zeros(len(X));pred=np.zeros(len(Xte));t0=time.time()
    for ti,vi in StratifiedKFold(NFOLD,shuffle=True,random_state=split_seed).split(X,yy):
        Xtr,Xva,Xt=X.iloc[ti].copy(),X.iloc[vi].copy(),Xte.copy();ytr=yy[ti]
        if kt: Xtr,Xva,Xt=te_fold(Xtr,Xva,Xt,ti,vi,ytr,kt,kte,inner_seed=split_seed+1,smooths=smooths,hier=hier)
        o,t=fit_fn(Xtr,ytr,Xva,Xt); oof[vi]=o; pred+=t/NFOLD
    a=roc_auc_score(yy,oof)
    print(f"{tag:<34} split={split_seed:<3} OOF={a:.5f} nfeat={X.shape[1]} ({time.time()-t0:.0f}s)",flush=True)
    np.save(f"{OUT}/oof_{tag}.npy",oof);np.save(f"{OUT}/pred_{tag}.npy",pred); return a,oof,pred
