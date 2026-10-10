"""Optuna on CV. Research says tuning is worth ~+-0.00005 here, so this is a
cheap 2-fold proxy search; the winner is then confirmed on the full 5-fold.
"""
import sys,numpy as np,optuna,lightgbm as lgb,json
sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
from common import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

X,Xte=build(tr),build(te);X,Xte=encode_cats(X,Xte)
kt,kte=keys(tr),keys(te);X,Xte=init_te(X,Xte,kt)
FOLDS=list(StratifiedKFold(NFOLD,shuffle=True,random_state=SEED).split(X,y))

def score(P,nf):
    aucs=[]
    for ti,vi in FOLDS[:nf]:
        Xtr,Xva,Xt=X.iloc[ti].copy(),X.iloc[vi].copy(),Xte.iloc[:1].copy();ytr=y[ti]
        Xtr,Xva,Xt=te_fold(Xtr,Xva,Xt,ti,vi,ytr,kt,kte)
        m=lgb.train(P,lgb.Dataset(Xtr,ytr),num_boost_round=20000,
                    valid_sets=[lgb.Dataset(Xva,y[vi])],
                    callbacks=[lgb.early_stopping(300,verbose=False)])
        aucs.append(roc_auc_score(y[vi],m.predict(Xva,num_iteration=m.best_iteration)))
    return float(np.mean(aucs))

def obj(t):
    P=dict(objective="binary",metric="auc",verbose=-1,num_threads=8,seed=SEED,
        learning_rate=t.suggest_float("lr",0.015,0.04,log=True),
        num_leaves=t.suggest_int("leaves",16,96),
        max_depth=t.suggest_int("depth",4,9),
        feature_fraction=t.suggest_float("ff",0.15,0.6),
        bagging_fraction=t.suggest_float("bf",0.6,1.0),bagging_freq=1,
        min_data_in_leaf=t.suggest_int("mdl",5,200,log=True),
        lambda_l1=t.suggest_float("l1",1e-3,5,log=True),
        lambda_l2=t.suggest_float("l2",1e-2,10,log=True),
        max_bin=t.suggest_categorical("mb",[255,511,1024]))
    return score(P,2)

st=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=0))
st.optimize(obj,n_trials=25,show_progress_bar=False)
print("best 2-fold:",round(st.best_value,5));print("best params:",st.best_params)
json.dump(st.best_params,open(f"{OUT}/optuna_best.json","w"),indent=2)
B=st.best_params
P=dict(objective="binary",metric="auc",verbose=-1,num_threads=8,seed=SEED,
       learning_rate=B["lr"],num_leaves=B["leaves"],max_depth=B["depth"],
       feature_fraction=B["ff"],bagging_fraction=B["bf"],bagging_freq=1,
       min_data_in_leaf=B["mdl"],lambda_l1=B["l1"],lambda_l2=B["l2"],max_bin=B["mb"])
print("full 5-fold with best params:",round(score(P,5),5))
