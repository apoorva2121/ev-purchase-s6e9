"""Endgame: LGBM + XGBoost + CatBoost, multi-seed, on the exp05 feature set.

Two corrections over earlier experiments:
  1. Early stopping no longer reads the fold it scores. Rounds are chosen on a
     10% slice carved from the TRAINING portion, then the model is refit on the
     full training portion with that count. Removes the optimism bias.
  2. Blend members are judged by MARGINAL gain over the running ensemble and by
     rank-correlation with it, not by solo score.
"""
import time, json, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr

RAW="C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
OUT="C:/dev/important repo/ml-bench/ev-purchase-s6e9/artifacts"
SEED,NFOLD,NINNER=42,5,5
TE_COLS=["Annual_Income_USD","Daily_Commute_km"]; SMOOTHS=[5,20,100]; BINS=[250,1000,5000]

tr=pd.read_csv(f"{RAW}/train.csv"); te=pd.read_csv(f"{RAW}/test.csv")
og=pd.read_csv(f"{RAW}/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y=(tr.Will_Buy_EV=="Yes").astype(int).values
raw=[c for c in tr.columns if c not in ("id","Will_Buy_EV")]
both=pd.concat([tr[raw],te[raw]],ignore_index=True)
ogv=np.sort(og.Annual_Income_USD.dropna().unique())
def gap(v):
    i=np.clip(np.searchsorted(ogv,v),1,len(ogv)-1)
    return np.minimum(np.abs(v-ogv[i-1]),np.abs(v-ogv[i]))
def build(df):
    X=df[raw].copy()
    for c in TE_COLS: X[f"freq_{c}"]=X[c].map(both[c].value_counts()).values
    X["is_income_floor"]=(X.Annual_Income_USD==30000).astype(int)
    X["is_commute_floor"]=(X.Daily_Commute_km==5.0).astype(int)
    X["dist_to_orig_income"]=gap(X.Annual_Income_USD.values)
    return X
def keys(df):
    k=[(c,df[c].values) for c in TE_COLS]
    return k+[(f"incbin{b}",(df.Annual_Income_USD.values//b).astype(np.int64)) for b in BINS]
def agg(k,l):
    d=pd.DataFrame({"k":k,"y":l}).groupby("k")["y"].agg(["sum","count"]); return d["sum"],d["count"]
def ap(k,s,c,p,m):
    ks=pd.Series(k); return ((ks.map(s).fillna(0)+p*m)/(ks.map(c).fillna(0)+m)).values

X,Xte=build(tr),build(te)
CATCOLS=[c for c in X.columns if X[c].dtype==object]
ORD={"Range_Anxiety_Level":["Low","Medium","High"]}  # true ordinal order, not alphabetical
for c in CATCOLS:
    cs=ORD.get(c) or sorted(set(X[c])|set(Xte[c])); mp={v:i for i,v in enumerate(cs)}
    X[c]=X[c].map(mp).astype(int); Xte[c]=Xte[c].map(mp).astype(int)
kt,kte=keys(tr),keys(te)
for n,_ in kt:
    for m in SMOOTHS: X[f"te{m}_{n}"]=np.nan; Xte[f"te{m}_{n}"]=0.0
print(f"feature matrix {X.shape}", flush=True)

LGBP=dict(objective="binary",metric="auc",learning_rate=0.02,num_leaves=32,max_depth=5,
          feature_fraction=0.30,bagging_fraction=0.8,bagging_freq=1,min_data_in_leaf=10,
          lambda_l1=0.071,lambda_l2=2.03,max_bin=1024,verbose=-1,num_threads=-1)
XGBP=dict(objective="binary:logistic",eval_metric="auc",tree_method="hist",
          learning_rate=0.02,max_depth=6,colsample_bytree=0.30,subsample=0.8,
          min_child_weight=10,reg_alpha=0.07,reg_lambda=2.0,max_bin=512,nthread=-1)

def fit_fold(Xtr,ytr,Xva,Xt,kind,seed):
    """ES on a slice of TRAIN only, then refit on full train portion."""
    Xa,Xb,ya,yb=train_test_split(Xtr,ytr,test_size=0.10,random_state=seed,stratify=ytr)
    if kind=="lgb":
        P=dict(LGBP,seed=seed,bagging_seed=seed,feature_fraction_seed=seed)
        m=lgb.train(P,lgb.Dataset(Xa,ya),num_boost_round=20000,valid_sets=[lgb.Dataset(Xb,yb)],
                    callbacks=[lgb.early_stopping(300,verbose=False)])
        n=max(int(m.best_iteration/0.9),100)
        m=lgb.train(P,lgb.Dataset(Xtr,ytr),num_boost_round=n)
        return m.predict(Xva),m.predict(Xt),n
    if kind=="xgb":
        P=dict(XGBP,seed=seed)
        d=xgb.DMatrix(Xa,ya); dv=xgb.DMatrix(Xb,yb)
        m=xgb.train(P,d,20000,evals=[(dv,"v")],early_stopping_rounds=300,verbose_eval=False)
        n=max(int(m.best_iteration/0.9),100)
        m=xgb.train(P,xgb.DMatrix(Xtr,ytr),n)
        return m.predict(xgb.DMatrix(Xva)),m.predict(xgb.DMatrix(Xt)),n
    c=CatBoostClassifier(iterations=20000,learning_rate=0.03,depth=6,l2_leaf_reg=3.0,
                         rsm=0.3,border_count=128,random_seed=seed,eval_metric="AUC",
                         verbose=0,thread_count=-1,early_stopping_rounds=300)
    c.fit(Xa,ya,eval_set=(Xb,yb))
    n=max(int(c.get_best_iteration()/0.9),100)
    c2=CatBoostClassifier(iterations=n,learning_rate=0.03,depth=6,l2_leaf_reg=3.0,rsm=0.3,
                          border_count=128,random_seed=seed,verbose=0,thread_count=-1)
    c2.fit(Xtr,ytr)
    return c2.predict_proba(Xva)[:,1],c2.predict_proba(Xt)[:,1],n

def run(kind,seed):
    oof=np.zeros(len(X)); pred=np.zeros(len(Xte)); t0=time.time(); ns=[]
    for ti,vi in StratifiedKFold(NFOLD,shuffle=True,random_state=SEED).split(X,y):
        Xtr,Xva,Xt=X.iloc[ti].copy(),X.iloc[vi].copy(),Xte.copy(); ytr=y[ti]
        for itr,iva in StratifiedKFold(NINNER,shuffle=True,random_state=SEED+1).split(Xtr,ytr):
            for n,kv in kt:
                s,c=agg(kv[ti][itr],ytr[itr]); p=ytr[itr].mean()
                for m in SMOOTHS: Xtr.iloc[iva,Xtr.columns.get_loc(f"te{m}_{n}")]=ap(kv[ti][iva],s,c,p,m)
        pr=ytr.mean()
        for (n,kv),(_,k2) in zip(kt,kte):
            s,c=agg(kv[ti],ytr)
            for m in SMOOTHS:
                Xva[f"te{m}_{n}"]=ap(kv[vi],s,c,pr,m); Xt[f"te{m}_{n}"]=ap(k2,s,c,pr,m)
        o,t,n=fit_fold(Xtr,ytr,Xva,Xt,kind,seed); oof[vi]=o; pred+=t/NFOLD; ns.append(n)
    a=roc_auc_score(y,oof)
    print(f"  {kind}-s{seed:<4} OOF={a:.5f}  rounds~{int(np.mean(ns))}  ({time.time()-t0:.0f}s)",flush=True)
    np.save(f"{OUT}/oof_{kind}{seed}.npy",oof); np.save(f"{OUT}/pred_{kind}{seed}.npy",pred)
    return a,oof,pred

members={}
for kind,seeds in [("lgb",(42,101,202)),("xgb",(42,101)),("cat",(42,))]:
    for s in seeds:
        try: members[f"{kind}{s}"]=run(kind,s)
        except Exception as e: print(f"  {kind}-{s} FAILED: {e}",flush=True)

# ---- hill-climb on RANKS, judged by MARGINAL gain ----
print("\n=== BLEND (marginal gain, rank space) ===",flush=True)
R={k:rankdata(v[1])/len(y) for k,v in members.items()}
RT={k:rankdata(v[2])/len(members[k][2]) for k,v in members.items()}
cur=np.zeros(len(y)); curt=np.zeros(len(Xte)); chosen=[]; best=0.5
for it in range(12):
    gains={}
    for k in R:
        cand=(cur*len(chosen)+R[k])/(len(chosen)+1)
        gains[k]=roc_auc_score(y,cand)-best
    k=max(gains,key=gains.get)
    if gains[k]<=1e-7: print(f"  stop: best marginal gain {gains[k]:+.7f}"); break
    if chosen:
        rho=spearmanr(cur,R[k]).statistic
        print(f"  + {k:<8} marginal={gains[k]:+.6f}  rho_with_ensemble={rho:.5f}")
    else:
        print(f"  + {k:<8} (seed) solo={roc_auc_score(y,R[k]):.5f}")
    cur=(cur*len(chosen)+R[k])/(len(chosen)+1); curt=(curt*len(chosen)+RT[k])/(len(chosen)+1)
    chosen.append(k); best=roc_auc_score(y,cur)
print(f"\n  BLEND OOF = {best:.5f}   members={chosen}")
# duplicate-member sanity test: adding an existing member must gain ~0
d=(cur*len(chosen)+R[chosen[0]])/(len(chosen)+1)
print(f"  duplicate-member test: {roc_auc_score(y,d)-best:+.7f} (must be ~0)")
np.save(f"{OUT}/oof_blend.npy",cur); np.save(f"{OUT}/pred_blend.npy",curt)
json.dump({k:v[0] for k,v in members.items()}|{"blend":best},open(f"{OUT}/multimodel.json","w"),indent=2)
