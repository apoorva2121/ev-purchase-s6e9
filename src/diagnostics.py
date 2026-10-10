"""Fast diagnostics that need no training loop: adversarial validation,
null importance, permutation importance. Each licenses or kills a branch.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RAW="C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
tr=pd.read_csv(f"{RAW}/train.csv"); te=pd.read_csv(f"{RAW}/test.csv")
y=(tr.Will_Buy_EV=="Yes").astype(int).values
f=[c for c in tr.columns if c not in ("id","Will_Buy_EV")]

print("=== [A] ADVERSARIAL VALIDATION (train vs test) ===")
A=pd.concat([tr[f],te[f]],ignore_index=True)
lab=np.r_[np.zeros(len(tr)),np.ones(len(te))]
for c in f:
    if A[c].dtype==object: A[c]=A[c].astype("category")
oof=np.zeros(len(A))
for ti,vi in StratifiedKFold(5,shuffle=True,random_state=0).split(A,lab):
    m=lgb.train(dict(objective="binary",metric="auc",learning_rate=0.05,num_leaves=31,
                     verbose=-1,num_threads=-1,seed=0),
                lgb.Dataset(A.iloc[ti],lab[ti]),num_boost_round=200)
    oof[vi]=m.predict(A.iloc[vi])
aa=roc_auc_score(lab,oof)
print(f"  train-vs-test AUC = {aa:.5f}  ({'exchangeable, transductive features licensed' if aa<0.52 else 'DRIFT PRESENT'})")

print("\n=== [B] NULL IMPORTANCE (vs shuffled-target models) ===")
X=tr[f].copy()
for c in f:
    if X[c].dtype==object: X[c]=X[c].astype("category")
P=dict(objective="binary",metric="auc",learning_rate=0.05,num_leaves=31,max_depth=5,
       feature_fraction=0.3,verbose=-1,num_threads=-1,seed=0)
real=lgb.train(P,lgb.Dataset(X,y),num_boost_round=400).feature_importance("gain")
nulls=[]
for s in range(8):
    ys=np.random.RandomState(s).permutation(y)
    nulls.append(lgb.train(P,lgb.Dataset(X,ys),num_boost_round=400).feature_importance("gain"))
nulls=np.array(nulls)
print(f"  {'feature':<30} {'real_gain':>12} {'null_mean':>11} {'z':>8}")
rows=[]
for i,c in enumerate(f):
    mu,sd=nulls[:,i].mean(),nulls[:,i].std()+1e-9
    rows.append((c,real[i],mu,(real[i]-mu)/sd))
for c,r,mu,z in sorted(rows,key=lambda t:-t[3]):
    print(f"  {c:<30} {r:12.1f} {mu:11.1f} {z:8.2f}")

print("\n=== [C] PERMUTATION IMPORTANCE on OOF ===")
oofp=np.load("C:/dev/important repo/ml-bench/ev-purchase-s6e9/artifacts/oof_exp05_binned_plus_exact.npy")
base=roc_auc_score(y,oofp)
mdl=lgb.train(P,lgb.Dataset(X,y),num_boost_round=400)
b2=roc_auc_score(y,mdl.predict(X))
rng=np.random.RandomState(0); out=[]
for c in f:
    Xp=X.copy(); Xp[c]=Xp[c].values[rng.permutation(len(Xp))]
    out.append((c,b2-roc_auc_score(y,mdl.predict(Xp))))
for c,d in sorted(out,key=lambda t:-t[1]):
    print(f"  {c:<30} drop={d:+.5f}")
