"""Can model-based imputation work here, and what do the features actually depend on?

If income is predictable from the other columns, MICE / regression imputation of the
floor rows is worth building. If it is not, any imputer will return ~the mean for
every row and the exercise is a no-op. Measure, don't assume.
"""
import numpy as np,pandas as pd,lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency

RAW="C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
tr=pd.read_csv(f"{RAW}/train.csv"); y=(tr.Will_Buy_EV=="Yes").astype(int).values
F=[c for c in tr.columns if c not in ("id","Will_Buy_EV")]
X=tr[F].copy()
for c in F:
    if X[c].dtype==object: X[c]=X[c].astype("category")

print("=== [1] IS INCOME PREDICTABLE FROM THE OTHER FEATURES? ===")
print("    (if R^2 ~ 0, model-based imputation returns the mean and cannot help)")
for tgt in ["Annual_Income_USD","Daily_Commute_km"]:
    floor=30000 if tgt=="Annual_Income_USD" else 5.0
    m=(tr[tgt].values!=floor)                       # fit on genuine (non-placeholder) rows
    Xa=X.loc[m,[c for c in F if c!=tgt]]; ya=tr[tgt].values[m]
    oof=np.zeros(len(ya))
    for ti,vi in KFold(5,shuffle=True,random_state=0).split(Xa):
        mdl=lgb.train(dict(objective="regression",metric="l2",learning_rate=0.05,num_leaves=63,
                           verbose=-1,num_threads=8,seed=0),
                      lgb.Dataset(Xa.iloc[ti],ya[ti]),num_boost_round=400)
        oof[vi]=mdl.predict(Xa.iloc[vi])
    print(f"  {tgt:<22} R^2 = {r2_score(ya,oof):+.5f}   sd(y)={ya.std():.1f}  sd(resid)={(ya-oof).std():.1f}")

print("\n=== [2] MUTUAL INFORMATION with the target (nats) ===")
Xn=X.copy()
for c in F:
    if str(Xn[c].dtype)=="category": Xn[c]=Xn[c].cat.codes
mi=mutual_info_classif(Xn,y,discrete_features=[Xn[c].nunique()<50 for c in F],random_state=0)
for c,v in sorted(zip(F,mi),key=lambda t:-t[1]): print(f"  {c:<30} {v:.5f}")

print("\n=== [3] CHI-SQUARE: is each feature independent of the target? ===")
for c in F:
    v=tr[c] if tr[c].nunique()<=20 else pd.qcut(tr[c],20,duplicates='drop')
    chi,p,dof,_=chi2_contingency(pd.crosstab(v,y))
    cram=np.sqrt(chi/(len(tr)*(min(pd.crosstab(v,y).shape)-1)))
    print(f"  {c:<30} chi2={chi:>12.1f}  p={p:.3e}  Cramers_V={cram:.4f}")

print("\n=== [4] FEATURE-FEATURE DEPENDENCE (Cramers V, categorical pairs) ===")
lc=[c for c in F if tr[c].nunique()<=20]
out=[]
for i,a in enumerate(lc):
    for b in lc[i+1:]:
        t=pd.crosstab(tr[a],tr[b]); chi,p,_,_=chi2_contingency(t)
        out.append((a,b,np.sqrt(chi/(len(tr)*(min(t.shape)-1))),p))
for a,b,v,p in sorted(out,key=lambda t:-t[2])[:8]:
    print(f"  {a:<28} x {b:<28} V={v:.4f} p={p:.2e}")
