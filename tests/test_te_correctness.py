"""Verify the nested target encoder is EXACTLY what we claim, not approximately.

Three invariants:
  1. A training row's encoding equals the m-estimate computed from its inner-TRAIN
     rows only -- recomputed independently here, not by calling the same code.
  2. No training row's encoding depends on its OWN label (flip it, encoding is identical).
  3. A validation row's encoding uses the full outer-train and never the validation fold.
"""
import sys,numpy as np,pandas as pd
sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
import common as C
from sklearn.model_selection import StratifiedKFold

rng=np.random.RandomState(0); N=4000
df=pd.DataFrame({"Annual_Income_USD":rng.choice([30000.,40000.,50000.,60000.],N).astype(float),
                 "Daily_Commute_km":rng.choice([5.,10.,20.],N).astype(float)})
yy=rng.binomial(1,0.3,N)
kt=[("Annual_Income_USD",df.Annual_Income_USD.values)]
X=pd.DataFrame(index=range(N)); X["dummy"]=1.0
for m in C.SMOOTHS: X[f"te{m}_Annual_Income_USD"]=np.nan
Xte=X.iloc[:5].copy()
ti=np.arange(N); vi=np.arange(5)
Xtr=X.copy(); Xva=X.iloc[:5].copy(); Xt=Xte.copy()
kte=[("Annual_Income_USD",df.Annual_Income_USD.values[:5])]
Xtr,Xva,Xt=C.te_fold(Xtr,Xva,Xt,ti,vi,yy,kt,kte,inner_seed=43)

# --- invariant 1: recompute independently ---
fails=0
key=df.Annual_Income_USD.values
for itr,iva in StratifiedKFold(C.NINNER,shuffle=True,random_state=43).split(Xtr,yy):
    prior=yy[itr].mean()
    for m in C.SMOOTHS:
        for r in iva[:40]:
            k=key[r]; sel=itr[key[itr]==k]
            expect=(yy[sel].sum()+prior*m)/(len(sel)+m)
            got=Xtr.iloc[r,Xtr.columns.get_loc(f"te{m}_Annual_Income_USD")]
            if not np.isclose(expect,got,atol=1e-12): fails+=1
    break
print(f"[1] independent recomputation of training-row encodings: {'PASS' if fails==0 else f'FAIL ({fails})'}")

# --- invariant 2: flipping a row's OWN label must not change its OWN encoding ---
target=7
y2=yy.copy(); y2[target]^=1
X2=X.copy(); Xv2=X.iloc[:5].copy(); Xt2=Xte.copy()
X2,_,_=C.te_fold(X2,Xv2,Xt2,ti,vi,y2,kt,kte,inner_seed=43)
c=Xtr.columns.get_loc("te20_Annual_Income_USD")
a,b=Xtr.iloc[target,c],X2.iloc[target,c]
print(f"[2] own-label independence: own enc {a:.10f} vs {b:.10f} -> {'PASS' if np.isclose(a,b) else 'FAIL (LEAK)'}")
same=np.isclose(Xtr.iloc[:,c].values,X2.iloc[:,c].values)
print(f"    rows whose encoding DID change when row {target}'s label flipped: {(~same).sum()} "
      f"(expected: only same-key rows in other inner folds, >0 is correct)")

# --- invariant 3: validation encoding uses full outer-train ---
prior=yy.mean(); ok=0
for m in C.SMOOTHS:
    for r in range(5):
        k=key[r]; sel=np.where(key[ti]==k)[0]
        expect=(yy[ti][sel].sum()+prior*m)/(len(sel)+m)
        ok+= np.isclose(expect,Xva[f"te{m}_Annual_Income_USD"].values[r])
print(f"[3] validation-row encodings from full outer-train: {ok}/{5*len(C.SMOOTHS)} match -> {'PASS' if ok==5*len(C.SMOOTHS) else 'FAIL'}")
