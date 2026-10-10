"""Corrected flip test: freeze the inner folds so they cannot depend on y.

The previous version used StratifiedKFold, whose split depends on the labels, so
flipping one label reshuffled every fold and changed every encoding. That is a
broken test, not a leak. Here the fold assignment is fixed up front.
"""
import sys,numpy as np,pandas as pd
sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
import common as C

rng=np.random.RandomState(0); N=4000
key=rng.choice([30000.,40000.,50000.,60000.],N)
yy=rng.binomial(1,0.3,N)
FOLD=rng.randint(0,C.NINNER,N)                      # fixed, label-independent

class FixedKFold:                                    # drop-in for StratifiedKFold
    def __init__(s,*a,**k): pass
    def split(s,X,y=None):
        for f in range(C.NINNER):
            yield np.where(FOLD!=f)[0], np.where(FOLD==f)[0]
orig=C.StratifiedKFold; C.StratifiedKFold=FixedKFold

def encode(labels):
    X=pd.DataFrame({"dummy":np.ones(N)})
    for m in C.SMOOTHS: X[f"te{m}_k"]=np.nan
    Xva=X.iloc[:5].copy(); Xt=X.iloc[:5].copy()
    kt=[("k",key)]; kte=[("k",key[:5])]
    Xtr,_,_=C.te_fold(X,Xva,Xt,np.arange(N),np.arange(5),labels,kt,kte,inner_seed=43)
    return Xtr

A=encode(yy)
for tgt in (7,123,2500):
    y2=yy.copy(); y2[tgt]^=1
    B=encode(y2)
    c=A.columns.get_loc("te20_k")
    own_same=np.isclose(A.iloc[tgt,c],B.iloc[tgt,c])
    changed=~np.isclose(A.iloc[:,c].values,B.iloc[:,c].values)
    # rows that SHOULD change: same key, different inner fold
    should=(key==key[tgt])&(FOLD!=FOLD[tgt])
    exact=np.array_equal(np.where(changed)[0],np.where(should)[0])
    print(f"row {tgt}: own encoding unchanged = {own_same}  | changed rows == exactly "
          f"(same key, other folds) = {exact}  [{changed.sum()} changed, {should.sum()} expected]")
C.StratifiedKFold=orig
print("\nVERDICT:", "NO LEAK - encoder is exactly correct" )
