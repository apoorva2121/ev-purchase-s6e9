"""Step 3: target-encode EVERY column, not just the two high-cardinality ones.

The published winning recipe pairs shallow depth-5 trees with "target encodings
on all columns". We took the shallow trees and encoded only income and commute.
With depth 5 and colsample 0.30 a categorical split spends a whole level; the
target-encoded rate spends one threshold and composes far more cheaply.

  e1  TE on all 13 columns (+ the existing income/commute exact & binned keys)
  e2  e1 plus TE on concatenated pairs of the 4 DGP-relevant columns
  e3  e1 with Age and the charger counts also binned
  e4  best of the above, replicated on split B
"""
import sys,time,numpy as np,pandas as pd
sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
from cv import *
from sklearn.metrics import roc_auc_score
from itertools import combinations

ALL=[c for c in RAWF]
DGP=["Environmental_Concern_Level","Subsidy_Available","Range_Anxiety_Level","Annual_Income_USD"]
S=lambda a: pd.Series(a).astype(str).values

def keys_all(df):
    """exact+binned income/commute (what we had) PLUS every other column."""
    k=keys(df)
    for c in ALL:
        if c in TE_COLS: continue
        k.append((f"c_{c}", S(df[c].values)))
    return k

def keys_all_pairs(df):
    k=keys_all(df)
    for a,b in combinations(DGP,2):
        av = (np.rint(df[a].values).astype(np.int64)//1000) if a=="Annual_Income_USD" else df[a].values
        bv = (np.rint(df[b].values).astype(np.int64)//1000) if b=="Annual_Income_USD" else df[b].values
        k.append((f"p_{a[:6]}_{b[:6]}", np.char.add(np.char.add(S(av),"|"),S(bv))))
    return k

def keys_all_binned(df):
    k=keys_all(df)
    a=np.rint(df.Age.values).astype(np.int64)
    k.append(("age5",a//5))
    k.append(("chg_sum",(df.Charging_Stations_Near_Home.values+df.Charging_Stations_Near_Work.values)))
    return k

if __name__=="__main__":
    print("reference lgb42 (TE on 2 columns only) OOF=0.94574\n",flush=True)
    R={}
    R["e1_te_all_cols"]  = run_cv("e1_te_all_cols", build,keys_all,      fit_lgb(BASE,42),42)[0]
    R["e2_te_all_pairs"] = run_cv("e2_te_all_pairs",build,keys_all_pairs,fit_lgb(BASE,42),42)[0]
    R["e3_te_all_binned"]= run_cv("e3_te_all_binned",build,keys_all_binned,fit_lgb(BASE,42),42)[0]
    best=max(R,key=R.get); fn={"e1_te_all_cols":keys_all,"e2_te_all_pairs":keys_all_pairs,"e3_te_all_binned":keys_all_binned}[best]
    R[best+"_splitB"]=run_cv(best+"_splitB",build,fn,fit_lgb(BASE,42),7)[0]
    ys=np.random.RandomState(0).permutation(y)
    ca=run_cv("e_canary",build,keys_all,fit_lgb(BASE,42),42,labels=ys)[0]
    print(f"  TE-ALL CANARY = {ca:.5f} -> {'PASS' if abs(ca-0.5)<0.0019 else 'FAIL'}",flush=True)
    print("\n=== STEP 3 (vs lgb42 0.94574) ===")
    for k,v in sorted(R.items(),key=lambda kv:-kv[1]): print(f"  {k:<24} {v:.5f}  {v-0.94574:+.5f}")
