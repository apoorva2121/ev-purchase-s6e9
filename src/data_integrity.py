"""Is any of this data actually wrong? Garbage, impossible values, or logical
contradictions that would mislead the model.

Synthetic data can contain combinations the real world forbids. If a rule holds in
the SOURCE table but is violated in the competition data, those violated rows are
generator errors -- and how the label behaves on them is worth knowing.
"""
import numpy as np,pandas as pd
from scipy.stats import chisquare
RAW="C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
tr=pd.read_csv(f"{RAW}/train.csv"); te=pd.read_csv(f"{RAW}/test.csv")
og=pd.read_csv(f"{RAW}/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y=(tr.Will_Buy_EV=="Yes").astype(int).values
F=[c for c in tr.columns if c not in ("id","Will_Buy_EV")]

print("=== [1] RANGE SANITY (competition vs source) ===")
for c in F:
    if tr[c].dtype==object:
        extra=set(tr[c].unique())-set(og[c].dropna().unique())
        print(f"  {c:<30} levels {sorted(tr[c].unique())}  not-in-source: {extra or 'none'}")
    else:
        print(f"  {c:<30} comp [{tr[c].min():>9.1f},{tr[c].max():>9.1f}]  src [{og[c].min():>9.1f},{og[c].max():>9.1f}]"
              f"  out-of-src-range: {int(((tr[c]<og[c].min())|(tr[c]>og[c].max())).sum())}")

print("\n=== [2] IMPOSSIBLE / SUSPECT VALUES ===")
print(f"  negative or zero income      : {(tr.Annual_Income_USD<=0).sum()}")
print(f"  age outside [18,100]         : {((tr.Age<18)|(tr.Age>100)).sum()}   range [{tr.Age.min()},{tr.Age.max()}]")
print(f"  commute <= 0                 : {(tr.Daily_Commute_km<=0).sum()}")
print(f"  concern outside 1..5         : {(~tr.Environmental_Concern_Level.isin([1,2,3,4,5])).sum()}")
print(f"  non-integer income           : {(tr.Annual_Income_USD%1!=0).sum()}")
print(f"  cars owned range             : [{tr.Number_of_Cars_Owned.min()},{tr.Number_of_Cars_Owned.max()}]")

print("\n=== [3] LOGICAL CONTRADICTIONS (rule holds in source? violated in competition?) ===")
def rule(name,fn):
    v_og=fn(og).mean(); v_tr=fn(tr).mean(); v_te=fn(te).mean()
    n=int(fn(tr).sum())
    lr=y[fn(tr).values].mean() if n else float("nan")
    print(f"  {name:<52} src {v_og:.4f} | train {v_tr:.4f} (n={n}) | test {v_te:.4f} | label rate {lr:.4f}")
rule("Home_Charging=Yes AND 0 chargers near home", lambda d:(d.Home_Charging_Possible=="Yes")&(d.Charging_Stations_Near_Home==0))
rule("Home_Charging=No AND many (>=10) near home", lambda d:(d.Home_Charging_Possible=="No")&(d.Charging_Stations_Near_Home>=10))
rule("Rural AND >=10 chargers near work",          lambda d:(d.City_Type=="Rural")&(d.Charging_Stations_Near_Work>=10))
rule("Urban AND 0 chargers anywhere",              lambda d:(d.City_Type=="Urban")&(d.Charging_Stations_Near_Home==0)&(d.Charging_Stations_Near_Work==0))
rule("0 cars owned",                               lambda d:(d.Number_of_Cars_Owned==0))
rule("Anxiety=Low AND commute>=80 AND no home chg",lambda d:(d.Range_Anxiety_Level=="Low")&(d.Daily_Commute_km>=80)&(d.Home_Charging_Possible=="No"))

print("\n=== [4] LAST-DIGIT UNIFORMITY (a generator fingerprint, and a garbage check) ===")
for c,v in [("income",np.rint(tr.Annual_Income_USD.values).astype(int)),
            ("commute*10",np.rint(tr.Daily_Commute_km.values*10).astype(int)),
            ("age",tr.Age.values)]:
    d=v%10; obs=np.bincount(d,minlength=10)
    chi,p=chisquare(obs)
    print(f"  {c:<12} last-digit counts {obs.tolist()}  chi2={chi:.0f} p={p:.2e}")

print("\n=== [5] TRAIN vs TEST level parity ===")
for c in F:
    a,b=set(tr[c].unique()),set(te[c].unique())
    if a!=b and tr[c].nunique()<100:
        print(f"  {c}: only-train {sorted(a-b)[:5]} only-test {sorted(b-a)[:5]}")
print("  (no output above = every low-cardinality level appears in both)")
