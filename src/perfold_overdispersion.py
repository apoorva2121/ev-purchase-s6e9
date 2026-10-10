"""Corrected per-value overdispersion test (review finding A1).

The earlier test pooled obs and pred across folds. Fold-k predictions for value
c are built from the other folds' labels of value c, so pooled pred_c shares
noise with pooled obs_c and the z-SD is deflated by construction.

Fix: per fold k, compare obs over fold-k rows only to pred from the fold-k model
only. Those are independent given the true rate. Also compute the NULL z-SD by
drawing labels from the model's own probabilities, so the reference is exact
for this construction (discreteness, estimated variance) rather than 1.0.
Ratio real/null > ~1.1 = unencoded structure on that key.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
RAW="C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
OUT="C:/dev/important repo/ml-bench/ev-purchase-s6e9/artifacts"
tr=pd.read_csv(f"{RAW}/train.csv"); y=(tr.Will_Buy_EV=="Yes").astype(int).values
p=np.load(f"{OUT}/oof_exp05_binned_plus_exact.npy")
fold=np.zeros(len(y),int)
for k,(ti,vi) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,y)): fold[vi]=k

inc=tr.Annual_Income_USD.values; cmt=tr.Daily_Commute_km.values
S=lambda a: pd.Series(a).astype(str).values
def pair(a,b): return np.char.add(np.char.add(S(a),"|"),S(b))
KEYS={
 "income exact":            S(inc),
 "commute exact":           S(cmt),
 "(income,subsidy)":        pair(inc,tr.Subsidy_Available.values),
 "(income,concern)":        pair(inc,tr.Environmental_Concern_Level.values),
 "(income,anxiety)":        pair(inc,tr.Range_Anxiety_Level.values),
 "(income,home_charging)":  pair(inc,tr.Home_Charging_Possible.values),
 "(income,commute)":        pair(inc,cmt),
 "(commute,subsidy)":       pair(cmt,tr.Subsidy_Available.values),
 "(commute,anxiety)":       pair(cmt,tr.Range_Anxiety_Level.values),
 "(income,city)":           pair(inc,tr.City_Type.values),
}
def zstats(key,yy,pp,nmin,perfold=True):
    zs=[]
    groups=[fold==k for k in range(5)] if perfold else [np.ones(len(yy),bool)]
    for m in groups:
        d=pd.DataFrame({"k":key[m],"y":yy[m],"p":pp[m]}).groupby("k").agg(n=("y","size"),obs=("y","mean"),pred=("p","mean"))
        d=d[d.n>=nmin]
        zs.append(((d.obs-d.pred)/np.sqrt(d.pred*(1-d.pred)/d.n)).values)
    z=np.concatenate(zs); z=z[np.isfinite(z)]
    return z.std(),(np.abs(z)>2).mean(),len(z)
rng=np.random.RandomState(0)
SIMS=[(rng.rand(len(y))<p).astype(int) for _ in range(3)]

print("=== DEMONSTRATING A1: pooled vs per-fold on income exact, n>=100 ===")
for pf,lab in ((False,"pooled (old test)"),(True,"per-fold (corrected)")):
    sd,f2,n=zstats(KEYS["income exact"],y,p,100,pf)
    sd0=np.mean([zstats(KEYS["income exact"],s,p,100,pf)[0] for s in SIMS])
    print(f"  {lab:<22} cells={n:>6}  real z-SD={sd:.3f}   null z-SD={sd0:.3f}   ratio={sd/sd0:.3f}")

print("\n=== PER-FOLD TEST ON EVERY KEY (ratio real/null; >1.10 = structure left) ===")
print(f"{'key':<24}{'n>=':>5}{'cells':>8}{'real':>8}{'null':>8}{'ratio':>8}{'|z|>2 real':>12}{'null':>8}")
rows=[]
for name,key in KEYS.items():
    for nmin in (20,50):
        sd,f2,n=zstats(key,y,p,nmin)
        if n<50: continue
        sims=[zstats(key,s,p,nmin) for s in SIMS]
        sd0=np.mean([s[0] for s in sims]); f20=np.mean([s[1] for s in sims])
        rows.append((name,nmin,n,sd,sd0,sd/sd0,f2,f20))
        print(f"{name:<24}{nmin:>5}{n:>8}{sd:>8.3f}{sd0:>8.3f}{sd/sd0:>8.3f}{f2:>12.4f}{f20:>8.4f}",flush=True)
flag=[r for r in rows if r[5]>1.10]
print("\nKEYS WITH STRUCTURE LEFT (ratio>1.10):", [(r[0],r[1],round(r[5],3)) for r in flag] or "none")
pd.DataFrame(rows,columns=["key","nmin","cells","real_sd","null_sd","ratio","frac2_real","frac2_null"]).to_csv(f"{OUT}/perfold_overdispersion.csv",index=False)
