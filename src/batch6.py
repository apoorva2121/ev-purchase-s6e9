"""Review B1 rows 2, 4, 5 + B3 canary: exact-pair TE shrunk toward income TE,
partial source-row matching, anxiety-sigma diagnostic, canary on honest pipeline."""
import sys;sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
from cv import *
from scipy.stats import norm
from scipy.optimize import minimize
R={}
# ---- exact-pair TE, hierarchical toward the income TE ----
def keys_pairs(df):
    k=keys(df); inc=df.Annual_Income_USD.astype(str)
    k+=[("inc_x_sub",(inc+"|"+df.Subsidy_Available.astype(str)).values),
        ("inc_x_con",(inc+"|"+df.Environmental_Concern_Level.astype(str)).values),
        ("inc_x_anx",(inc+"|"+df.Range_Anxiety_Level.astype(str)).values)]
    return k
H=[("inc_x_sub","Annual_Income_USD"),("inc_x_con","Annual_Income_USD"),("inc_x_anx","Annual_Income_USD")]
R["pair_hier"]=run_cv("b6_pairhier_sB",build,keys_pairs,fit_lgb(BASE,42),7,hier=H)[0]

# ---- partial source-row matching ----
key3=["Annual_Income_USD","Age","Daily_Commute_km"]
other=[c for c in RAWF if c not in key3]
src=og.dropna(subset=key3).reset_index(drop=True); src["src_id"]=np.arange(len(src))
def src_match(df):
    d=df[["id"]+key3+other].merge(src[key3+other+["src_id"]],on=key3,how="left",suffixes=("","_s"))
    hit=d.src_id.notna()
    nm=np.zeros(len(d),int)
    for c in other: nm+=(d[c].astype(str).values==d[c+"_s"].astype(str).values)&hit.values
    d["nm"]=np.where(hit,nm,-1)
    d=d.sort_values(["id","nm"],ascending=[True,False]).drop_duplicates("id")
    d=d.set_index("id").loc[df.id.values]
    return d.nm.values, np.where(d.src_id.notna(),d.src_id.fillna(-1).astype(int).astype(str),"none")
nm_tr,sid_tr=src_match(tr); nm_te,sid_te=src_match(te)
print(f"  triple-match rate train={np.mean(nm_tr>=0):.4f} test={np.mean(nm_te>=0):.4f}; "
      f"among matched, mean extra cols matched={nm_tr[nm_tr>=0].mean():.2f} of {len(other)}",flush=True)
print(f"  label rate by n_matched: "+", ".join(f"{k}:{y[nm_tr==k].mean():.4f}(n={int((nm_tr==k).sum())})" for k in sorted(set(nm_tr)) if (nm_tr==k).sum()>500),flush=True)
SID={"tr":sid_tr,"te":sid_te}; NM={"tr":nm_tr,"te":nm_te}
def build_src(df):
    X=build(df); which="tr" if len(df)==len(tr) else "te"; X["n_src_match"]=NM[which]; return X
def keys_src(df):
    which="tr" if len(df)==len(tr) else "te"; return keys(df)+[("src_row",SID[which])]
R["src_row"]=run_cv("b6_srcrow_sB",build_src,keys_src,fit_lgb(BASE,42),7)[0]

# ---- anxiety-sigma diagnostic (closes the loose thread; no model) ----
print("\n  probit per anxiety level (income coef = 1/sigma proxy):")
for lvl in ["Low","Medium","High"]:
    m=(tr.Range_Anxiety_Level==lvl).values
    Xp=np.column_stack([tr.Annual_Income_USD.values[m]/1e5,tr.Environmental_Concern_Level.values[m],
                        (tr.Subsidy_Available.values[m]=="Yes").astype(float),np.ones(m.sum())])
    yy=y[m]
    nll=lambda b:-(yy*np.log(np.clip(norm.cdf(Xp@b),1e-12,1))+(1-yy)*np.log(np.clip(1-norm.cdf(Xp@b),1e-12,1))).sum()
    b=minimize(nll,np.array([1.4,0.7,2.3,-6.7]),method="L-BFGS-B").x
    print(f"    {lvl:<7} n={m.sum():>7} pos={yy.mean():.4f}  inc={b[0]:+.3f} concern={b[1]:+.3f} subsidy={b[2]:+.3f} const={b[3]:+.3f}  implied sigma={1.2/b[0]:.3f}",flush=True)

# ---- canary on the honest-ES pipeline (review check 5) ----
ys=np.random.RandomState(0).permutation(y)
a,_,_=run_cv("canary_honest",build,keys,fit_lgb(BASE,42),42,labels=ys)
print(f"  CANARY honest-ES pipeline = {a:.5f}  band |a-0.5|<0.0019 -> {'PASS' if abs(a-0.5)<0.0019 else 'FAIL'}",flush=True)
print("\n=== BATCH6 on split B ===")
for k,v in R.items(): print(f"  {k:<12} {v:.5f}")
