"""Review B2 + B1 hierarchical shrink. All on split B (seed 7), honest ES.
Anything clearing +0.00016 vs exp05_full_sB gets confirmed on split A."""
import sys;sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
from cv import *
R={}
R["hier250"]=run_cv("b5_hier250_sB",build,keys,fit_lgb(BASE,42),7,hier=[("Annual_Income_USD","incbin250")])[0]
R["m_1_500"]=run_cv("b5_m1to500_sB",build,keys,fit_lgb(BASE,42),7,smooths=[1,5,20,100,500])[0]
R["mdl50"]  =run_cv("b5_mdl50_sB",build,keys,fit_lgb(dict(BASE,min_data_in_leaf=50),42),7)[0]
R["mdl200"] =run_cv("b5_mdl200_sB",build,keys,fit_lgb(dict(BASE,min_data_in_leaf=200),42),7)[0]
R["lr01"]   =run_cv("b5_lr01_sB",build,keys,fit_lgb(dict(BASE,learning_rate=0.01),42),7)[0]
def build_nat(df):
    X=build(df); X["income_cat"]=df.Annual_Income_USD.astype(str).values; return X
keys_nat=lambda df:[("Daily_Commute_km",df.Daily_Commute_km.values)]+[(f"incbin{b}",(df.Annual_Income_USD.values//b).astype(np.int64)) for b in BINS]
Pn=dict(BASE,cat_smooth=50,max_cat_threshold=64,min_data_per_group=50,cat_l2=10)
R["lgb_native_income"]=run_cv("b5_lgbnative_income_sB",build_nat,keys_nat,fit_lgb(Pn,42),7,extra_cat=("income_cat",))[0]
def build_str(df):
    X=df[RAWF].copy(); X["Annual_Income_USD"]=X.Annual_Income_USD.astype(str); X["Daily_Commute_km"]=X.Daily_Commute_km.astype(str)
    X["is_income_floor"]=(df.Annual_Income_USD==30000).astype(int); X["is_commute_floor"]=(df.Daily_Commute_km==5.0).astype(int)
    X["freq_income"]=df.Annual_Income_USD.map(both.Annual_Income_USD.value_counts()).values
    X["dist_to_orig_income"]=gap(df.Annual_Income_USD.values); return X
catcols=[c for c in RAWF if tr[c].dtype==object]+["Annual_Income_USD","Daily_Commute_km"]
R["cat_solo_native"]=run_cv("b5_catsolo_native_sB",build_str,lambda df:[],fit_cat_native(42,catcols),7,cat_mode="string")[0]
print("\n=== BATCH5 on split B (compare to exp05_full_sB) ===")
for k,v in sorted(R.items(),key=lambda kv:-kv[1]): print(f"  {k:<20} {v:.5f}")
