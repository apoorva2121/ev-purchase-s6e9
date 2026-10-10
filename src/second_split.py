"""Review A2: replicate the anchor experiments on a second split (seed 7),
all with honest ES, so every delta is judged on two independent yardsticks."""
import sys;sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
from cv import *
raw_build=lambda df: df[RAWF].copy()
R={}
R["exp01_sA"]=run_cv("exp01_exact_honest_sA",build,keys_exact,fit_lgb(BASE,42),42)[0]
R["exp00_sB"]=run_cv("exp00_raw_sB",raw_build,lambda df:[],fit_lgb(BASE,42),7)[0]
R["exp01_sB"]=run_cv("exp01_exact_sB",build,keys_exact,fit_lgb(BASE,42),7)[0]
R["exp05_sB"]=run_cv("exp05_full_sB",build,keys,fit_lgb(BASE,42),7)[0]
print("\n=== SECOND-SPLIT REPLICATION (honest ES throughout) ===")
print(f"  split A: exp05=0.94574 (lgb42)  exp01={R['exp01_sA']:.5f}  delta05-01={0.94574-R['exp01_sA']:+.5f}")
print(f"  split B: exp05={R['exp05_sB']:.5f}  exp01={R['exp01_sB']:.5f}  delta05-01={R['exp05_sB']-R['exp01_sB']:+.5f}")
print(f"  split B: exp01-exp00={R['exp01_sB']-R['exp00_sB']:+.5f}   (split A was +0.00346)")
