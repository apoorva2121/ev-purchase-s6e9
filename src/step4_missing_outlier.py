"""Step 4: treat the censored/imputed floors as MISSING, and add outlier scores.

Finding that motivates this: the source table was mode-imputed before synthesis,
and its modes are exactly our two floors (income 30000, commute 5.0) plus concern=1.
So those cells are placeholders, not measurements. Feeding 30000 as a number forces
a placeholder into the numeric ordering and the histogram bins; NaN lets LightGBM
learn a separate route and keeps it out of the bins.

  f1  floors -> NaN (keep the indicator flags so nothing is lost)
  f2  f1 + IsolationForest anomaly score + kNN distance (outlier-ness as a feature)
  f3  outlier scores only, floors left numeric (isolates which half matters)
  f4  best of the above on split B
"""
import sys,time,numpy as np,pandas as pd
sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
from cv import *
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

OF=["Annual_Income_USD","Daily_Commute_km","Age","Environmental_Concern_Level",
    "Charging_Stations_Near_Home","Charging_Stations_Near_Work","Number_of_Cars_Owned"]
print("fitting outlier scorers on train+test (label-free)...",flush=True)
_sc=StandardScaler().fit(both[OF]); _Z=_sc.transform(both[OF])
_iso=IsolationForest(n_estimators=200,max_samples=100000,random_state=0,n_jobs=8).fit(_Z)
_sub=_Z[np.random.RandomState(0).choice(len(_Z),60000,replace=False)]
_nn=NearestNeighbors(n_neighbors=20,n_jobs=8).fit(_sub)
def outlier_cols(df):
    Z=_sc.transform(df[OF]); d,_=_nn.kneighbors(Z)
    return -_iso.score_samples(Z), d.mean(1), d[:,-1]

def build_nan(df):
    X=build(df)
    X["Annual_Income_USD"]=np.where(df.Annual_Income_USD.values==30000,np.nan,df.Annual_Income_USD.values)
    X["Daily_Commute_km"] =np.where(df.Daily_Commute_km.values==5.0,np.nan,df.Daily_Commute_km.values)
    return X
def build_out(df):
    X=build(df); a,b,c=outlier_cols(df); X["iso"]=a; X["knn_mean"]=b; X["knn_far"]=c; return X
def build_nan_out(df):
    X=build_nan(df); a,b,c=outlier_cols(df); X["iso"]=a; X["knn_mean"]=b; X["knn_far"]=c; return X

if __name__=="__main__":
    print("reference lgb42 OOF=0.94574\n",flush=True)
    R={}
    R["f1_floors_nan"]   = run_cv("f1_floors_nan",build_nan,keys,fit_lgb(BASE,42),42)[0]
    R["f2_nan_outlier"]  = run_cv("f2_nan_outlier",build_nan_out,keys,fit_lgb(BASE,42),42)[0]
    R["f3_outlier_only"] = run_cv("f3_outlier_only",build_out,keys,fit_lgb(BASE,42),42)[0]
    best=max(R,key=R.get); fn={"f1_floors_nan":build_nan,"f2_nan_outlier":build_nan_out,"f3_outlier_only":build_out}[best]
    R[best+"_splitB"]=run_cv(best+"_splitB",fn,keys,fit_lgb(BASE,42),7)[0]
    print("\n=== STEP 4 (vs lgb42 0.94574) ===")
    for k,v in sorted(R.items(),key=lambda kv:-kv[1]): print(f"  {k:<22} {v:.5f}  {v-0.94574:+.5f}")
