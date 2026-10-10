"""batch3's remaining low-priority variants, re-run on split B with honest ES.
DART at a FIXED 2000 rounds (it ignores early stopping and was blocking the queue)."""
import sys;sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
from cv import *
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
KMF=["Annual_Income_USD","Daily_Commute_km","Age","Environmental_Concern_Level","Charging_Stations_Near_Home","Charging_Stations_Near_Work"]
sc=StandardScaler().fit(both[KMF]); km=MiniBatchKMeans(24,random_state=0,n_init=5,batch_size=4096).fit(sc.transform(both[KMF]))
def build_km(df):
    X=build(df); Z=sc.transform(df[KMF]); X["km_id"]=km.predict(Z); d=km.transform(Z); X["km_d0"]=d.min(1); X["km_d1"]=np.sort(d,1)[:,1]; return X
CATC=[c for c in RAWF if tr[c].dtype==object]
def build_oh(df):
    X=build(df)
    for c in CATC:
        for v in sorted(both[c].unique())[1:]: X[f"oh_{c}_{v}"]=(df[c].values==v).astype(np.int8)
    return X.drop(columns=CATC)
def build_ng(df): return build(df).drop(columns=["Gender"])
def fit_dart(P,seed,n=2000):
    def f(Xtr,ytr,Xva,Xt):
        Q=dict(P,seed=seed,bagging_seed=seed,feature_fraction_seed=seed,boosting="dart",drop_rate=0.05)
        m=lgb.train(Q,lgb.Dataset(Xtr,ytr),n); return m.predict(Xva),m.predict(Xt)
    return f
R={}
R["goss"]       =run_cv("b3b_goss_sB",build,keys,fit_lgb(dict(BASE,boosting="goss",bagging_fraction=1.0,bagging_freq=0),42),7)[0]
R["linear_tree"]=run_cv("b3b_lineartree_sB",build,keys,fit_lgb(dict(BASE,linear_tree=True,linear_lambda=1.0),42),7)[0]
R["catparams"]  =run_cv("b3b_catparams_sB",build,keys,fit_lgb(dict(BASE,cat_smooth=50,min_data_per_group=200,max_cat_threshold=64),42),7)[0]
R["kmeans"]     =run_cv("b3b_kmeans_sB",build_km,keys,fit_lgb(BASE,42),7)[0]
R["onehot"]     =run_cv("b3b_onehot_sB",build_oh,keys,fit_lgb(BASE,42),7)[0]
R["drop_gender"]=run_cv("b3b_dropgender_sB",build_ng,keys,fit_lgb(BASE,42),7)[0]
R["dart2000"]   =run_cv("b3b_dart2000_sB",build,keys,fit_dart(BASE,42),7)[0]
print("\n=== BATCH3b on split B (compare to exp05_full_sB) ===")
for k,v in sorted(R.items(),key=lambda kv:-kv[1]): print(f"  {k:<12} {v:.5f}")
