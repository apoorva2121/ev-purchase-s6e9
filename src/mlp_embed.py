"""Review B2 last row: MLP with a learned embedding on exact income (13,214
entries) and commute - the neural equivalent of TE, learned end-to-end, and the
only member plausibly decorrelated from the GBDTs. 5 seeds, fixed epochs, no ES
on the scored fold."""
import os;os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys,time,numpy as np,pandas as pd,torch,torch.nn as nn
sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
from common import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
torch.set_num_threads(8)
inc_codes={v:i+1 for i,v in enumerate(sorted(both.Annual_Income_USD.unique()))}
cmt_codes={v:i+1 for i,v in enumerate(sorted(both.Daily_Commute_km.unique()))}
def codes(df): return df.Annual_Income_USD.map(inc_codes).values, df.Daily_Commute_km.map(cmt_codes).values
X,Xte=build(tr),build(te); X,Xte=encode_cats(X,Xte,numeric=True)
kt,kte=keys(tr),keys(te); X,Xte=init_te(X,Xte,kt)
ci_tr,cc_tr=codes(tr); ci_te,cc_te=codes(te)
class Net(nn.Module):
    def __init__(s,d):
        super().__init__(); s.ei=nn.Embedding(len(inc_codes)+1,16); s.ec=nn.Embedding(len(cmt_codes)+1,8)
        s.f=nn.Sequential(nn.Linear(d+24,256),nn.BatchNorm1d(256),nn.SiLU(),nn.Dropout(0.2),
                          nn.Linear(256,128),nn.BatchNorm1d(128),nn.SiLU(),nn.Dropout(0.1),nn.Linear(128,1))
    def forward(s,x,i,c): return s.f(torch.cat([x,s.ei(i),s.ec(c)],1)).squeeze(-1)
SEEDS=(0,1,2,3,4); oof=np.zeros(len(X)); pred=np.zeros(len(Xte)); t0=time.time()
for ti,vi in StratifiedKFold(NFOLD,shuffle=True,random_state=SEED).split(X,y):
    Xtr,Xva,Xt=X.iloc[ti].copy(),X.iloc[vi].copy(),Xte.copy(); ytr=y[ti]
    Xtr,Xva,Xt=te_fold(Xtr,Xva,Xt,ti,vi,ytr,kt,kte)
    q=QuantileTransformer(n_quantiles=1000,output_distribution="normal",subsample=200000,random_state=0)
    A=np.nan_to_num(q.fit_transform(Xtr.values.astype(np.float32))); B=np.nan_to_num(q.transform(Xva.values.astype(np.float32))); C=np.nan_to_num(q.transform(Xt.values.astype(np.float32)))
    At,Bt,Ct=[torch.tensor(v,dtype=torch.float32) for v in (A,B,C)]; yt=torch.tensor(ytr,dtype=torch.float32)
    It,Iv,Ite=[torch.tensor(v,dtype=torch.long) for v in (ci_tr[ti],ci_tr[vi],ci_te)]
    Kt,Kv,Kte=[torch.tensor(v,dtype=torch.long) for v in (cc_tr[ti],cc_tr[vi],cc_te)]
    fo=np.zeros(len(vi)); ft=np.zeros(len(Xt))
    for sd in SEEDS:
        torch.manual_seed(sd); m=Net(A.shape[1]); opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4)
        steps=14*((len(At)+4095)//4096); sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=3e-3,total_steps=steps); lossf=nn.BCEWithLogitsLoss()
        for ep in range(14):
            m.train(); perm=torch.randperm(len(At))
            for i in range(0,len(At),4096):
                j=perm[i:i+4096]; opt.zero_grad(); l=lossf(m(At[j],It[j],Kt[j]),yt[j]); l.backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad(): fo+=torch.sigmoid(m(Bt,Iv,Kv)).numpy()/len(SEEDS); ft+=torch.sigmoid(m(Ct,Ite,Kte)).numpy()/len(SEEDS)
    oof[vi]=fo; pred+=ft/NFOLD
    print(f"  fold auc={roc_auc_score(y[vi],fo):.5f} ({time.time()-t0:.0f}s)",flush=True)
a=roc_auc_score(y,oof); print(f"mlp_embed_5seed OOF={a:.5f} ({time.time()-t0:.0f}s)")
np.save(f"{OUT}/oof_mlp_embed.npy",oof); np.save(f"{OUT}/pred_mlp_embed.npy",pred)
