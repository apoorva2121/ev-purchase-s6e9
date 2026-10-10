"""Tabular MLP for blend diversity. Research is explicit: single-seed NNs earn
ZERO blend weight here; only multi-seed averages earn any. So 5 seeds minimum.
"""
import os;os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys,time,numpy as np,pandas as pd,torch,torch.nn as nn
sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
from common import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
torch.set_num_threads(8)

X,Xte=build(tr),build(te);X,Xte=encode_cats(X,Xte,numeric=True)
kt,kte=keys(tr),keys(te);X,Xte=init_te(X,Xte,kt)

class Net(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.f=nn.Sequential(nn.Linear(d,256),nn.BatchNorm1d(256),nn.SiLU(),nn.Dropout(0.2),
                          nn.Linear(256,128),nn.BatchNorm1d(128),nn.SiLU(),nn.Dropout(0.1),
                          nn.Linear(128,1))
    def forward(s,x):return s.f(x).squeeze(-1)

SEEDS=(0,1,2,3,4)
oof=np.zeros(len(X));pred=np.zeros(len(Xte));t0=time.time()
for ti,vi in StratifiedKFold(NFOLD,shuffle=True,random_state=SEED).split(X,y):
    Xtr,Xva,Xt=X.iloc[ti].copy(),X.iloc[vi].copy(),Xte.copy();ytr=y[ti]
    Xtr,Xva,Xt=te_fold(Xtr,Xva,Xt,ti,vi,ytr,kt,kte)
    q=QuantileTransformer(n_quantiles=1000,output_distribution="normal",subsample=200000,random_state=0)
    A=q.fit_transform(Xtr.values.astype(np.float32));B=q.transform(Xva.values.astype(np.float32))
    C=q.transform(Xt.values.astype(np.float32))
    A=np.nan_to_num(A);B=np.nan_to_num(B);C=np.nan_to_num(C)
    At=torch.tensor(A,dtype=torch.float32);yt=torch.tensor(ytr,dtype=torch.float32)
    Bt=torch.tensor(B,dtype=torch.float32);Ct=torch.tensor(C,dtype=torch.float32)
    fo=np.zeros(len(vi));ft=np.zeros(len(Xt))
    for sd in SEEDS:
        torch.manual_seed(sd);m=Net(A.shape[1])
        opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=3e-3,total_steps=12*((len(At)+4095)//4096))
        lossf=nn.BCEWithLogitsLoss()
        for ep in range(12):
            m.train();perm=torch.randperm(len(At))
            for i in range(0,len(At),4096):
                j=perm[i:i+4096];opt.zero_grad()
                l=lossf(m(At[j]),yt[j]);l.backward();opt.step();sch.step()
        m.eval()
        with torch.no_grad():
            fo+=torch.sigmoid(m(Bt)).numpy()/len(SEEDS)
            ft+=torch.sigmoid(m(Ct)).numpy()/len(SEEDS)
    oof[vi]=fo;pred+=ft/NFOLD
    print(f"  fold done auc={roc_auc_score(y[vi],fo):.5f} ({time.time()-t0:.0f}s)",flush=True)
a=roc_auc_score(y,oof)
print(f"exp28_mlp5seed OOF={a:.5f} ({time.time()-t0:.0f}s)")
np.save(f"{OUT}/oof_exp28_mlp.npy",oof);np.save(f"{OUT}/pred_exp28_mlp.npy",pred)
