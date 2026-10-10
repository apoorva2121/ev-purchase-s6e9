"""Review A3/B2: 10-seed bag of the best config, equal rank weights, nothing fitted on OOF."""
import sys,os;sys.path.insert(0,"C:/dev/important repo/ml-bench/ev-purchase-s6e9/src")
from cv import *
from scipy.stats import rankdata
for s in (303,404,505,606,707,808,909): run_cv(f"lgb{s}",build,keys,fit_lgb(BASE,s),42)
oofs=[];preds=[];names=[]
for s in (42,101,202,303,404,505,606,707,808,909):
    f=f"{OUT}/oof_lgb{s}.npy"
    if os.path.exists(f): oofs.append(np.load(f));preds.append(np.load(f"{OUT}/pred_lgb{s}.npy"));names.append(s)
solo=[roc_auc_score(y,o) for o in oofs]
bag=np.mean([rankdata(o)/len(y) for o in oofs],0); bagt=np.mean([rankdata(p)/len(p) for p in preds],0)
print(f"\n=== SEED BAG ({len(names)} seeds, equal rank weights) ===")
print(f"  solo mean={np.mean(solo):.5f} sd={np.std(solo):.5f} min={min(solo):.5f} max={max(solo):.5f}")
print(f"  BAG OOF = {roc_auc_score(y,bag):.5f}   gain over best single = {roc_auc_score(y,bag)-max(solo):+.5f}")
np.save(f"{OUT}/oof_seedbag10.npy",bag);np.save(f"{OUT}/pred_seedbag10.npy",bagt)
