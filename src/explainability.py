"""Information Value + WOE scorecard: a model-independent feature ranking and a
fully explainable baseline, to sit alongside the black-box GBDT.

IV is unbiased w.r.t. cardinality in a way gain-importance is not, which matters
here: one feature has 13,214 levels and nine have under 45.
"""
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RAW = "C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
tr = pd.read_csv(f"{RAW}/train.csv")
y = (tr.Will_Buy_EV == "Yes").astype(int).values
feats = [c for c in tr.columns if c not in ("id", "Will_Buy_EV")]
NP, NN = y.sum(), len(y) - y.sum()

def binned(s, nbins=20):
    if s.dtype == object or s.nunique() <= nbins:
        return s.astype(str)
    return pd.qcut(s, nbins, duplicates="drop").astype(str)

def woe_iv(keys, y):
    d = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    pos, n = d["sum"].values, d["count"].values
    neg = n - pos
    ppos = (pos + 0.5) / (NP + 0.5); pneg = (neg + 0.5) / (NN + 0.5)
    woe = np.log(ppos / pneg)
    iv = float(np.sum((ppos - pneg) * woe))
    return pd.Series(woe, index=d.index), iv

def band(iv):
    return ("useless   " if iv < 0.02 else "weak      " if iv < 0.1 else
            "medium    " if iv < 0.3 else "strong    " if iv < 0.5 else "suspicious")

print("=== INFORMATION VALUE (model-independent feature ranking) ===")
print(f"{'feature':<30} {'IV':>8}  strength     levels")
rows = []
for c in feats:
    k = binned(tr[c])
    _, iv = woe_iv(k, y)
    rows.append((c, iv, k.nunique()))
for c, iv, nl in sorted(rows, key=lambda r: -r[1]):
    print(f"{c:<30} {iv:8.4f}  {band(iv)}  {nl}")

print("\n=== WOE SCORECARD: a fully explainable model ===")
# Additive WOE scorecard, fold-safe, no interactions. This is the model you can
# print on one page and hand to a regulator.
oof = np.zeros(len(tr))
for ti, vi in StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y):
    tot = np.zeros(len(vi))
    for c in feats:
        k = binned(tr[c])
        w, _ = woe_iv(k.iloc[ti], y[ti])
        tot += k.iloc[vi].map(w).fillna(0.0).values
    oof[vi] = tot
print(f"  additive WOE scorecard OOF AUC : {roc_auc_score(y, oof):.5f}")

# the recovered generating function, as an interpretable score
P = dict(inc=1.4095, concern=0.7358, subsidy=2.3410, anx_med=-1.0172,
         anx_high=-2.9972, const=-6.7148)
s = (P["inc"]*tr.Annual_Income_USD/1e5 + P["concern"]*tr.Environmental_Concern_Level
     + P["subsidy"]*(tr.Subsidy_Available=="Yes") + P["anx_med"]*(tr.Range_Anxiety_Level=="Medium")
     + P["anx_high"]*(tr.Range_Anxiety_Level=="High") + P["const"]).values
print(f"  recovered probit score OOF-free: {roc_auc_score(y, s):.5f}  (5 parameters, no fitting per fold)")
print(f"  sigma-units from threshold: mean {s.mean():.3f}, sd {s.std():.3f}")
print("\n  reference: exp01 GBDT = 0.94565")
print(f"  cost of full explainability (scorecard vs GBDT): {roc_auc_score(y,oof)-0.94565:+.5f}")
print(f"  cost of a 5-parameter equation vs GBDT        : {roc_auc_score(y,s)-0.94565:+.5f}")
