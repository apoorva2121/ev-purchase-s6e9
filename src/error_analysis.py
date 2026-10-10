"""Where does exp01 fail, and is the failure structured?

DISCIPLINE: residuals are a diagnostic only. Any feature proposed at the end
must be computable from X alone, with no access to y.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RAW = "C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
OUT = "C:/dev/important repo/ml-bench/ev-purchase-s6e9/artifacts"
tr = pd.read_csv(f"{RAW}/train.csv")
y = (tr.Will_Buy_EV == "Yes").astype(int).values
p = np.load(f"{OUT}/oof_exp01.npy")
res = y - p
print(f"exp01 OOF AUC {roc_auc_score(y,p):.5f}   logloss "
      f"{-(y*np.log(p)+(1-y)*np.log(1-p)).mean():.5f}")

# ---------- 1. calibration: is the model wrong anywhere systematically? ----------
print("\n=== [1] CALIBRATION BY PREDICTED DECILE ===")
dec = pd.qcut(p, 10, labels=False)
g = pd.DataFrame({"p": p, "y": y, "d": dec}).groupby("d").agg(
    pred=("p","mean"), obs=("y","mean"), n=("y","size"))
g["z"] = (g.obs-g.pred)/np.sqrt(g.pred*(1-g.pred)/g.n)
print(g.round(5).to_string())

# ---------- 2. PER-EXACT-VALUE OVERDISPERSION (the key test) ----------
print("\n=== [2] PER-INCOME-VALUE OVERDISPERSION ===")
df = pd.DataFrame({"v": tr.Annual_Income_USD.values, "y": y, "p": p})
agg = df.groupby("v").agg(n=("y","size"), obs=("y","mean"), pred=("p","mean"))
for nmin in [50, 100, 200]:
    a = agg[agg.n >= nmin]
    z = (a.obs - a.pred) / np.sqrt(a.pred*(1-a.pred)/a.n)
    print(f"  values with n>={nmin}: {len(a):>5}  z-score sd = {z.std():.4f}  "
          f"(1.000 = fully captured)  frac|z|>2 = {(z.abs()>2).mean():.4f} (exp 0.0455)")
a = agg[agg.n >= 100]
z = (a.obs - a.pred)/np.sqrt(a.pred*(1-a.pred)/a.n)
print(f"  -> residual per-value rate SD  = {(a.obs-a.pred).std():.5f}")
print(f"     binomial-only expectation   = {np.sqrt((a.pred*(1-a.pred)/a.n)).mean():.5f}")

# same for commute
df2 = pd.DataFrame({"v": tr.Daily_Commute_km.values, "y": y, "p": p})
a2 = df2.groupby("v").agg(n=("y","size"), obs=("y","mean"), pred=("p","mean"))
a2 = a2[a2.n >= 100]
z2 = (a2.obs-a2.pred)/np.sqrt(a2.pred*(1-a2.pred)/a2.n)
print(f"  commute values n>=100: {len(a2)}  z sd = {z2.std():.4f}")

# ---------- 3. is the residual PREDICTABLE from X alone? ----------
print("\n=== [3] CAN A SECOND MODEL PREDICT THE RESIDUAL FROM X? ===")
feats = [c for c in tr.columns if c not in ("id","Will_Buy_EV")]
X = tr[feats].copy()
for c in feats:
    if X[c].dtype == object: X[c] = X[c].astype("category")
oof_r = np.zeros(len(X))
for ti, vi in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    m = lgb.train(dict(objective="regression", metric="l2", learning_rate=0.05,
                       num_leaves=64, max_depth=8, feature_fraction=0.8,
                       verbose=-1, num_threads=-1, seed=1),
                  lgb.Dataset(X.iloc[ti], res[ti]), num_boost_round=300)
    oof_r[vi] = m.predict(X.iloc[vi])
ss_res = ((res-oof_r)**2).sum(); ss_tot = ((res-res.mean())**2).sum()
print(f"  R^2 of residual model on X : {1-ss_res/ss_tot:+.5f}  (>0 = leftover signal)")
print(f"  corr(pred_resid, resid)    : {np.corrcoef(oof_r,res)[0,1]:+.5f}")
print(f"  AUC of (p + pred_resid)    : {roc_auc_score(y, p+oof_r):.5f}  vs base {roc_auc_score(y,p):.5f}")

# ---------- 4. where are the worst rows concentrated? ----------
print("\n=== [4] PROFILE OF THE WORST-PREDICTED ROWS ===")
loss = -(y*np.log(np.clip(p,1e-9,1)) + (1-y)*np.log(np.clip(1-p,1e-9,1)))
hard = loss > np.quantile(loss, 0.95)
print(f"  worst 5% by logloss: n={hard.sum()}, actual buy rate {y[hard].mean():.4f} "
      f"vs {y[~hard].mean():.4f} overall")
for c in feats:
    if tr[c].dtype == object or tr[c].nunique() <= 20:
        t = pd.crosstab(tr[c], hard, normalize="index")
        if True in t.columns:
            sp = t[True].max() - t[True].min()
            if sp > 0.03:
                print(f"  {c}: hard-rate spread {sp:.3f}")
                print("    " + t[True].round(3).to_dict().__repr__())
    else:
        q = pd.qcut(tr[c], 10, labels=False, duplicates="drop")
        r = pd.Series(hard).groupby(q).mean()
        print(f"  {c} hard-rate by decile: {r.round(3).tolist()}")

# ---------- 5. uncertainty band ----------
print("\n=== [5] LOW-CONFIDENCE BAND ===")
unc = (p > 0.10) & (p < 0.40)
print(f"  rows with 0.10<p<0.40: {unc.sum()} ({unc.mean():.2%}), "
      f"buy rate {y[unc].mean():.4f}, AUC within band {roc_auc_score(y[unc], p[unc]):.5f}")
