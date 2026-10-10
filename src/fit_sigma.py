"""The claimed N(0,1) DGP is contradicted by our own baseline. Fit the probit
properly (free scale) and recompute the ceiling from the fitted model.

CAVEAT, stated up front: this is an oracle bound CONDITIONAL on the probit's
own probabilities. If the true DGP contains value-memorisation effects the
probit cannot express, this UNDERSTATES the ceiling. It bounds the semantic
part of the problem, not the artifact part.
"""
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

RAW = "C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
tr = pd.read_csv(f"{RAW}/train.csv")
y = (tr.Will_Buy_EV == "Yes").astype(int).values

X = np.column_stack([
    tr.Annual_Income_USD.values / 1e5,
    tr.Environmental_Concern_Level.values,
    (tr.Subsidy_Available == "Yes").astype(float).values,
    (tr.Range_Anxiety_Level == "Medium").astype(float).values,
    (tr.Range_Anxiety_Level == "High").astype(float).values,
    np.ones(len(tr)),
])
names = ["inc", "concern", "subsidy", "anx_med", "anx_high", "const"]

def nll(b):
    p = np.clip(norm.cdf(X @ b), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

b0 = np.array([1.2, 0.6, 2.0, -1.0, -3.0, -5.5])
r = minimize(nll, b0, method="L-BFGS-B")
b = r.x
print("=== fitted probit (free scale) ===")
for n, v, c in zip(names, b, b0):
    print(f"  {n:<9} fitted={v:+.4f}   claimed={c:+.4f}   ratio={v/c:+.3f}")
sigma = 1.0 / (b[0] / 1.2)   # implied noise sd if the claimed inc coef is the truth
print(f"\n  implied noise sigma (vs claimed N(0,1)): {sigma:.4f}")

p = norm.cdf(X @ b)
print(f"  fitted-probit AUC          : {roc_auc_score(y, p):.5f}")
print(f"  claimed-formula AUC        : 0.93769")

def analytic_auc(p):
    p = np.sort(p)
    w1 = p / p.sum(); w0 = (1 - p) / (1 - p).sum()
    return float(np.sum(w1 * (np.cumsum(w0) - w0 / 2)))

print("\n=== ceilings ===")
print(f"  ceiling implied by FITTED probit  : {analytic_auc(p):.5f}")
print(f"  our baseline (raw features, LGBM) : 0.94219")
print(f"  reported field maximum            : ~0.94674")
print(f"  reported isotonic-oracle ceiling  : 0.94660 +/- 0.00028")

# How much of our baseline's edge over the probit is memorisation?
print(f"\n  baseline - fitted probit = {0.94219 - roc_auc_score(y, p):+.5f}")
print("  (that gap cannot come from the 4 semantic features; it is artifact)")
