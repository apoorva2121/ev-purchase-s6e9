"""Recover the data-generating process, then work out what is left to win.

If the label rule is score>5.5 with N(0,1) noise, the Bayes predictor is
Phi(score-5.5) and the ceiling AUC is computable analytically. Anything a model
gains above that comes from the competition's resampling artifacts, not semantics.
"""
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RAW = "C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
tr = pd.read_csv(f"{RAW}/train.csv")
og = pd.read_csv(f"{RAW}/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y = (tr.Will_Buy_EV == "Yes").astype(int).values

def design(d):
    return pd.DataFrame({
        "inc":     d.Annual_Income_USD / 1e5,
        "concern": d.Environmental_Concern_Level,
        "subsidy": (d.Subsidy_Available == "Yes").astype(float),
        "anx_med": (d.Range_Anxiety_Level == "Medium").astype(float),
        "anx_high":(d.Range_Anxiety_Level == "High").astype(float),
    })

CLAIM = dict(inc=1.2, concern=0.6, subsidy=2.0, anx_med=-1.0, anx_high=-3.0)
THRESH = 5.5

print("=== [1] Does the claimed formula hold on the ORIGINAL 10k? ===")
ogc = og.dropna(subset=["Annual_Income_USD","Environmental_Concern_Level"]).reset_index(drop=True)
print(f"  usable original rows after dropping nulls: {len(ogc)} of {len(og)}")
Do, yo = design(ogc), (ogc.Will_Buy_EV == "Yes").astype(int).values
s_og = sum(CLAIM[k] * Do[k] for k in CLAIM)
print(f"  claimed-formula score AUC on original : {roc_auc_score(yo, s_og):.5f}")
print(f"  deterministic rule (score>5.5) acc    : {((s_og>THRESH).astype(int)==yo).mean():.5f}")

# probit refit on the original to recover coefficients empirically
pr = LogisticRegression(max_iter=2000, C=1e6).fit(Do, yo)
scale = 1.702  # logistic -> probit conversion
print("  refit coefficients (converted toward probit scale):")
for k, c in zip(Do.columns, pr.coef_[0]):
    print(f"    {k:<9} refit={c/scale:+.3f}   claimed={CLAIM[k]:+.3f}")
print(f"    intercept  refit={pr.intercept_[0]/scale:+.3f}   claimed={-THRESH:+.3f}")

print("\n=== [2] Formula score on the COMPETITION data ===")
D = design(tr)
s = sum(CLAIM[k] * D[k] for k in CLAIM)
print(f"  claimed-formula score AUC (no training): {roc_auc_score(y, s):.5f}")
p_bayes = norm.cdf(s - THRESH)
print(f"  Phi(score-5.5) as probability, AUC     : {roc_auc_score(y, p_bayes):.5f}")
print(f"  its mean {p_bayes.mean():.5f} vs actual base rate {y.mean():.5f}")

# refit the formula ON the competition data - does the DGP differ here?
pr2 = LogisticRegression(max_iter=2000, C=1e6).fit(D, y)
s2 = pr2.decision_function(D)
print(f"  refit-on-competition score AUC         : {roc_auc_score(y, s2):.5f}")
print("  refit coefficients (competition):")
for k, c in zip(D.columns, pr2.coef_[0]):
    print(f"    {k:<9} refit={c/scale:+.3f}   claimed={CLAIM[k]:+.3f}")

print("\n=== [3] ANALYTIC CEILING under the stated DGP ===")
# If p_i = Phi(score_i - 5.5) is the true probability, expected AUC of the
# perfect predictor is P(p_i > p_j | y_i=1, y_j=0), computable by integration
# over the empirical distribution of p.
p = np.sort(p_bayes)
w1 = p / p.sum()                      # density of positives over p
w0 = (1 - p) / (1 - p).sum()          # density of negatives over p
cdf0 = np.cumsum(w0)
auc_ceiling = float(np.sum(w1 * (cdf0 - w0 / 2)))
print(f"  analytic max achievable AUC under this DGP: {auc_ceiling:.5f}")
print(f"  our current baseline                      : 0.94219")
print(f"  -> headroom vs DGP ceiling                : {auc_ceiling-0.94219:+.5f}")

print("\n=== [4] What signal exists BEYOND the formula? ===")
# Residualise: bucket by Bayes probability, then look for label-rate variation
# driven by features that are NOT in the formula.
bucket = pd.qcut(p_bayes, 50, labels=False, duplicates="drop")
extras = ["Age","Gender","City_Type","Daily_Commute_km","Number_of_Cars_Owned",
          "Current_Car_Type","Home_Charging_Possible",
          "Charging_Stations_Near_Home","Charging_Stations_Near_Work"]
print("  within-bucket residual AUC of each NON-formula feature")
print("  (0.500 = the formula already explains it; higher = real extra signal)")
res = y - p_bayes
rows = []
for c in extras:
    v = tr[c]
    if v.dtype == object:
        v = v.map(tr.groupby(c).Will_Buy_EV.apply(lambda s:(s=="Yes").mean()))
    # correlation of feature with residual, and residual-AUC
    a = roc_auc_score(y, v.values) if v.nunique() > 1 else 0.5
    corr = np.corrcoef(v.values.astype(float), res)[0,1]
    rows.append((c, a, corr))
for c, a, corr in sorted(rows, key=lambda r: -abs(r[2])):
    print(f"    {c:<30} raw_AUC={a:.5f}  corr_with_residual={corr:+.5f}")
