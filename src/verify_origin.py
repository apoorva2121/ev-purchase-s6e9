"""Verify the load-bearing claims: income-value copying from the 10k original,
and the standalone power of the %1000 remainder."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

RAW = "C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
tr = pd.read_csv(f"{RAW}/train.csv"); te = pd.read_csv(f"{RAW}/test.csv")
og = pd.read_csv(f"{RAW}/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
print(f"original {og.shape}  pos_rate {(og.Will_Buy_EV=='Yes').mean():.5f} vs comp {y.mean():.5f}")

# --- CLAIM A: train incomes are exact copies of original income values ---
og_inc = set(og.Annual_Income_USD.unique())
in_og = tr.Annual_Income_USD.isin(og_inc)
print(f"\n[A] original distinct incomes: {len(og_inc)}")
print(f"    train rows whose income appears in original: {in_og.mean():.4%}")
print(f"    train distinct incomes: {tr.Annual_Income_USD.nunique()}, "
      f"of which in original: {tr.Annual_Income_USD.isin(og_inc).groupby(tr.Annual_Income_USD).any().sum()}")
# chance baseline: if incomes were drawn freely over the observed integer range
rng = tr.Annual_Income_USD.max() - tr.Annual_Income_USD.min()
print(f"    chance if uniform over range: {len(og_inc)/rng:.4%}")
print(f"    -> label rate  in_original={y[in_og].mean():.5f}  novel={y[~in_og].mean():.5f}  base={y.mean():.5f}")
print(f"    novel rows: {(~in_og).sum()} ({(~in_og).mean():.2%})")

te_in_og = te.Annual_Income_USD.isin(og_inc)
print(f"    test rows in original: {te_in_og.mean():.4%}  (consistency check)")

# --- CLAIM B: income % 1000 remainder is strong on its own ---
print("\n[B] standalone AUC of single engineered features")
inc = tr.Annual_Income_USD.values
cands = {
    "income (raw)":            inc,
    "income %1000 remainder":  inc % 1000,
    "income %100  remainder":  inc % 100,
    "income %10   remainder":  inc % 10,
    "is_income_floor(30000)":  (inc == 30000).astype(float),
    "income_in_original":      in_og.values.astype(float),
    "commute (raw)":           tr.Daily_Commute_km.values,
}
for name, v in cands.items():
    a = roc_auc_score(y, v)
    print(f"    {name:<26} AUC={a:.5f}  (|.5-a|={abs(a-.5):.5f})")

# Is the %1000 signal just the floor? Recheck excluding floor rows.
m = inc != 30000
print(f"\n    EXCLUDING the 30000 floor ({m.sum()} rows):")
for name, v in [("income %1000", inc % 1000), ("income raw", inc)]:
    print(f"      {name:<16} AUC={roc_auc_score(y[m], v[m]):.5f}")

# --- CLAIM C: does the original's label, looked up by exact income, carry signal? ---
og_rate = og.groupby("Annual_Income_USD").Will_Buy_EV.apply(lambda s: (s=="Yes").mean())
mapped = tr.Annual_Income_USD.map(og_rate)
mm = mapped.notna()
print(f"\n[C] original label-rate looked up by exact income: covers {mm.mean():.2%} of train")
print(f"    AUC of that lookup alone (on covered rows): {roc_auc_score(y[mm], mapped[mm]):.5f}")
