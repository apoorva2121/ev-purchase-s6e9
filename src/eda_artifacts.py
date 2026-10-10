"""Test the synthetic-generator artifact hypotheses on S6E9 before any modelling.

The premise (from public S6E9 writeups): the synthesizer over-produces certain
exact values it memorised from the original dataset, and the label rate at those
values deviates from the smooth underlying formula. If true, value-frequency and
"roundness" features carry signal that the semantic features do not.
"""
import numpy as np, pandas as pd

RAW = "C:/dev/important repo/ml-bench/ev-purchase-s6e9/data/raw"
tr = pd.read_csv(f"{RAW}/train.csv")
te = pd.read_csv(f"{RAW}/test.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int)
feats = [c for c in tr.columns if c not in ("id", "Will_Buy_EV")]

print(f"train {tr.shape}  test {te.shape}  pos_rate {y.mean():.6f}")
print(f"nulls train={tr.isna().sum().sum()} test={te.isna().sum().sum()}")

print("\n=== CARDINALITY / DTYPE ===")
for c in feats:
    print(f"{c:<30} {str(tr[c].dtype):<8} nuniq_tr={tr[c].nunique():>7} nuniq_te={te[c].nunique():>7}")

print("\n=== NON-INTEGER VALUES IN NOMINALLY-INTEGER COLS ===")
for c in ["Environmental_Concern_Level", "Annual_Income_USD", "Daily_Commute_km"]:
    if pd.api.types.is_numeric_dtype(tr[c]):
        frac = (tr[c] % 1 != 0).mean()
        print(f"{c:<30} frac_non_integer={frac:.6f}  min={tr[c].min()} max={tr[c].max()}")
        if 0 < frac < 1:
            m = tr[c] % 1 != 0
            print(f"    -> label rate  non-int={y[m].mean():.5f}  int={y[~m].mean():.5f}  (base {y.mean():.5f})")

print("\n=== OVER-PRODUCED VALUES (generator memorisation) ===")
for c in ["Annual_Income_USD", "Daily_Commute_km", "Age"]:
    vc = tr[c].value_counts()
    exp = len(tr) / tr[c].nunique()
    over = vc[vc > 20 * exp]
    print(f"\n{c}: {tr[c].nunique()} distinct, mean count {exp:.1f}, "
          f"{len(over)} values appear >20x expected (covering {vc[vc>20*exp].sum()/len(tr):.2%} of rows)")
    print(f"  top 8 values: {list(vc.head(8).items())}")
    if len(over):
        m = tr[c].isin(over.index)
        print(f"  -> label rate ON over-produced={y[m].mean():.5f}  OFF={y[~m].mean():.5f}  (base {y.mean():.5f})")

print("\n=== ROUNDNESS / DIGIT ARTIFACTS on Annual_Income_USD ===")
inc = tr["Annual_Income_USD"]
for mod, name in [(1000, "round to 1000"), (500, "round to 500"), (100, "round to 100")]:
    m = (inc % mod == 0)
    print(f"  {name:<16} frac={m.mean():.4f}  label_rate={y[m].mean():.5f} vs {y[~m].mean():.5f}")
ndig = inc.astype(np.int64).astype(str).str.len()
print("  digit-count label rates:")
for d, g in y.groupby(ndig):
    print(f"    {d} digits: n={len(g):>7} label_rate={g.mean():.5f}")

print("\n=== DUPLICATES ===")
print(f"  exact dup rows in train (feats only): {tr.duplicated(subset=feats).sum()}")
print(f"  exact dup rows in test  (feats only): {te.duplicated(subset=feats).sum()}")
key_tr = tr[feats].astype(str).agg("|".join, axis=1)
key_te = te[feats].astype(str).agg("|".join, axis=1)
overlap = set(key_tr) & set(key_te)
print(f"  distinct feature-rows shared train<->test: {len(overlap)}")
if overlap:
    m = key_tr.isin(overlap)
    print(f"  -> {m.sum()} train rows ({m.mean():.2%}) share a feature-row with test; "
          f"label rate {y[m].mean():.5f} vs {y[~m].mean():.5f}")

print("\n=== CATEGORICAL LEVELS ===")
for c in feats:
    if tr[c].dtype == object:
        print(f"{c:<30} {sorted(tr[c].unique().tolist())}")
