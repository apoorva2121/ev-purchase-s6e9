"""How much headroom is left? Oracle bounds on the only channel that has ever paid.

DIAGNOSTIC ONLY, and it uses all 668,665 rows INCLUDING the frozen test rows. Every oracle below is deliberately leaky - it uses each row's own label - so
none of it is submittable. The point is to bound what is achievable, not to achieve it.

Two known reference points:
  - semantic ceiling, from the prior work's recovered probit (sigma = 0.851):  0.93689
  - this run's grouped diagnostic (no shared income values between folds):     0.936222
Everything above ~0.937 is recovery of source-table identity. This script asks how much of that
channel is left un-extracted.

Oracles, each an UPPER bound that no honest model can reach:
  O1  per-exact-income target rate, computed on all labels including the row's own
  O2  O1 + per-exact-commute
  O3  per-exact (income, commute) PAIR
  O4  per-exact FULL feature vector (the absolute ceiling of memorisation)
  O5  the honest model's own predictions, for reference

Rerunnable:  python code/12_ceiling.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from elapsed import elapsed_min   # noqa: E402

ROOT = HERE.parent
REPO = ROOT.parent
ART = ROOT / "artifacts"


def oracle_rate(keys: pd.Series, y: np.ndarray, m: float = 0.0) -> np.ndarray:
    """Per-key target rate using ALL labels, the row's own included. Leaky by design."""
    df = pd.DataFrame({"k": keys.to_numpy(), "y": y})
    g = df.groupby("k")["y"].agg(["sum", "count"])
    prior = y.mean()
    rate = (g["sum"] + prior * m) / (g["count"] + m)
    return df["k"].map(rate).to_numpy()


def main() -> None:
    tr = pd.read_csv(REPO / "data/raw/train.csv")
    y = (tr["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
    feats = [c for c in tr.columns if c not in ("id", "Will_Buy_EV")]

    inc = tr["Annual_Income_USD"]
    com = tr["Daily_Commute_km"]
    pair = inc.astype(str) + "|" + com.astype(str)
    full = tr[feats].astype(str).agg("|".join, axis=1)

    out: dict[str, object] = {}

    out["O1_income_exact_oracle"] = float(roc_auc_score(y, oracle_rate(inc, y)))
    o2 = oracle_rate(inc, y) + oracle_rate(com, y)
    out["O2_income_plus_commute_oracle"] = float(roc_auc_score(y, o2))
    out["O3_income_commute_pair_oracle"] = float(roc_auc_score(y, oracle_rate(pair, y)))
    out["O4_full_feature_vector_oracle"] = float(roc_auc_score(y, oracle_rate(full, y)))

    # group-size context: an oracle on a key with one row per group is just memorising y
    for name, k in [("income", inc), ("pair", pair), ("full_vector", full)]:
        vc = k.value_counts()
        out[f"keysize_{name}"] = {
            "n_keys": int(len(vc)),
            "median_rows_per_key": float(vc.median()),
            "share_of_rows_in_singleton_keys": round(float(vc[vc == 1].sum() / len(k)), 4),
        }

    # the honest model, for reference
    pm = ART / "postmortem_cache" / "oof_G_both_ladders_all.npy"
    if pm.exists():
        out["honest_best_oof_G"] = float(roc_auc_score(y, np.load(pm)))
    pa = ART / "postmortem_cache" / "oof_A_this_run_final.npy"
    if pa.exists():
        out["honest_oof_A"] = float(roc_auc_score(y, np.load(pa)))

    out["reference_semantic_ceiling_prior_probit"] = 0.93689
    out["reference_grouped_diagnostic_this_run"] = 0.936222
    out["reference_prior_best_lb"] = 0.94615
    out["reference_this_run_lb"] = 0.94583
    out["reference_realistic_field_top"] = 0.94674
    out["target_asked_about"] = 0.95

    (ART / "ceiling.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nelapsed {elapsed_min()} min")


if __name__ == "__main__":
    main()
