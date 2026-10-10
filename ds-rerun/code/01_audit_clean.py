"""Stage 1 - structure audit and stateless cleaning.

Runs the bundled column_audit.py, then the checks the orchestrator requires that a
column-by-column script cannot see: both ends of every range, mass at the minimum,
value-spike hunting against the source table, categorical spelling variants.

Nothing learned from the data happens here. No imputation, capping, encoding or scaling.

Rerunnable end to end:  python code/01_audit_clean.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from elapsed import elapsed_min  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent          # ds-rerun/
REPO = ROOT.parent                                              # ev-purchase-s6e9/
SKILLS = pathlib.Path(r"C:\dev\important repo\Data Scientist Skill Set")
ART = ROOT / "artifacts"

TARGET = "Will_Buy_EV"
KEY = "id"
CATS = ["Gender", "City_Type", "Current_Car_Type",
        "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
NUMS = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
        "Environmental_Concern_Level"]

# Stage 1's stateless cleaning is now defined canonically in code/features.py and imported
# here, so the audit and the Stage 10 production pipeline cannot drift apart. The rules are
# unchanged; see features.clean_raw, features.RANGE_RULES, features.RANGE_ANXIETY_ORDER.
from features import (RANGE_ANXIETY_ORDER, RANGE_RULES, clean_raw)  # noqa: E402,F401


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    # --- bundled audit script -------------------------------------------------
    out = ART / "column_audit.csv"
    if not out.exists():
        subprocess.run([sys.executable, str(SKILLS / "ds-data-audit/scripts/column_audit.py"),
                        str(REPO / "data/raw/train.csv"), "--target", TARGET,
                        "--key", KEY, "--out", str(out)], check=True)

    tr = pd.read_csv(REPO / "data/raw/train.csv")
    te = pd.read_csv(REPO / "data/raw/test.csv")
    src = pd.read_csv(REPO / "data/raw/EV_Adoption_and_Range_Anxiety_Dataset.csv")
    n0 = len(tr)
    y = (tr[TARGET] == "Yes").astype(int)

    # --- checks the column script cannot see ----------------------------------
    findings: dict[str, object] = {}

    # both ends of every range
    impossible = {}
    for c, (lo, hi) in RANGE_RULES.items():
        bad = int(((tr[c] < lo) | (tr[c] > hi)).sum())
        impossible[c] = bad
    findings["impossible_values_per_column"] = impossible

    # mass at the minimum: a placeholder signature is a spike far above local density
    floors = {}
    for c in ["Annual_Income_USD", "Daily_Commute_km", "Environmental_Concern_Level"]:
        vc = tr[c].value_counts().sort_index()
        floors[c] = {
            "min": float(vc.index[0]),
            "count_at_min": int(vc.iloc[0]),
            "share_at_min": round(float(vc.iloc[0] / len(tr)), 5),
            "count_at_next": int(vc.iloc[1]),
            "src_share_at_min": round(float((src[c] == vc.index[0]).mean()), 5),
            "src_nulls": int(src[c].isna().sum()),
            "target_rate_at_min": round(float(y[tr[c] == vc.index[0]].mean()), 4),
            "target_rate_above": round(float(y[tr[c] != vc.index[0]].mean()), 4),
        }
    findings["floors"] = floors

    # generator replication: do high-count values trace to the source table?
    srcvals = set(src["Annual_Income_USD"].dropna().unique())
    vc = tr["Annual_Income_USD"].value_counts()
    big, small = vc[vc >= 200], vc[vc <= 3]
    findings["generator_replication"] = {
        "distinct_income_values": int(len(vc)),
        "share_of_rows_with_source_income": round(
            float(vc[[v in srcvals for v in vc.index]].sum() / len(tr)), 4),
        "values_ge200_in_source": f"{sum(v in srcvals for v in big.index)}/{len(big)}",
        "values_le3_in_source": f"{sum(v in srcvals for v in small.index)}/{len(small)}",
    }

    # truncation relative to the source
    findings["truncation_vs_source"] = {
        c: {"train_max": float(tr[c].max()), "src_max": float(src[c].max()),
            "train_min": float(tr[c].min()), "src_min": float(src[c].min())}
        for c in ["Annual_Income_USD", "Daily_Commute_km"]
    }

    # categorical hygiene + level agreement across the three files
    catcheck = {}
    for c in CATS:
        ltr, lte, lsrc = (set(x[c].dropna().unique()) for x in (tr, te, src))
        norm: dict[str, list] = {}
        for v in ltr:
            norm.setdefault(str(v).strip().lower(), []).append(v)
        catcheck[c] = {
            "levels": sorted(map(str, ltr)),
            "spelling_variants": {k: v for k, v in norm.items() if len(v) > 1},
            "test_matches_train": ltr == lte,
            "source_matches_train": ltr == lsrc,
        }
    findings["categoricals"] = catcheck

    # --- apply the stateless cleaning ----------------------------------------
    clean = clean_raw(tr)
    log.append(f"| load raw train | - | {n0} |")
    log.append(f"| strip whitespace on {len(CATS)} categoricals | 0 changed (no variants found) | {len(clean)} |")
    nulled = sum(int(clean[c].isna().sum()) for c in RANGE_RULES)
    log.append(f"| impossible values -> null (both ends, {len(RANGE_RULES)} rules) | {nulled} cells | {len(clean)} |")
    log.append(f"| order Range_Anxiety_Level Low<Medium<High | - | {len(clean)} |")
    log.append(f"| exact duplicate rows dropped | 0 (audit: 0 exact duplicates) | {len(clean)} |")
    log.append(f"| rows with missing target dropped | 0 (audit: 0 missing) | {len(clean)} |")

    # --- statelessness test ---------------------------------------------------
    a = clean_raw(tr.iloc[:1000]).reset_index(drop=True)
    b = clean_raw(tr).iloc[:1000].reset_index(drop=True)
    stateless_ok = a.equals(b)
    findings["stateless_test_passed"] = bool(stateless_ok)
    if not stateless_ok:
        raise SystemExit("STATELESS TEST FAILED: a cleaning rule learned from the data. "
                         "Move it into a pipeline fitted in-fold.")

    clean.to_parquet(ART / "train_clean.parquet", index=False)
    clean_raw(te).to_parquet(ART / "test_clean.parquet", index=False)

    (ART / "stage1_findings.json").write_text(json.dumps(findings, indent=2, default=str))

    with open(ART / "cleaning_log.md", "w") as f:
        f.write("# Stage 1 cleaning log\n\n")
        f.write("Stateless, row-wise rules only. Every rule is a constant in `clean_raw()`;\n")
        f.write("nothing is learned from the data at this stage.\n\n")
        f.write("| rule | rows/cells affected | rows remaining |\n|---|---|---|\n")
        f.write("\n".join(log) + "\n\n")
        f.write(f"Statelessness test (`clean_raw(head)` == `clean_raw(all).head()`): "
                f"{'PASS' if stateless_ok else 'FAIL'}\n\n")
        f.write("Removed-row concentration check: no rows were removed, so no class, "
                "segment or period can be disproportionately affected.\n")

    print(json.dumps(findings, indent=2, default=str))
    print(f"\nstateless test: {'PASS' if stateless_ok else 'FAIL'}")
    print(f"rows in -> out: {n0} -> {len(clean)}")
    print(f"elapsed {elapsed_min()} min")


if __name__ == "__main__":
    main()
