"""Stage 8b - break the finalist tie honestly, BEFORE the test set is touched.

The Stage 8 search returned six finalists spanning 0.0003 in CV ROC-AUC. Every one is flagged
`within_noise_floor` against the operative 0.00050 sampling floor, so by the skill's own rule
("differences within the noise are ties") they are all tied. The bundled `keep` flag singles out
one of them only because the Nadeau-Bengio paired SE rounds to 0.0000 here - the same degenerate
statistic documented at Stage 2, where repeats over fixed rows collapse the spread.

Protocol, fixed before running and before the test set is read:
  1. Take the best XGBoost finalist and the best LightGBM finalist - the two genuinely different
     designs. The other four are near-duplicates of these.
  2. Refit each on the train part only and score the LOCKED HOLDOUT.
  3. If the holdout agrees with the CV ordering, take the CV winner: two independent measurements
     concurring is enough.
  4. If they disagree, the candidates are tied on both and the skill's tie-break applies:
     simpler family, fewer active steps, options listed earlier.

The holdout is used here to CONFIRM, not to rank. With ~17,500 positives its own SE is about
0.0013, four times the spread being examined, so a six-way selection on it would be selection on
noise. Two candidates, one comparison, decision rule written down in advance.

Rerunnable:  python code/08b_tiebreak.py
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SKILLS = pathlib.Path(r"C:\dev\important repo\Data Scientist Skill Set")
for p in ("ds-validation-design/scripts", "ds-method-comparison/scripts"):
    sys.path.insert(0, str(SKILLS / p))
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cv_folds import load_split   # noqa: E402
from elapsed import elapsed_min   # noqa: E402
from features import CATS         # noqa: E402
from screen import fresh_copy     # noqa: E402

ROOT = HERE.parent
ART = ROOT / "artifacts"

sys.argv = ["08b"]
_spec = importlib.util.spec_from_file_location("js8", HERE / "08_joint_search.py")
js8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(js8)

COMPLEXITY = {"linear": 0, "gbdt": 2, "xgb": 3}


def main() -> None:
    trials = pd.read_csv(ART / "search" / "trials.csv")
    fin = pd.read_csv(ART / "search" / "finalists.csv")
    done = trials[trials["state"] == "COMPLETE"]

    # best completed trial per family, from the search's own record
    cands = {}
    for fam in ["xgb", "gbdt"]:
        sub = done[done["params_family"] == fam]
        if len(sub) == 0:
            continue
        row = sub.loc[sub["value"].idxmax()]
        num = int(row["number"])
        name = next((n for n in fin["name"] if n.startswith(f"T{num}:")), f"T{num}:{fam}")
        params = {c[len("params_"):]: row[c] for c in trials.columns
                  if c.startswith("params_") and pd.notna(row[c])}
        choices = {k.split("|")[0]: v for k, v in params.items() if "|" in k}
        cands[fam] = {"trial": num, "name": name, "cv": float(row["value"]),
                      "params": params, "choices": choices}

    tr = pd.read_parquet(ART / "train_clean.parquet")
    s = load_split(str(ART / "split" / "split.csv"), ids=tr["id"])
    feats = js8.NUM + js8.CAT
    X = tr[feats].copy()
    for c in CATS:
        X[c] = X[c].astype(str)
    y = (tr["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
    Xtr, ytr = X[s.train_mask], y[s.train_mask]
    Xho, yho = X[s.holdout_mask], y[s.holdout_mask]

    for fam, c in cands.items():
        pipe = js8.build(fam, c["choices"], js8.FAMILIES[fam](js8.fixed_trial(c["params"])))
        t0 = time.time()
        fitted = fresh_copy(pipe).fit(Xtr, ytr)
        c["holdout"] = float(roc_auc_score(yho, fitted.predict_proba(Xho)[:, 1]))
        c["fit_seconds"] = round(time.time() - t0, 1)
        print(f"{fam:5s} T{c['trial']:<3d} cv {c['cv']:.6f}  holdout {c['holdout']:.6f}  "
              f"({c['fit_seconds']}s)  choices={c['choices']}")

    cv_best = max(cands, key=lambda f: cands[f]["cv"])
    ho_best = max(cands, key=lambda f: cands[f]["holdout"])
    agree = cv_best == ho_best

    if agree:
        pick_fam, rule = cv_best, ("CV and the locked holdout agree on the same family; two "
                                   "independent measurements concurring is sufficient")
    else:
        pick_fam = min(cands, key=lambda f: COMPLEXITY[f])
        rule = ("CV and the holdout disagree, so the candidates are tied on both. The skill's "
                "tie-break applies: simpler and cheaper family wins")

    out = {
        "finalist_spread_cv": round(float(fin["mean"].max() - fin["mean"].min()), 6),
        "all_within_noise_floor": bool(fin["within_noise_floor"].all()),
        "operative_noise_floor": 0.00050,
        "holdout_se_hanley_mcneil_approx": 0.0013,
        "candidates": {f: {k: v for k, v in c.items() if k != "params"}
                       for f, c in cands.items()},
        "cv_best_family": cv_best, "holdout_best_family": ho_best, "agree": agree,
        "decision_rule": rule, "picked_family": pick_fam,
        "picked_trial": cands[pick_fam]["trial"], "picked_name": cands[pick_fam]["name"],
        "picked_choices": cands[pick_fam]["choices"],
    }
    (ART / "tiebreak.json").write_text(json.dumps(out, indent=2, default=str))

    # rewrite result.json so Stage 9 and 10 consume the confirmed pick
    res = json.loads((ART / "search" / "result.json").read_text())
    res["pick"] = cands[pick_fam]["name"]
    res["pick_params"] = cands[pick_fam]["params"]
    res["pick_choices"] = cands[pick_fam]["choices"]
    res["cv_mean_pick"] = cands[pick_fam]["cv"]
    res["holdout"] = {f: c["holdout"] for f, c in cands.items()}
    res["tiebreak"] = {"rule": rule, "agree": agree}
    (ART / "search" / "result.json").write_text(json.dumps(res, indent=2, default=str))

    print(f"\nagree={agree}  ->  {rule}")
    print(f"FINAL PICK: {out['picked_name']}  choices={out['picked_choices']}")
    print(f"elapsed {elapsed_min()} min")


if __name__ == "__main__":
    main()
