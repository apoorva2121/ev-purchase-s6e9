"""Stage 8 joint pipeline search over the choices carried forward from Stages 4-7.

TEMPLATE: the search, selection and holdout logic is generic. Adapt the ADAPT sections (column
lists, slot builders, model families) to the project, then run:

    python joint_search.py --data cleaned.csv --split split/ --space search_space.json \
        --id <id column> --target <target> --trials 60 --timeout 900 --top 5 --seeds 3 --out search/

What it does
  1. Every carried-forward option becomes a categorical choice, conditional on the model family
     (an option is only offered when the family is in its `applies_to`). Parameter names are
     "<slot>|<family>" so each has fixed choices, which Optuna requires.
  2. Family hyperparameters are searched alongside. TPE with multivariate=True, group=True handles
     the conditional space; MedianPruner stops weak trials after a few folds.
  3. The top K distinct pipelines (family + slot choices, each at its best-tuned trial) are re-scored
     with several seeds on the same folds.
  4. One-standard-error rule: among configurations within one corrected SE (or the noise floor) of
     the best, pick the simplest (family order, then fewest active steps).
  5. The pick and the best-by-mean are refitted on the train part and scored on the locked holdout.
Outputs: search/trials.csv, search/finalists.csv, search/result.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

HERE = Path(__file__).resolve().parent


def _skills_root() -> Path:
    """Find the skill set even when this template was copied into a project folder.

    Order: DS_SKILLS_ROOT environment variable, the folder two levels up (running in place),
    then the default Claude Code install location ~/.claude/skills.
    """
    candidates = [os.environ.get("DS_SKILLS_ROOT", ""), str(HERE.parents[1]),
                  str(Path.home() / ".claude" / "skills")]
    for c in candidates:
        if c and (Path(c) / "ds-validation-design" / "scripts" / "cv_folds.py").exists():
            return Path(c)
    raise SystemExit("Cannot find the skill set; set DS_SKILLS_ROOT to the folder holding ds-validation-design/")


ROOT = _skills_root()
sys.path.insert(0, str(ROOT / "ds-method-comparison" / "scripts"))
sys.path.insert(0, str(ROOT / "ds-validation-design" / "scripts"))
from cv_folds import load_split  # noqa: E402
from screen import evaluate, fresh_copy, summarize, topk_scorer  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================ ADAPT: columns
NUM = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
       "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
       "Environmental_Concern_Level"]
CAT = ["Gender", "City_Type", "Current_Car_Type",
       "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
SKEWED = []   # Stage 5 measured |skew| <= 0.70 on every numeric and 0 rows beyond 3 IQR:
              # nothing here is heavily skewed, so the log1p option would be a no-op.

sys.path.insert(0, str(HERE))
from features import MultiTE, FloorIndicator, FloorsToNaN, ladder_specs  # noqa: E402

from features import _com, floor_div  # noqa: E402

LADDER3 = ladder_specs((50, 500))          # Stage 6b winner: exact + //50 + //500
# Stage 6c: adding a commute ladder beat ladder3 by +0.000576 on the LINEAR proxy (clears the
# 0.00050 noise floor) but only +0.000119 on gbdt (a tie). A tree can build the commute shape
# itself; a linear model cannot. Both are carried so the joint search decides per family.
LADDER3_COM = LADDER3 + [("com_exact", _com),
                         ("com_1", floor_div(_com, 1)),
                         ("com_5", floor_div(_com, 5))]
TE_COLS = MultiTE(LADDER3).new_cols()
TE_COLS_COM = MultiTE(LADDER3_COM).new_cols()
FLOOR_COLS = FloorIndicator().new_cols()


# ============================================================ ADAPT: slot builders
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learned on the training fold (so it is fitted in-fold)."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_, self.hi_ = np.nanquantile(X, self.lower, axis=0), np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def _log1p_skewed(X):
    X = pd.DataFrame(X, columns=NUM).copy()
    for c in SKEWED:
        X[c] = np.log1p(X[c].clip(lower=0))
    return X.to_numpy()


SLOTS = {  # slot -> option name -> builder(family). List options SIMPLEST FIRST: order breaks ties.
    "outliers": {
        "none": lambda fam: "passthrough",
        "winsorize_p01_p99": lambda fam: Winsorizer(0.01, 0.99),
    },
    "missing_values": {
        # This data has ZERO missing values. The slot instead carries the Stage 4 question of
        # whether the measured censoring floors become NaN. "native_nan" is the no-op reference.
        "native_nan": lambda fam: "passthrough",
        "median_indicator": lambda fam: SimpleImputer(strategy="median", add_indicator=True),
    },
    "encoding": {
        "onehot": lambda fam: OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                            min_frequency=20),
        "target": lambda fam: TargetEncoder(random_state=0),
    },
}

# Families whose estimator accepts NaN directly. "floors: to_nan" creates NaNs, so it may only
# be offered to these; a linear model would raise. Caught by the Stage 8 smoke test on a 30k
# subsample before the real search was launched.
NATIVE_NAN_FAMILIES = {"gbdt", "xgb", "cat"}

# Slots applied to the WHOLE frame before the ColumnTransformer, because they add columns.
# Listed simplest first, same tie-break convention as SLOTS.
FRAME_SLOTS = {
    "income_te": {
        "none": lambda: None,
        "ladder3": lambda: MultiTE(LADDER3),
        "ladder3_commute": lambda: MultiTE(LADDER3_COM),
    },
    "floors": {
        "keep": lambda: None,
        "indicator": lambda: FloorIndicator(),
        "to_nan": lambda: FloorsToNaN(),
    },
}


# ============================================================ ADAPT: model families
def _linear(trial):
    return LogisticRegression(C=trial.suggest_float("linear_C", 1e-3, 10, log=True), max_iter=3000)


def _gbdt(trial):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=trial.suggest_int("gbdt_n_estimators", 100, 600, step=50),
        learning_rate=trial.suggest_float("gbdt_lr", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("gbdt_num_leaves", 4, 63, log=True),
        min_child_samples=trial.suggest_int("gbdt_min_child", 10, 200, log=True),
        subsample=trial.suggest_float("gbdt_subsample", 0.5, 1.0), subsample_freq=1,
        colsample_bytree=trial.suggest_float("gbdt_colsample", 0.4, 1.0),
        reg_lambda=trial.suggest_float("gbdt_lambda", 1e-3, 30, log=True),
        verbose=-1, n_jobs=2, random_state=0)


def _xgb(trial):
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=trial.suggest_int("xgb_n_estimators", 200, 900, step=100),
        learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("xgb_max_depth", 3, 10),
        min_child_weight=trial.suggest_float("xgb_min_child_weight", 1e-2, 50, log=True),
        subsample=trial.suggest_float("xgb_subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("xgb_colsample", 0.4, 1.0),
        reg_lambda=trial.suggest_float("xgb_lambda", 1e-3, 30, log=True),
        tree_method="hist", eval_metric="auc", n_jobs=4, random_state=0)


def _cat(trial):
    import catboost as cb
    return cb.CatBoostClassifier(
        iterations=trial.suggest_int("cat_iterations", 300, 1200, step=100),
        learning_rate=trial.suggest_float("cat_lr", 0.01, 0.2, log=True),
        depth=trial.suggest_int("cat_depth", 4, 9),
        l2_leaf_reg=trial.suggest_float("cat_l2", 1.0, 30.0, log=True),
        random_seed=0, verbose=0, allow_writing_files=False, thread_count=4)


# CatBoost is DROPPED from the search, on measured cost. Stage 7: 0.944548 vs LightGBM's
# 0.944667 - a deficit of 0.000119, four times below the 0.00050 noise floor, i.e. a tie - for
# 1797s against LightGBM's 27.6s, a 65x cost. The skill's tie rule sends ties to the cheaper
# option. Its factory is kept below so the decision can be reversed by editing one line.
FAMILIES = {"linear": _linear, "gbdt": _gbdt, "xgb": _xgb}
COMPLEXITY = {"linear": 0, "rf": 1, "gbdt": 2, "xgb": 3, "cat": 4}
NEEDS_SCALING = {"linear"}


# ============================================================ generic from here on
def build(fam: str, choices: dict, model) -> Pipeline:
    """Assemble the pipeline for one trial.

    Frame-level steps run FIRST and add columns, so the ColumnTransformer's numeric list has to
    grow to match. Order within them matters: the income target encoder is applied before any
    floor treatment, so it keys on the real censored value (income 30000 is 61,605 rows at a
    4.43% target rate - a genuinely informative category) rather than on a NaN.

    Categoricals go through the encoding slot for every family, including XGBoost and CatBoost.
    That keeps one code path and one comparison; the alternative (native categorical handling per
    library) would make the families differ by more than the thing being compared.
    """
    frame_steps, extra_num = [], []

    te_choice = choices.get("income_te", "none")
    te = FRAME_SLOTS["income_te"][te_choice]()
    if te is not None:
        frame_steps.append(("income_te", te))
        extra_num += TE_COLS_COM if te_choice == "ladder3_commute" else TE_COLS

    fl = FRAME_SLOTS["floors"][choices.get("floors", "keep")]()
    if fl is not None:
        frame_steps.append(("floors", fl))
        if isinstance(fl, FloorIndicator):
            extra_num += FLOOR_COLS

    num_cols = NUM + extra_num
    num_steps = [SLOTS["outliers"][choices.get("outliers", "none")](fam),
                 SLOTS["missing_values"][choices.get("missing_values", "native_nan")](fam)]
    if fam in NEEDS_SCALING:
        num_steps.append(StandardScaler())
    num_steps = [s for s in num_steps if s != "passthrough"] or ["passthrough"]
    num_pipe = num_steps[0] if len(num_steps) == 1 else make_pipeline(*num_steps)
    enc = SLOTS["encoding"][choices.get("encoding", "onehot")](fam)
    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", enc, CAT)])
    return Pipeline([*frame_steps, ("pre", pre), ("model", model)])


def fixed_trial(params: dict):
    """Rebuild a model from stored trial params (FixedTrial replays suggest_* calls)."""
    return optuna.trial.FixedTrial(params)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--space", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="search")
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    s = load_split(Path(a.split) / "split.csv", ids=df[a.id])
    space = json.loads(Path(a.space).read_text())
    metric = space["primary_metric"]
    scorer = topk_scorer(float(metric.split("@")[1])) if metric.startswith("precision@") else get_scorer(metric)
    noise = space.get("noise_floor_sd")
    def _encode_y(v):
        """XGBoost requires numeric class labels; LogisticRegression, LightGBM and CatBoost
        tolerate the raw 'No'/'Yes' strings. Encode once, here, so every family sees the same
        y and the comparison is not confounded by label representation.
        Caught by the Stage 8 smoke test on a 30k subsample."""
        v = np.asarray(v)
        if v.dtype.kind in "OUS":
            classes = np.array(sorted(set(v.tolist())))
            if set(classes.tolist()) == {"No", "Yes"}:
                classes = np.array(["No", "Yes"])          # keep "Yes" as the positive class
            return np.searchsorted(classes, v) if len(classes) != 2 else (v == classes[1]).astype(int)
        return v

    tr = df[s.train_mask].reset_index(drop=True)
    X, y = tr[NUM + CAT], _encode_y(tr[a.target].to_numpy())
    families = [f for f in FAMILIES
                if any(f in o["applies_to"] for opts in space["stages"].values() for o in opts)]
    families = families or list(FAMILIES)

    def options(slot, fam):
        return [o["name"] for o in space["stages"].get(slot, []) if fam in o["applies_to"] and o["name"] in SLOTS[slot]]

    # ---- validate the space BEFORE spending compute on it -------------------
    # Two silent filters shrink the space without any error: an option name that is not a key of
    # SLOTS[slot], and an applies_to list that omits a family. Either one makes the slot default
    # quietly, and the search then reports a clean result over a space much smaller than designed.
    problems = []
    for slot in SLOTS:
        entries = space["stages"].get(slot, [])
        if not entries:
            problems.append(f"slot '{slot}': no entries in search_space.json -> always defaults to "
                            f"'{next(iter(SLOTS[slot]))}'")
            continue
        unknown = [o["name"] for o in entries if o["name"] not in SLOTS[slot]]
        if unknown:
            problems.append(f"slot '{slot}': names not in SLOTS[{slot}] and therefore IGNORED: "
                            f"{unknown}  (valid: {list(SLOTS[slot])})")
        for fam in families:
            if not options(slot, fam):
                problems.append(f"slot '{slot}' x family '{fam}': ZERO options -> silently defaults")
    for slot in space["stages"]:
        if slot not in SLOTS and slot not in ("models", "imbalance", "features"):
            problems.append(f"stage '{slot}' in search_space.json matches no SLOT and is ignored")
    # a family defined in FAMILIES but absent from every applies_to is silently never searched
    dropped = [f for f in FAMILIES if f not in families]
    if dropped:
        problems.append(f"families defined but in NO applies_to list, so never searched: {dropped}")
    # a slot offering exactly one option to a family is not being searched for that family
    for slot in SLOTS:
        for fam in families:
            opts = options(slot, fam)
            if len(opts) == 1:
                problems.append(f"slot '{slot}' x family '{fam}': only one option {opts} -> not searched, fixed")
    if problems:
        print("=" * 72)
        print("SEARCH SPACE WARNINGS - the space actually searched is smaller than the one written")
        for w in problems:
            print("  ! " + w)
        print("=" * 72, flush=True)
    else:
        print("search space validated: every slot offers options to every family", flush=True)
    print("effective space:")
    for fam in families:
        print(f"  {fam:8s} " + ", ".join(f"{s}={options(s, fam) or ['<default>']}" for s in SLOTS),
              flush=True)
    if a.validate_only:
        return

    def objective(trial):
        fam = trial.suggest_categorical("family", families)
        choices = {}
        for slot in SLOTS:
            opts = options(slot, fam)
            if opts:
                choices[slot] = trial.suggest_categorical(f"{slot}|{fam}", opts)
        # Frame slots add columns rather than transform them. They apply to every family,
        # except that "floors: to_nan" is only legal where the estimator accepts NaN.
        for slot, opts in FRAME_SLOTS.items():
            names = list(opts)
            if slot == "floors" and fam not in NATIVE_NAN_FAMILIES:
                names = [n for n in names if n != "to_nan"]
            choices[slot] = trial.suggest_categorical(f"{slot}|{fam}", names)
        pipe = build(fam, choices, FAMILIES[fam](trial))
        scores = []
        for k, (i_tr, i_va) in enumerate(s.folds):
            p = fresh_copy(pipe).fit(X.iloc[i_tr], y[i_tr])
            scores.append(scorer(p, X.iloc[i_va], y[i_va]))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("choices", choices)
        trial.set_user_attr("fold_scores", scores)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, group=True, n_startup_trials=15),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials, timeout=a.timeout)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"{len(study.trials)} trials ({len(done)} complete, "
          f"{sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)} pruned) in {time.time() - t0:.0f}s")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)

    # top K distinct PIPELINES (family + slot choices), each represented by its best-tuned trial,
    # re-scored with several seeds. Distinct pipelines, not distinct hyperparameters, so the
    # one-standard-error rule chooses between genuinely different designs.
    seen, finalists = set(), {}
    for t in sorted(done, key=lambda t: -t.value):
        key = json.dumps({"family": t.params["family"], **t.user_attrs["choices"]}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        fam = t.params["family"]
        finalists[f"T{t.number}:{fam}:" + ",".join(f"{k}={v}" for k, v in t.user_attrs["choices"].items())] = (t, fam)
        if len(finalists) == a.top:
            break
    pipes = {name: build(fam, t.user_attrs["choices"], FAMILIES[fam](fixed_trial(t.params)))
             for name, (t, fam) in finalists.items()}
    long = evaluate(pipes, X, y, folds=s.folds, scoring=scorer, seeds=range(a.seeds), seed_param="model__random_state"
                    if all(hasattr(p.named_steps["model"], "random_state") for p in pipes.values()) else None)
    table = summarize(long, noise_floor_sd=noise)
    active = {n: sum(v not in ("none", "native_nan", "passthrough", "keep")
                     for v in finalists[n][0].user_attrs["choices"].values())
              for n in finalists}
    table["family"] = [finalists[n][1] for n in table["name"]]
    # tie-break: simpler family, then fewer active steps, then options listed earlier in SLOTS (simplest first)
    def _slot_index(slot, v):
        table_ = SLOTS.get(slot) or FRAME_SLOTS.get(slot) or {}
        return list(table_).index(v) if v in table_ else 0

    position = {n: sum(_slot_index(slot, v)
                       for slot, v in finalists[n][0].user_attrs["choices"].items())
                for n in finalists}
    table["complexity"] = [(COMPLEXITY.get(finalists[n][1], 9), active[n], position[n]) for n in table["name"]]
    table.to_csv(out / "finalists.csv", index=False)
    print(table.drop(columns="complexity").to_string(index=False))

    best_by_mean = table.iloc[0]["name"]
    pick = table[table["keep"]].sort_values("complexity").iloc[0]["name"]
    print(f"\nbest by mean: {best_by_mean}\n1-SE pick   : {pick}")

    # locked holdout confirmation
    ho = df[s.holdout_mask]
    Xh, yh = ho[NUM + CAT], _encode_y(ho[a.target].to_numpy())
    holdout = {}
    for name in dict.fromkeys([pick, best_by_mean]):
        p = fresh_copy(pipes[name]).fit(X, y)
        holdout[name] = float(scorer(p, Xh, yh))
        print(f"holdout {metric}: {holdout[name]:.4f}  <- {name}")
    cv_mean = float(table.set_index("name").loc[pick, "mean"])
    result = {"metric": metric, "pick": pick, "best_by_mean": best_by_mean, "pick_params": finalists[pick][0].params,
              "pick_choices": finalists[pick][0].user_attrs["choices"], "cv_mean_pick": cv_mean, "holdout": holdout,
              "noise_floor_sd": noise, "n_trials": len(study.trials)}
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"written: {out / 'result.json'}")


if __name__ == "__main__":
    main()
