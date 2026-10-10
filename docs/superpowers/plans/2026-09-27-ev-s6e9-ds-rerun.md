# EV S6E9 Ten-Stage Rebuild — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the S6E9 EV-purchase model from raw CSVs through the `data-scientist` orchestrator's ten gates, blind to the prior analysis until Stage 9, ending in a once-scored frozen local test result, a comparison against the prior work, and a Kaggle submission chosen on local evidence alone.

**Architecture:** One rerunnable script per stage in `ds-rerun/code/`, each calling the step skills' bundled scripts rather than re-implementing their checks. Stage N writes its gate artifact plus a `decision_log.md` block before Stage N+1 starts. Carry-forward candidates accumulate in `search_space.json` and are searched jointly at Stage 8.

**Tech Stack:** Python 3.13.9 (`C:\Users\apoor\anaconda3\python.exe`), scikit-learn 1.7.2, pandas 2.3.3, numpy 2.3.5, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, optuna 5.0.0, scipy 1.16.3.

**Spec:** `docs/superpowers/specs/2026-09-27-ev-s6e9-ds-rerun-design.md`

## Global Constraints

- Skill set lives at `C:\dev\important repo\Data Scientist Skill Set`; skills are read from there, **not** installed into `~/.claude/skills`.
- **Blind until Stage 9.** Off-limits: `src/`, `artifacts/`, `subs/`, `nb/`, `tests/`, `logs_*.txt`, `reports/02_scoreboard.md`, `reports/03_backlog.md`, `C:\dev\important repo\reports\Kaggle S6E9 EV winning approach.md`, and the Kaggle public leaderboard. Any consult becomes a dated `CONTAMINATION` block in `decision_log.md`.
- **Use the bundled scripts.** If one is replaced, log which of its checks the replacement covers.
- **The frozen test set is scored exactly once, at Stage 9.** `verify_test_untouched("split")` must return True immediately before that scoring.
- **Fitted things are fitted in-fold.** Only stateless row-wise fixes may run before the split.
- **Every technique gets a reason both ways** — applied because (evidence from this data) or skipped because (reason tied to this data). "Not needed" is not a reason.
- **Differences within the noise are ties.** Noise floor = sd of the CV mean, measured in ROC-AUC, recorded in `search_space.json`.
- Elapsed minutes are measured with a clock (`time.time()` against `ds-rerun/artifacts/clock_start.txt`, `START_UTC=2026-09-27T15:41:57Z`) and written into every gate block. Never estimated.
- Full fidelity: **compact mode is not used**. Budget ~4–6h, split ~25% stages 0–3, ~35% stages 4–7, ~25% stage 8, ~15% stages 9–10.
- Nothing is written outside `ds-rerun/` and `docs/superpowers/`.
- Primary metric: **ROC-AUC** (set by the competition). Secondary: calibration, per-slice AUC.

---

### Task 0: Stage 0 — frame the problem

**Files:**
- Create: `ds-rerun/problem_statement.md`
- Create: `ds-rerun/decision_log.md`
- Create: `ds-rerun/report.md` (skeleton)
- Create: `ds-rerun/search_space.json` (empty scaffold)

**Interfaces:**
- Produces: the twelve framing fields the orchestrator requires; `ASSUMED` markers on anything not stated by the user; the stage time budget in minutes.

- [ ] **Step 1: Read the orchestrator's templates**

`cat "C:/dev/important repo/Data Scientist Skill Set/data-scientist/references/templates.md"`

- [ ] **Step 2: Measure the data's shape for the framing table**

Row count, column count, positive rate, dtype per column, memory footprint. Load with `pandas`, print only; no cleaning yet.

- [ ] **Step 3: Write `problem_statement.md`**

Every field from the orchestrator's Stage 0 table: unit of analysis, target, prediction moment, decision fed, problem type, primary metric + reason, secondary metrics, baseline to beat, success threshold, constraints, data size, time and compute budget. Mark assumptions `ASSUMED`.

Answer explicitly: **does this need ML at all?**

- [ ] **Step 4: Write the report skeleton and empty search space**

`report.md` with one heading per stage, each marked `PENDING`. `search_space.json` as `{"noise_floor_sd": null, "stages": {}}`.

- [ ] **Step 5: Write the Stage 0 decision-log block**

Include the blindness protocol as a named constraint and the measured elapsed minutes.

**Gate:** `problem_statement.md` complete, path chosen (tabular binary classification), clock started, report skeleton exists.

---

### Task 1: Stage 1 — structure audit and stateless cleaning

**Files:**
- Create: `ds-rerun/code/01_audit_clean.py`
- Create: `ds-rerun/artifacts/column_audit.csv` (with a `verdict` column)
- Create: `ds-rerun/artifacts/cleaning_log.md`
- Create: `ds-rerun/artifacts/train_clean.csv`

**Interfaces:**
- Consumes: `data/raw/train.csv`, `data/raw/test.csv`.
- Produces: `clean_raw(df) -> df`, the stateless cleaning function reused verbatim as the first step of the Stage 10 production pipeline; candidate key / group / time columns handed to Stage 2.

- [ ] **Step 1: Load the skill**

`cat "C:/dev/important repo/Data Scientist Skill Set/ds-data-audit/SKILL.md"` — Part A only.

- [ ] **Step 2: Run the bundled column audit**

```bash
python "C:/dev/important repo/Data Scientist Skill Set/ds-data-audit/scripts/column_audit.py" \
  data/raw/train.csv --target Will_Buy_EV --key id --out ds-rerun/artifacts/column_audit.csv
```

Expected: a row per column with flags. Reads everything as raw strings, so sentinels survive type inference.

- [ ] **Step 3: Resolve every flag with a verdict**

Add a `verdict` column. Every flagged row gets one, per the skill's flag table. Check **both ends** of every range rule.

Specific to this data — establish independently, by measurement, not by recall:
- Whether `Annual_Income_USD`, `Daily_Commute_km` and `Environmental_Concern_Level` have a spike at their minimum (a mass far above the neighbouring density is a placeholder signature, not a measurement).
- Whether the 10,001-row `EV_Adoption_and_Range_Anxiety_Dataset.csv` shares those same minima — if the source table's mode equals the competition floor, the floors are imputed placeholders.

- [ ] **Step 4: Write `clean_raw(df)` and the cleaning log**

One function holding every stateless rule. `cleaning_log.md` gets one line per step: rule, rows affected, rows remaining. Then confirm removed rows are not concentrated in one class.

- [ ] **Step 5: Verify the cleaning is stateless**

Run: `clean_raw(df.iloc[:1000])` and `clean_raw(df).iloc[:1000]` and assert the two frames are equal.
Expected: PASS. A failure means a rule learned something from the data and belongs in a pipeline instead.

- [ ] **Step 6: Target summary and Stage 1 decision-log block**

Class balance, rows with a missing target and why. Elapsed minutes.

**Gate:** `column_audit.csv` with a verdict for every flagged column; `cleaning_log.md` with row counts; target summary; candidate keys/groups/time columns handed to Stage 2.

---

### Task 2: Stage 2 — validation design and carving the test set

**Files:**
- Create: `ds-rerun/code/02_validation.py`
- Create: `ds-rerun/artifacts/split/split.csv`
- Create: `ds-rerun/artifacts/split/split_spec.json`
- Modify: `ds-rerun/search_space.json` (write `noise_floor_sd`)

**Interfaces:**
- Consumes: `train_clean.csv`, candidate group columns from Task 1.
- Produces: `s = load_split(...)` giving `s.train_mask`, `s.folds`; the frozen test-ID SHA-256; `noise_floor_sd` in ROC-AUC units.

- [ ] **Step 1: Load the skill**

`cat "C:/dev/important repo/Data Scientist Skill Set/ds-validation-design/SKILL.md"`

- [ ] **Step 2: Answer the three deployment questions in the decision log**

Is the model scoring new rows of known entities, entirely new entities, or a future period? Is there drift? Do labels overlap in time? There is no date column in this data, so the drift and overlap answers must say so explicitly rather than being skipped.

- [ ] **Step 3: Measure the near-duplicate structure — the deferred decision**

668,665 rows synthesised from a 10,001-row source table. Measure, do not assume:
- Count exact duplicate feature-vectors (all 13 features equal) within train, and their target agreement rate.
- Count train rows whose feature-vector also appears in `test.csv`.
- For each candidate near-duplicate key, the group-size distribution.

**STOP HERE AND ASK APOORVA.** This is the contribution point named in the spec: a random `stratified` split treats near-duplicates as independent (matching how Kaggle's own test set relates to the source table); a `group` split on a near-duplicate key defends against memorising a shared ancestor. Present both with the measured numbers and let them write the rule.

- [ ] **Step 4: Carve and freeze**

```bash
python "C:/dev/important repo/Data Scientist Skill Set/ds-validation-design/scripts/make_split.py" \
  ds-rerun/artifacts/train_clean.csv --target Will_Buy_EV --scheme <chosen> --id id \
  --test-size 0.2 --holdout-size 0.15 --n-splits 5 --seed 42 --out ds-rerun/artifacts/split/
```

- [ ] **Step 5: Test-set precision estimate**

Hanley-McNeil standard error for ROC-AUC at the observed positive/negative counts. Compare that interval against the smallest difference that matters. Record it — with ~134k test rows this interval is the thing that decides whether the frozen test can resolve anything at all.

- [ ] **Step 6: Measure the noise floor in ROC-AUC**

`RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)` on one simple pipeline. Record `repeat_sd` as `noise_floor_sd` and `fold_sd` alongside. On 535k training rows, time one fold first and extrapolate before committing to 25 fits.

- [ ] **Step 7: Split sanity checks**

Target distribution per part matches; `group_overlap` is 0 where the policy requires it; `folds_sharing_groups` is 0. Run adversarial validation train-vs-test and confirm it is near 0.5.

- [ ] **Step 8: Stage 2 decision-log block**

At least one rejected scheme, with why. Elapsed minutes.

**Gate:** split spec with rejected alternatives; frozen test IDs with their SHA-256; noise floor in `search_space.json`; sanity checks passed.

---

### Task 3: Stage 3 — leakage hunt and relationship EDA (training rows only)

**Files:**
- Create: `ds-rerun/code/03_leak_eda.py`
- Create: `ds-rerun/artifacts/leak_screen.csv`
- Create: `ds-rerun/artifacts/leak_table.csv`

**Interfaces:**
- Consumes: `s.train_mask` from Task 2 — **the test and holdout rows are not read at this stage**.
- Produces: the fixed drop-list of leaking columns, never searched over.

- [ ] **Step 1: Load the skill**

`cat "C:/dev/important repo/Data Scientist Skill Set/ds-data-audit/SKILL.md"` — Part B.

- [ ] **Step 2: Run the bundled leak screen on training rows only**

```bash
python "C:/dev/important repo/Data Scientist Skill Set/ds-data-audit/scripts/leak_screen.py" \
  ds-rerun/artifacts/train_portion.csv --target Will_Buy_EV --task binary \
  --exclude id --out ds-rerun/artifacts/leak_screen.csv
```

- [ ] **Step 3: Build the leak table with a record-timing verdict per column**

Columns: `column, score, rank, recorded_when, available_at_prediction, verdict, evidence`. Every flagged column, the top five by score, plus any column whose name suggests an after-the-fact record.

For this dataset the honest answer for most columns is that timing is **undocumented** — there is no data dictionary. Undocumented plus `NEAR_PERFECT`/`DOMINANT` is a presumed leak; undocumented plus a modest score, on a feature that is plainly a buyer attribute rather than an outcome, is `legit` with the assumption written down. Write the assumption, do not silently assume it.

- [ ] **Step 4: Check the leaks a column-by-column screen cannot see**

Multivariate leakage, duplicate rows straddling the split, and the target-rate-by-level tables for each categorical. Record pairwise association between features (Cramér's V for categorical pairs) — independence must be measured, not assumed.

- [ ] **Step 5: Stage 3 decision-log block**

**Gate:** `leak_table.csv` with a record-timing verdict per suspect column.

---

### Task 4: Stages 4–5 — missing values and outliers

**Files:**
- Create: `ds-rerun/code/04_missing.py`
- Create: `ds-rerun/code/05_outliers.py`
- Create: `ds-rerun/artifacts/missing_profile.csv`, `ds-rerun/artifacts/outlier_profile.csv`
- Modify: `ds-rerun/search_space.json`

**Interfaces:**
- Consumes: training rows and folds from Task 2; the drop-list from Task 3.
- Produces: 2–3 carried-forward imputers and 2–3 outlier treatments, each with score, spread and the families it applies to.

- [ ] **Step 1: Run the bundled profilers**

```bash
python ".../ds-missing-values/scripts/missing_profile.py" <train portion> --target Will_Buy_EV --out ds-rerun/artifacts/missing_profile.csv
python ".../ds-outliers/scripts/outlier_profile.py"  <train portion> --target Will_Buy_EV --out ds-rerun/artifacts/outlier_profile.csv
```

These run in full even under time pressure — the orchestrator classes them as looking, not searching.

- [ ] **Step 2: Diagnose the mechanism, then screen the menu with two proxy models**

A regularised linear proxy and a small GBDT proxy under the Stage 2 folds, per the carry-forward rule. Keep a candidate within 1 SE of the stage best, or where it is the best option for a family the others do not serve.

If the Stage 1 floor investigation concluded the minima are imputed placeholders, then **floors-as-NaN is a candidate to screen**, not a conclusion to apply — screen it and record the measured delta.

- [ ] **Step 3: Outlier verdicts — error, extreme, or signal**

Each flagged cluster gets one of the three verdicts with its evidence. On synthetic data an "outlier" is often generator structure rather than an error; that has to be argued from the data.

- [ ] **Step 4: Decision-log blocks for Stages 4 and 5, and write survivors to `search_space.json`**

**Gate:** mechanism table; outlier table with error/extreme/signal verdicts; 2–3 candidates carried forward at each stage.

---

### Task 5: Stage 6 — features

**Files:**
- Create: `ds-rerun/code/06_features.py`
- Create: `ds-rerun/artifacts/feature_ledger.csv`
- Modify: `ds-rerun/search_space.json`

**Interfaces:**
- Produces: 2–3 carried-forward feature sets with their ablation deltas.

- [ ] **Step 1: Load the skill and build the feature ledger**

One row per feature: name, source, availability time, rationale. A feature whose availability time is after the prediction moment does not enter the ledger.

- [ ] **Step 2: Screen the encoding menu under the Stage 2 folds**

All 13 features are low-to-moderate cardinality except `Annual_Income_USD` and `Daily_Commute_km`, which are numeric with many distinct values. Target encoding of high-cardinality numerics is a candidate like any other and must be **fitted strictly in-fold** — an encoder fitted outside the fold is the classic silent leak on this kind of data.

- [ ] **Step 3: Ablate**

Each candidate set measured against the previous best on paired per-fold differences, judged against `noise_floor_sd`.

- [ ] **Step 4: Decision-log block**

**Gate:** feature ledger with availability time and ablations; 2–3 sets carried forward.

---

### Task 6: Stage 7 — baselines, families, imbalance

**Files:**
- Create: `ds-rerun/code/07_models.py`
- Create: `ds-rerun/artifacts/baseline_table.csv`, `ds-rerun/artifacts/family_screen.csv`

- [ ] **Step 1: Baselines first**

`DummyClassifier`, plus the bundled `rule_baseline.py` for a business-rule baseline. A model that loses to the rule is not shipped.

- [ ] **Step 2: Screen families cheaply before spending compute**

Regularised logistic, LightGBM, XGBoost, CatBoost, and a small MLP. Time one fold on a subsample and extrapolate before committing — 535k rows makes this the first stage where the compute rule bites.

- [ ] **Step 3: Learnability verdict**

If the best family does not beat the simple baseline by more than the noise floor and the learning curve is flat, that is a **stop signal**: report it rather than pressing on.

- [ ] **Step 4: Imbalance handling**

Check the positive rate first. Resampling happens **inside** the training fold, never before the split.

- [ ] **Step 5: Decision-log block**

**Gate:** baseline table; family screen with spread; learnability verdict.

---

### Task 7: Stage 8 — joint conditional search

**Files:**
- Create: `ds-rerun/code/08_joint_search.py`
- Create: `ds-rerun/artifacts/search_log.csv`, `ds-rerun/artifacts/finalists.json`

- [ ] **Step 1: Run the bundled joint search**

`joint_search.py` turns every `search_space.json` entry into a categorical hyperparameter beside the model's own, in a conditional space, with pruning. The winner is the best *combination*, not the best of each stage in isolation.

- [ ] **Step 2: Apply the one-standard-error rule**

Report the winner and every runner-up within 1 SE. Ties go to the simpler, cheaper, more interpretable option.

- [ ] **Step 3: Locked-holdout check**

Score the finalists on the Stage 2 holdout — the search overfits its own folds, and this is how that shows up. A finalist that wins on folds and loses on the holdout is rejected.

- [ ] **Step 4: Decision-log block**

**Gate:** search log; winner vs runners-up within 1 SE; locked-holdout check.

---

### Task 8: Stage 9 — evaluation, scored once

**Files:**
- Create: `ds-rerun/code/09_final_eval.py`
- Create: `ds-rerun/artifacts/final_eval.json`, `ds-rerun/artifacts/model_card.md`

- [ ] **Step 1: Verify the test set is untouched**

```python
from cv_folds import verify_test_untouched
assert verify_test_untouched("ds-rerun/artifacts/split") is True
```
Expected: True. If False, stop — the run's central claim is void and no test score may be reported.

- [ ] **Step 2: Score the frozen test set, once**

`final_eval.py`: test AUC with bootstrap CI, calibration, threshold, per-slice AUC, error analysis.

- [ ] **Step 3: Sanity tests**

Shuffled-label canary — refit the exact final pipeline on permuted labels and confirm AUC sits at chance. A canary meaningfully above 0.5 means label information is reaching the pipeline and invalidates the result.

- [ ] **Step 4: Decision-log block**

**Gate:** test score with CI, threshold, calibration, slices, error analysis, sanity tests.

---

### Task 9: Stage 9b — the blind-mode comparison

**Files:**
- Create: `ds-rerun/artifacts/blind_comparison.md`

**This task is the point of the exercise and is not optional.**

- [ ] **Step 1: Only now, open the prior analysis**

`reports/02_scoreboard.md`, `reports/03_backlog.md`, and `src/`.

- [ ] **Step 2: Write the comparison table**

One row per finding: `Finding (this run) | Prior analysis says | Verdict`, where verdict is `AGREED`, `DISAGREED`, `NEW`, or `MISSED`.

- [ ] **Step 3: Draw the conclusion about the skill set**

`MISSED` rows are gaps in the skill set and are the most informative output. Say plainly whether the skill set found independently what ~60 prior experiments found, and what it cost in time and tokens.

---

### Task 10: Stage 10 — handoff and submission

**Files:**
- Create: `ds-rerun/code/10_handoff.py`
- Create: `ds-rerun/artifacts/pipeline.joblib`
- Create: `ds-rerun/subs/submission_dsrerun.csv`
- Modify: `ds-rerun/report.md` (final fill)

- [ ] **Step 1: Save the whole fitted pipeline**

Preprocessing included, library versions and seeds pinned. Its **first step is `FunctionTransformer(clean_raw)`** so raw production rows get exactly the cleaning the training data got.

- [ ] **Step 2: Score raw rows end to end**

Feed a handful of raw, uncleaned rows through the saved pipeline; confirm output and latency. Feed one row twice and confirm identical scores.

- [ ] **Step 3: Refit on all training rows and predict the Kaggle test set**

Only now. The pipeline was chosen on local evidence alone; this step does not revisit that choice.

- [ ] **Step 4: Write the model card and fill the report**

Problem, data found, decisions with reasons, results against baselines with intervals, error analysis, limitations (including every `ASSUMED` line), next steps.

**Gate:** saved pipeline scoring raw rows; model card; monitoring plan; final report; submission file.

---

## Self-review

**Spec coverage:** Workspace → Task 0. Blindness protocol → Global Constraints + Task 9. Stage plan → Tasks 0–8, 10. Stage 9b → Task 9. Compute strategy → Tasks 2 (step 6), 6 (step 2). Deferred split question → Task 2 step 3, with an explicit stop. Definition of done, items 1–6 → Tasks 0–10 gates, Task 9, Task 10 step 4. All covered.

**Placeholder scan:** Later-stage tasks specify script, gate artifact, verification and decision rule rather than literal code, because their content is determined by earlier stages' measurements. This is a data-dependent procedure, not an unfilled placeholder; the decision rule in each step is the executable part.

**Type consistency:** `clean_raw(df) -> df` (Task 1) is reused in Task 10 step 1. `s.train_mask` / `s.folds` from `load_split` (Task 2) are consumed in Tasks 3–7. `noise_floor_sd` (Task 2) is consumed in Tasks 4–7. `verify_test_untouched` (Task 2 interface) is called in Task 8. Consistent.
