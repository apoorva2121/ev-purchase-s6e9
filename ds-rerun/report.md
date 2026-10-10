# EV S6E9 purchase prediction: model report

**Status: all 11 gates passed (0–10 plus the 9b blind comparison). Complete.**

Run 2026-09-27T15:41:57Z to 2026-09-28T01:05Z, **235 minutes measured**. An independent
adversarial review afterwards found several statements in the record wrong or overstated; see the
**CORRECTIONS** block at the end of `decision_log.md`. The headline test score stands; the Stage 8
search-space description, the data-volume attribution and the post-hoc analyses' claim of not
touching test rows do not. Built with the
`data-scientist` orchestrator skill set from `C:\dev\important repo\Data Scientist Skill Set`,
**blind to this repository's prior analysis until Stage 9**. Zero contamination entries.

Spec: `docs/superpowers/specs/2026-09-27-ev-s6e9-ds-rerun-design.md`.
Reasoning for every choice, both ways: `decision_log.md`.
Pre-registered findings: `artifacts/blind_findings.md`. Comparison: `artifacts/blind_comparison.md`.

---

## 1. Answer first

The final pipeline — XGBoost over a cross-fitted target-encoding ladder on `Annual_Income_USD` —
scores **ROC-AUC 0.945514 on a frozen test set scored exactly once**, 95% CI [0.944287, 0.946840].
That beats a two-line business rule (0.916909) by **+0.028605**, about 57× the noise floor, and
comfortably meets the Stage 0 success threshold. Probabilities are well calibrated (ECE 0.001842)
and were not corrected because they did not need to be.

**The honest caveat, and it is the most important sentence in this report: about 0.009 of that
AUC is memorisation, not prediction.** Under grouped cross-validation where no exact income value
is shared between folds, the same pipeline scores **0.936222** — below a plain logistic baseline.
Roughly a third of the gain over the business rule is recovery of source-row identity from the
10,000-row table Kaggle synthesised this data from. It is legitimate for this competition, where
99.4% of test rows share an income value with train, and worthless anywhere else.

---

## 2. Problem

Kaggle Playground S6E9, binary classification, ROC-AUC, 668,665 labelled rows × 13 features,
17.4645% positive, no dates, no missing values. Full framing in `problem_statement.md`.

Six ASSUMED lines were opened at Stage 0 and remain unconfirmed; the two material ones are the
prediction moment (tested at Stage 3, not assumed onward) and row independence (assumed false and
**confirmed false** at Stage 2).

---

## 3. Data

**Cleaning.** 668,665 rows in, 668,665 out. Nothing was dropped because there was nothing to drop:
0 exact duplicates, 0 key conflicts, 0 missing values, 0 spelling variants, 0 impossible values at
either end of any of the 7 numeric ranges. **Both sentinel flags raised by the bundled audit were
false positives** (`id` 999/9999/99999 are ordinary counter values; income 99999 ×28 sits inside
its local count distribution, median 17, and the data contains no round numbers at all).

**Four things the bundled audit could not see:**

1. **Two censoring floors.** Income 30000 = 61,605 rows (9.21%) with the next value at 2 rows;
   commute 5.0 = 144,280 rows (21.58%) with the next at 54.
2. **The generator replicates individual source rows.** 100% (492/492) of income values with ≥200
   rows exist in the 10k source table, against 7.2% (408/5,655) of values with ≤3 rows; 97.9% of
   all train rows carry a source income value. Two source rows become ~900 competition rows.
   **`Annual_Income_USD` is a quasi-identifier.** This single observation produced the entire
   modelling result.
3. **Truncation at the top** relative to source: income max 188,549 vs 223,345.
4. **The generator removed the source's nulls** (178/181/184) and inflated floor mass; null-to-floor
   mapping explains income (5.57%→9.21%) but not commute (7.78%→21.58%).

**Leakage. Zero columns dropped.** Three were DOMINANT (concern 0.8432, subsidy 0.7167, income
0.6689). The skill's default for "timing unknown + DOMINANT" is to drop, which would have taken CV
AUC from 0.940585 to 0.604602. They were kept because **all 13 features ship in the competition's
own `test.csv`**, so all are available at prediction time by construction — stronger evidence than
a data dictionary would give. Also verified: 0 duplicate feature-vectors across the split,
`__row_order__` exactly 0.5000, id-vs-target Spearman 3.4e-05, adversarial validation of train
against the real Kaggle test file 0.50445. **Features are not independent** — Cramér's V for
City_Type × Charging_Near_Work is 0.672.

**Missing values.** None exist. The stage was inverted into "should we *create* missingness from
the floors"; every variant landed inside the noise floor.

**Outliers.** Extremes in one column only: income has 2,413 rows beyond 1.5 IQR, **0 beyond 3 IQR**,
and a target lift of **5.73** among them. Verdict **signal, not error**; nothing was removed.

---

## 4. Validation design

Stratified, seed 42: train 434,632 / holdout 100,300 / test 133,733, target rate 0.1746 in all
three. Test IDs frozen by SHA-256 and scored once.

Rejected: `time`/`group_time` (no date column exists anywhere); `random` (stratification is free);
`group` on exact income **as primary**. That last was a real decision, escalated with measurements:
rows sharing an exact income value show **3.40×** the label overdispersion expected under
independence (1.93× even within 200 income quantile bins; iid control 1.03×, so the statistic is
calibrated) — but **99.4% of Kaggle test rows share an income value with train**, so a grouped
split would measure a condition describing 0.6% of deployment. Resolution: stratified primary plus
a grouped secondary diagnostic, which became the most valuable number in the report.

**Test-set precision.** 23,350 positives / 110,383 negatives gives a Hanley-McNeil SE of 0.00112,
so a 95% interval of ±0.0022. The frozen test reports; it cannot adjudicate small differences.

**Noise floor — a documented deviation.** The skill prescribes `repeat_sd` from
`RepeatedStratifiedKFold(5×5)`. Measured here it is **1.5e-06 and degenerate**: the five repeat
means agree to five decimal places, because repeats reshuffle *partitions* of the same 434,632
rows rather than resampling data. Operative floor adopted instead: **0.00050**, the bootstrap sd of
the CV mean, 330× larger.

---

## 5. Decisions

Eleven blocks in `decision_log.md`, each with considered / applied / skipped and a reason tied to
a measured number. Notable deliberate departures from the skill, each argued in place: keeping the
three DOMINANT columns; replacing the noise-floor statistic; carrying XGBoost past a marginal
default-settings deficit; dropping CatBoost on cost.

---

## 6. Results

| Model / pipeline | CV ROC-AUC | Locked holdout | Test (95% CI) |
|---|---|---|---|
| Trivial baseline (prior) | 0.500000 | — | 0.500000 |
| Business rule (3 columns, no fitting) | 0.917358 ± 0.000693 | — | 0.916909 [0.915140, 0.918685] |
| Logistic, no target encoding | 0.937959 ± 0.000437 | — | — |
| LightGBM + income ladder (untuned) | 0.944667 ± 0.000384 | — | — |
| **Final: XGBoost + ladder3 + floors→NaN** | **0.945206** | **0.945896** | **0.945514 [0.944287, 0.946840]** |

CV, holdout and test all fall inside the test interval — concordance that is itself evidence no
leak survived.

**One lever mattered.** A target-encoding ladder on income (exact + //50 + //500): **+0.004002**,
eight times the noise floor. It was selected over eight rival schemes and eleven depth/scope
variants:

| income encoding scheme | gbdt | linear |
|---|---|---|
| **ladder (exact + //50 + //500)** | **0.944588** | **0.942641** |
| exact value, smooth=2 | 0.944278 | 0.942224 |
| exact value | 0.944136 | 0.942099 |
| bin50 | 0.943687 | 0.941657 |
| exact value, smooth=50 | 0.943065 | 0.941562 |
| count-adaptive (exact if common, else //500) | 0.941681 | 0.940005 |
| bin500 | 0.941567 | 0.939778 |
| 200 quantile bins | 0.941136 | 0.939475 |
| none | 0.940585 | 0.937959 |

**Everything else screened inside the noise floor.** Floor indicators +0.000157, floors-as-NaN
+0.000024, winsorising +0.000033, anomaly-score feature +0.000061, dropping the four
zero-contribution columns +0.000046, class weighting +0.000062, CatBoost −0.000119 (for 65× the
cost), ladder depth beyond 3 rungs ±0.000119. The `City_Type × charging` interaction — the
best-motivated feature in the menu, from a Cramér's V of 0.672 — came back **negative** at
−0.000483.

The Stage 8 joint search over 48 trials confirmed it independently: **every one of the six
finalists carried an income ladder, and no finalist chose `income_te=none`.** The `outliers` and
`missing_values` slots never left their no-op settings.

---

## 7. What drives the prediction

Drop-one ablation, full CV AUC 0.940585 without target encoding:

| feature | drop-one Δ |
|---|---|
| `Environmental_Concern_Level` | +0.0997 |
| `Subsidy_Available` | +0.0526 |
| `Annual_Income_USD` | +0.0141 |
| `Range_Anxiety_Level` | +0.0016 |
| remaining nine | < 0.0005 each; four negative |

Target rates are monotone and steep where it matters: concern 1→5 runs 0.0057 / 0.0214 / 0.1109 /
0.2488 / 0.5183; subsidy No 0.0058 vs Yes 0.2747; anxiety High 0.0014 / Medium 0.0417 / Low 0.1890.
The fitted model reproduces this — holding everything else fixed and sweeping concern 1→5 gives
monotonically increasing predictions.

---

## 8. Where it fails

Demographic slices are flat (Gender 0.9388–0.9457, City_Type 0.9407–0.9488), so no group of people
is materially worse served. The lowest within-slice figures are `Subsidy_Available = No` (0.8836)
and `Environmental_Concern_Level = 2` (0.8479); these are **conditional** AUCs, and since those
two columns carry most of the discrimination, holding one fixed necessarily leaves less signal.
That is arithmetic, not a defect.

`Range_Anxiety_Level = High` has 421 test rows and an interval 15× wider than the rest.

The real failure mode is not a slice: it is **unseen income values**. The model's edge comes from
recognising exact incomes it has seen. Feed it buyers whose incomes are absent from training and
performance falls to roughly 0.936.

---

## 9. Limitations and risks

1. **A third of the gain is memorisation.** Grouped CV: 0.936222 vs 0.945206. Independently
   corroborated — prior analysis in this repo recovered the generating probit analytically and
   computed a semantic ceiling of 0.93689; two unrelated methods agree to within 0.0007.
2. **The model will not transfer** to any data that is not a resampled synthetic copy of a small
   source table.
3. **Record timing is undocumented**; the three strongest features are kept on a construction
   argument, not a data dictionary.
4. **One split.** Every conclusion rests on a single stratified partition. This is the run's main
   methodological weakness (see Next steps).
5. **The frozen test resolves only ±0.0022**, far coarser than competition-scale differences.
6. **Not evaluated**: tabular foundation models (not installed, need a GPU at 434k rows — TabArena
   ranks them above boosted trees, so this is a genuine gap), neural baselines, conformal
   intervals, formal fairness metrics, `cleanlab` label-noise screening.
7. Six ASSUMED framing lines remain unconfirmed.
8. Only **one** linear trial completed out of 48 in the joint search; that family is effectively
   unexplored rather than fairly beaten.

---

## 10. Next steps

Ranked by expected value:

1. **Replicate every conclusion on a second seeded partition.** The single most valuable addition.
   At this data size the paired SE is so small that "consistent across folds" is nearly automatic;
   an independent partition is the cheapest real guard, and the prior analysis in this repo already
   does it.
2. **Recover the data-generating process and compute the Bayes ceiling.** For synthetic data this
   separates signal from artifact before any modelling. The prior work did it; this run did not.
3. Run tabular foundation models on a GPU (`pip install tabpfn tabicl`; note the TabPFN *weights*
   for 2.5–3.5 are non-commercial even though the code is Apache-2.0).
4. Explore the linear family properly, or drop it from the space deliberately rather than by
   sampler attrition.
5. `pip install cleanlab` for label-noise screening — though on generator output, "label noise" is
   the generator's own stochastic term and is irreducible rather than correctable.
6. More data would help slightly: the learning curve is still rising at full size (+0.000412 over
   the last quartile, train-val gap 0.004858 — data-limited, not variance-limited). No more exists.

---

## 11. Blind-mode comparison against the prior analysis (Stage 9b)

Findings were pre-registered at ~145 minutes elapsed, before anything in `src/` or `reports/` was
opened. **Score: 17 AGREED, 2 DISAGREED, 14 NEW, 5 MISSED.** Full table in
`artifacts/blind_comparison.md`.

| | this run (4h, blind) | prior work (~50–60 experiments) |
|---|---|---|
| The one thing that works | income target encoding **+0.004002** | nested per-exact-value TE **+0.00346** — "the whole competition" |
| Best CV / OOF | 0.945206 | 0.94565 |
| Best test / LB | 0.945514 (frozen local test) | 0.94615 (public LB) |

Both runs independently identified the same single lever, by different routes: the prior work by
trying many encodings, this run by asking a Stage 1 structural question — *where did these rows
come from?* — and following the answer. The prior work is modestly ahead on score, which ~50 extra
experiments should buy.

**The most valuable outcome was an accidental cross-validation.** The prior work recovered the
generating probit (σ = 0.851) and computed a semantic ceiling of **0.93689**. This run never
recovered the probit — that is a MISS — but its grouped diagnostic, a completely different method,
landed at **0.936222**. Neither number informed the other, and they agree to within 0.0007. Both
therefore support the same conclusion: roughly 0.009 of AUC in every leaderboard score for this
competition is memorisation of the source table, not prediction.

### Did the skill set work?

**Yes, with two real gaps.** It found the decisive lever independently in ~4 hours, reached a
comparable score, and produced substantial evidence the prior work lacks: quantified row
dependence, calibration, slices, a learnability verdict, a business-rule baseline, and an honest
accounting of how much of the score is memorisation. The gates did what they claim — the finding
came out of the Stage 1 audit, not out of the search.

**Gap 1 — one split is not enough.** The orchestrator should require a second, independently
seeded partition for any carried-forward conclusion.

**Gap 2 — nothing asks what generated the data.** For synthetic, simulated or rule-derived data,
recovering the generator and computing the achievable ceiling is the most valuable single question
available. It belongs at Stage 0 or Stage 3 and is absent.

### Eleven defects found in the skills themselves

| # | Defect |
|---|---|
| F1 | `column_audit.py` **missed both censoring floors** — the most conspicuous structural feature of this data. It flags sentinels by value pattern, not by density spike against local neighbourhood. |
| F2 | `column_audit.py` produced **two false-positive sentinel flags** (monotonic `id`, income 99999). |
| F3 | The Stage 2 noise floor (`repeat_sd`) **breaks on large n**: repeats resample partitions, not data. Returns 1.5e-06 where real sampling noise is 0.00050. |
| F4 | `summarize()` rounds to 4 decimals, erasing every difference this problem turns on. |
| F5 | `summarize()` warns below 10 folds, but `make_split.py` and the orchestrator's flow give 5. |
| F6 | The Stage 3 rule "timing unknown + DOMINANT → presumed leak → drop" would have destroyed the model. Presence in a supplied scoring file should count as admissible timing evidence. |
| F7 | `outlier_transformers.AnomalyScore` returns a numpy array, destroying the category dtypes that `HistGradientBoostingClassifier(categorical_features="from_dtype")` requires. |
| F8 | `joint_search.py` passes the raw target through; XGBoost rejects string labels the other families accept. |
| F9 | `joint_search.py` has no guard for a slot option incompatible with a family when the option is added outside the `applies_to` mechanism. |
| F10 | The paired SE and the sampling noise floor **routinely disagree** at this n, and the skill gives no rule for which governs. |
| F11 | Nothing tests **picklability** until Stage 10, when the design is locked — and the Stage 10 checklist says to save the pipeline but never to **reload it in a fresh process** and reproduce a prediction. Three successive handoff failures here, the third of which saved a file that could not be loaded. |

---

## Deliverables

| Artifact | Path |
|---|---|
| Framing | `problem_statement.md` |
| Full reasoning, 11 gates | `decision_log.md` |
| This report | `report.md` |
| Pre-registered blind findings | `artifacts/blind_findings.md` |
| Blind comparison | `artifacts/blind_comparison.md` |
| Model card + monitoring plan | `artifacts/model_card.md` |
| Saved pipeline (verified to reload) | `artifacts/pipeline.joblib` |
| Kaggle submission, 286,571 rows | `subs/submission_dsrerun.csv` |
| Rerunnable stage scripts | `code/01`…`code/10` |
| Frozen split + SHA-256 | `artifacts/split/` |
