# EV S6E9 — independent rebuild under the Data Scientist Skill Set

Date: 2026-09-27
Status: approved by Apoorva, 2026-09-27
Clock started: 2026-09-27T15:41:57Z

## Purpose

Rebuild the Kaggle Playground S6E9 "Predicting Electric Vehicle Purchases" model from scratch,
driven by the `data-scientist` orchestrator in `C:\dev\important repo\Data Scientist Skill Set`,
without consulting the prior analysis in this repo until Stage 9.

Two deliverables, strictly sequenced:

1. A complete, auditable ten-stage methodology record ending in a frozen local test score.
2. Only after (1) signs off: a refit on all training rows and a Kaggle submission.

The primary deliverable is (1). This is as much a real-world test of the skill set as it is a
modelling exercise.

## Decisions taken at design time

| Decision | Choice | Reason |
|---|---|---|
| Dataset | EV Purchase S6E9 | Chosen by Apoorva from four candidates |
| Skill loading | Read `SKILL.md` files from the repo path | Installing junctions into `~/.claude/skills` affects every future session; parked as STATUS item 16 |
| Goal | Both, strictly sequenced | Methodology record first, submission second |
| Blindness | Blind until Stage 9, then compare | Strongest evidence about whether the skill set works |
| Budget | One long session, ~4–6h, full fidelity | No compact mode; repeated CV and a real joint search are affordable |
| Test set | Frozen local carve from the 668k train rows | Resolves the conflict with Kaggle's resubmittable LB |

## Workspace

`ml-bench/ev-purchase-s6e9/ds-rerun/`

| Path | Contents |
|---|---|
| `problem_statement.md` | Stage 0 framing, every field filled, ASSUMED lines marked |
| `decision_log.md` | One block per stage: considered / applied because / skipped because |
| `search_space.json` | Carry-forward candidates per stage, consumed by the Stage 8 joint search |
| `report.md` | Skeleton written at Stage 0, filled at every gate |
| `code/NN_stage.py` | One rerunnable script per stage |
| `artifacts/` | Tables, splits, frozen test IDs, plots, search logs |

Nothing outside `ds-rerun/` is written. The existing `src/`, `artifacts/`, `subs/`, `reports/`
and `logs_*.txt` are neither read nor modified.

## Blindness protocol

Read-allowlist until Stage 9 completes:

- `data/raw/*.csv`
- `ds-rerun/**`
- The skill set at `C:\dev\important repo\Data Scientist Skill Set\**` (excluding `evals/`)

Off-limits until Stage 9:

- `src/`, `artifacts/`, `subs/`, `nb/`, `tests/`, `logs_*.txt`
- `reports/02_scoreboard.md`, `reports/03_backlog.md`
- `C:\dev\important repo\reports\Kaggle S6E9 EV winning approach.md`
- The Kaggle public leaderboard, including as a source of an OOF→LB offset

Enforcement is by record, not by sandbox. Any consult becomes a dated `CONTAMINATION` entry in
`decision_log.md` naming the file read, the question asked, and what was done with the answer.

**Known, disclosed limitation.** The agent's context already contains Apoorva's memory file for
this competition, which includes prior conclusions. A fully blind rerun was therefore already
impossible at design time. The operating rule is: *where recall would let me skip a measurement,
I measure instead*, and the decision log records the measured number rather than the recalled one.

## Stage plan

Tabular binary-classification path, all ten stages. Each stage loads its step skill, produces its
gate artifact, and writes its decision-log block before the next stage starts.

| # | Stage | Skill | Gate artifact |
|---|---|---|---|
| 0 | Frame the problem | orchestrator | `problem_statement.md` complete, path chosen, clock started |
| 1 | Structure audit + stateless cleaning | `ds-data-audit` A | column audit table, cleaning log with row counts |
| 2 | Validation design + carve test | `ds-validation-design` | split spec with rejected alternatives, frozen test IDs, noise floor in AUC units |
| 3 | Leakage hunt + relationship EDA | `ds-data-audit` B | leak table with a record-timing verdict per suspect column |
| 4 | Missing values | `ds-missing-values` | mechanism table, screened imputers, 2–3 carried forward |
| 5 | Outliers | `ds-outliers` | outlier table with error / extreme / signal verdicts |
| 6 | Features | `ds-feature-engineering` | feature ledger with availability time and ablations |
| 7 | Baselines + families + imbalance | `ds-model-selection` | baseline table, family screen with spread, learnability verdict |
| 8 | Joint pipeline search | `ds-method-comparison` | search log, winner vs runners-up within 1 SE, locked-holdout check |
| 9 | Evaluation + sign-off | `ds-evaluation` | test AUC with CI, calibration, slices, error analysis, shuffled-label canary |
| 9b | Blind-mode comparison | this spec | diff of independent findings against `reports/03_backlog.md` |
| 10 | Handoff | orchestrator | saved pipeline scoring raw rows, model card, Kaggle refit and submission |

### Stage 9b — the comparison

Stage 9b is the point of the exercise and is not optional. After the frozen test is scored, open
the prior analysis and produce a table with one row per finding:

| Finding (this run) | Prior analysis says | Verdict |
|---|---|---|

Verdicts: `AGREED`, `DISAGREED`, `NEW` (found here, absent there), `MISSED` (there, absent here).
`MISSED` rows are the most informative — they are gaps in the skill set.

## Compute strategy

668,665 × 13 is large enough that naive repeated CV at every stage would exhaust the budget.
Per the orchestrator's compute rule, time one fold on a sample and extrapolate before committing
to any step expected to exceed a few minutes.

Planned shape, subject to those measurements:

- Stages 4–7: screen on a stratified subsample, size chosen from measured fold timing.
- Stages 8–9: full training portion.
- Every subsample decision is logged with the timing that justified it.

Time split across stages: ~25% stages 0–3, ~35% stages 4–7, ~25% stage 8, ~15% stages 9–10.
Elapsed minutes are measured with a clock and written into each gate's decision-log block.

## Open question deferred to Stage 2

The 668k competition rows are synthesised from a 10,001-row source table
(`EV_Adoption_and_Range_Anxiety_Dataset.csv`). Near-duplicate rows therefore almost certainly
straddle any random split. Two readings, and they disagree:

- **Group leakage.** Rows sharing a source ancestor are not independent; a random split lets the
  model memorise an ancestor seen in training. Defence: group the split by a near-duplicate key.
- **Correct structure.** The Kaggle test set has the same relationship to the same source table,
  so a random split matches the deployment condition and a grouped split would understate
  achievable performance.

This choice sets the noise floor and everything downstream. Both alternatives will be measured
and the decision put to Apoorva with the numbers in hand.

## Risks accepted

- Blind mode may re-spend compute on ideas the prior backlog already killed with a number. This
  cost is the price of the evidence requested.
- Ranks 2–500 on this competition span 0.00038 AUC. The probability of beating the prior 0.94615
  is low. The confident deliverable is the methodology record, not a rank.
- A frozen local test set scored once is a weaker estimate of LB position than the prior work's
  calibrated OOF→LB offset. That is a deliberate trade: local honesty over LB calibration.

## Definition of done

1. All ten gate artifacts exist in `ds-rerun/`.
2. `decision_log.md` has a block per stage, with reasons both ways.
3. The frozen test set was scored exactly once, and the shuffled-label canary passed.
4. Stage 9b comparison table is written.
5. `report.md` is complete: problem, data found, decisions with reasons, results against baselines
   with intervals, error analysis, limitations, next steps.
6. A Kaggle submission file exists, produced by a pipeline chosen on local evidence alone.
