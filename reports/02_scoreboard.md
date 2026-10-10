# Experiment scoreboard — S6E9

Rules, so results stay comparable and honest:
1. Every run uses the **frozen split**: StratifiedKFold(n=5, shuffle=True, random_state=42).
2. Changes are measured as a **paired delta vs the stated parent**, on identical folds.
3. A delta is only "real" if it survives a paired test. Fold SEM on a single run
   is ~0.0004, so an unpaired gap under ~0.001 means nothing.
4. Negative results are recorded, not deleted. Knowing what fails is the deliverable.

| # | Experiment | Parent | OOF AUC | Delta | Verdict | Notes |
|---|---|---|---|---|---|---|
| 00 | Raw features, LGBM d5 ff0.3 lr0.03 | — | **0.94219** | — | baseline | 243s, ~1470 iters. Canary 0.50174 (clean). |
| 01 | Nested per-exact-value TE (income+commute, m=5/20/100) + freq + floor flags + dist-to-original | 00 | **0.94565** | **+0.00346** | KEEP | paired bootstrap sd 0.00008, P=1.000. 256s. |

## Reference points
- Public LB leader: 0.94945 (anomalous, +0.0027 over 2nd — presumed leak/LB-overfit)
- Realistic top of field: ~0.94674
- Reported top-50 threshold: ~0.94652
- Baseline gap to top-50: +0.0043
- **Current (exp01) gap to field max 0.94674: +0.00109**
- Paired-bootstrap resolution: sd 0.00008, so ~0.0002 effects are measurable

## Canary log
Any pipeline change that adds target-derived features MUST be re-run with
shuffled labels. Expected OOF ~0.500. Anything above ~0.52 means a leak.

| # | Pipeline | Shuffled-y OOF | Pass |
|---|---|---|---|
| 00 | Raw features | 0.50174 | yes |
| 01 | Nested TE (early stopping OFF for canary) | 0.50149 | yes |

## Closed dead ends (measured, do not revisit)
| Idea | Evidence | Verdict |
|---|---|---|
| `id` row-order leak | AUC 0.49999, ids monotone | dead |
| Duplicate rows train<->test | 0 shared feature-rows | dead |
| `income % 1000` binary flag | identical to the 30,000 floor rows | redundant |
| `income % 1000` continuous | standalone AUC 0.52706; 0.48326 off-floor | weak |
| `income_in_original` flag | standalone AUC 0.49851 | dead |
| Original-label lookup by income | standalone AUC 0.51609 | weak |
| Concatenating original 10k as rows | lost 0/5 folds in published ablation | skip |

## Data-generating process (measured locally)
Claimed rule `1.2*inc/1e5 + 0.6*concern + 2*subsidy - 1*(anx=Med) - 3*(anx=High) + N(0,1) > 5.5`
scores AUC 0.93769 on competition data — but its analytic ceiling is 0.91482, which
our 0.94219 baseline already beat. The noise is **not** N(0,1).

Free-scale probit refit: `inc +1.4095, concern +0.7358, subsidy +2.3410,
anx_med -1.0172, anx_high -2.9972, const -6.7148`; implied **sigma = 0.851**.
Fitted-probit AUC 0.93777, its analytic ceiling 0.93689 (self-consistent).

**Consequence:** everything above ~0.9378 is artifact exploitation, not semantics.
exp01 is +0.00788 above the best semantic model. Nine of thirteen features have
|corr with Bayes residual| < 0.017 and are near-worthless on their own.
