# EV purchase prediction — Kaggle Playground S6E9

Predicting whether a person will buy an electric vehicle, from 13 survey
features over 668,665 rows. Metric: **ROC AUC**.

This repository is less about the final score than about *how the score was
earned*. The interesting result here is not the AUC — it is the measurement that
**everything above ~0.9378 on this dataset is artifact exploitation, not
semantics.** That finding is documented below and it changed how I spent the
rest of the time.

| | OOF AUC | Notes |
|---|---|---|
| Baseline — LightGBM on raw features | 0.94219 | 243 s, ~1,470 iterations |
| **Nested target encoding + frequency + floor flags** | **0.94565** | **+0.00346 over baseline, paired bootstrap P = 1.000** |
| Realistic top of the field | ~0.94674 | My gap: +0.00109 |
| Reported top-50 threshold | ~0.94652 | |
| Public LB leader | 0.94945 | Anomalous: +0.0027 over 2nd place, presumed leak or LB overfit |
| Best *semantic* model achievable | ~0.9378 | Measured, see "The ceiling" below |

> **How to read these numbers.** Every figure marked OOF is out-of-fold on a
> frozen `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` split, so
> each row is scored by models that never saw it. Fold standard error on a
> single run is ~0.0004, which means **an unpaired gap smaller than ~0.001 is
> noise.** All deltas below are paired on identical folds, which resolves
> effects down to ~0.0002.

---

## The rules this repository follows

These four rules are the reason the numbers above are trustworthy, and they are
enforced in the code, not just stated:

1. **One frozen split.** Every experiment uses the same folds. A result that
   needs a different split to look good is not a result.
2. **Paired deltas only.** Every change is measured against a named parent on
   identical folds, with a paired bootstrap. Unpaired comparisons are not
   recorded.
3. **A label-shuffle canary for every target-derived feature.** Target encoding
   leaks if you get it even slightly wrong, and a leak looks exactly like
   success. So any pipeline that touches the target is re-run with shuffled
   labels; it must score ~0.500. Anything above 0.52 means a leak.
4. **Negative results are kept, not deleted.** Knowing what fails is the
   deliverable. See "Dead ends" below.

### Canary log

| Pipeline | Shuffled-label OOF | Pass |
|---|---|---|
| Raw features | 0.50174 | yes |
| Nested target encoding | 0.50149 | yes |

Both sit on 0.500 as they must. The encoding is clean.

---

## The ceiling — the result I actually care about

The competition describes a generating rule roughly like:

```
1.2·income/1e5 + 0.6·concern + 2·subsidy − 1·(anxiety=Med) − 3·(anxiety=High) + N(0,1) > 5.5
```

Scoring that rule directly on the competition data gives AUC 0.93769 — but its
**analytic ceiling is 0.91482**, which the 0.94219 baseline already beats. So
the stated noise term cannot be `N(0,1)`.

Refitting a free-scale probit gives `income +1.4095, concern +0.7358,
subsidy +2.3410, anx_med −1.0172, anx_high −2.9972, const −6.7148`, implying
**sigma = 0.851**. That fitted probit scores 0.93777 against its own analytic
ceiling of 0.93689 — self-consistent, so the fit is right.

**The consequence:** the best any genuinely semantic model can do here is about
0.9378. My 0.94565 is +0.00788 above that. Nine of the thirteen features have
|correlation with the Bayes residual| < 0.017 and are near-worthless alone. The
top of this leaderboard is therefore a contest in exploiting synthetic-data
artifacts — specifically the value-level structure left by the generator — not
in modelling EV buyers.

That is worth knowing before spending another week on it, and it is the kind of
thing a scoreboard alone will never tell you.

---

## Dead ends, measured

Recorded so nobody repeats them, including me:

| Idea | Evidence | Verdict |
|---|---|---|
| `id` row-order leak | AUC 0.49999, ids monotone | dead |
| Duplicate rows train↔test | 0 shared feature-rows | dead |
| `income % 1000` binary flag | Identical to the 30,000 floor rows | redundant |
| `income % 1000` continuous | Standalone AUC 0.52706; 0.48326 off-floor | weak |
| `income_in_original` flag | Standalone AUC 0.49851 | dead |
| Original-label lookup by income | Standalone AUC 0.51609 | weak |
| Concatenating the 10k original rows | Lost 0 of 5 folds in published ablation | skip |

---

## What the data looks like

From `reports/01_eda_findings.md`, measured on the real `train.csv`
(668,665 rows) rather than taken from public write-ups:

- 13 features plus `id` and target `Will_Buy_EV`. Positive rate **0.174645**.
- Zero nulls in train or test.
- `sample_submission.csv` is prefilled with 0.17464500160768098 — the exact base
  rate, which is a small tell about how the data was made.
- The only real encoding targets are the high-cardinality continuous columns;
  the rest are low-cardinality categoricals.

---

## Reproduce

Requires Python 3.10+. No GPU needed.

```bash
git clone https://github.com/apoorva2121/ev-purchase-s6e9.git
cd ev-purchase-s6e9
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Get the data.** It is not in this repository (Kaggle competition data should
not be redistributed). Download it with the Kaggle CLI and place it as shown:

```bash
pip install kaggle                      # needs ~/.kaggle/kaggle.json
kaggle competitions download -c playground-series-s6e9 -p data/raw
cd data/raw && unzip playground-series-s6e9.zip && cd ../..
```

`data/raw/` must end up containing:

```
train.csv
test.csv
sample_submission.csv
EV_Adoption_and_Range_Anxiety_Dataset.csv     # the original dataset this was generated from
```

The last file is the source dataset on Kaggle
("EV Adoption and Range Anxiety"); `src/common.py` reads it to test whether the
original labels help. They do not — see Dead ends.

**Run it:**

```bash
python src/eda_artifacts.py      # reproduces reports/01_eda_findings.md
python src/baseline_lgbm.py      # the 0.94219 baseline
python src/batch3b.py            # the nested target-encoding experiments
python src/dgp_analysis.py        # the ceiling argument above
```

Expect ~4 minutes for the baseline on a modern laptop CPU.

## Layout

```
src/
  common.py            data loading, target encoding, fold helpers
  cv.py                the frozen split, used everywhere
  baseline_lgbm.py     the 0.94219 reference model
  batch2–batch6.py     experiment batches, each a paired test vs a named parent
  nested_te.py         out-of-fold nested target encoding
  dgp_analysis.py      probit refit and the analytic ceiling
  data_integrity.py    leak checks (duplicates, id ordering)
  diagnostics.py       canary runs with shuffled labels
  error_analysis.py    where the model is wrong
  explainability.py    feature attribution
  mlp.py, mlp_embed.py neural baselines with entity embeddings
reports/
  01_eda_findings.md   measured EDA
  02_scoreboard.md     every experiment, every delta, every dead end
  03_backlog.md        what was left untried
artifacts/             out-of-fold prediction arrays (.npy), not in git
```

`artifacts/` holds one `.npy` of out-of-fold predictions per experiment, which
is what makes the paired tests possible after the fact. They are regenerated by
the scripts and excluded from git.

## What I would do next

`reports/03_backlog.md` has the full list. The honest summary: the remaining
+0.001 to the top of the field is artifact mining, so the better use of the time
is elsewhere. I stopped deliberately.

## Licence

[MIT](LICENSE). The competition data is **not** included and remains subject to
Kaggle's competition rules.
