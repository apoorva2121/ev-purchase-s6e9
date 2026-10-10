# S6E9 — first-hand EDA findings (2026-09-26)

Run: `src/eda_artifacts.py`. All numbers measured on the real `train.csv`
(668,665 rows), not taken from public writeups.

## Confirmed spec
- 13 features + `id` + target `Will_Buy_EV` (Yes/No). Positive rate **0.174645**.
- Zero nulls in train or test.
- `sample_submission.csv` is prefilled with 0.17464500160768098 = the exact base rate.

## Schema, by cardinality
High-cardinality continuous (the only real encoding targets):
| col | distinct (train) | distinct (test) |
|---|---|---|
| `Annual_Income_USD` | 13,214 | 10,526 |
| `Daily_Commute_km` | 805 | 781 |

Low-cardinality — everything else:
`Age` 45, `Charging_Stations_Near_Work` 20, `Charging_Stations_Near_Home` 15,
`Environmental_Concern_Level` 5 (values 1.0–5.0, no non-integers),
`Number_of_Cars_Owned` 4, `Current_Car_Type` 4 (Hatchback/SUV/Sedan/Truck),
`Gender` 3 (Female/Male/Other), `City_Type` 3 (Rural/Suburban/Urban),
`Range_Anxiety_Level` 3 (High/Low/Medium), `Home_Charging_Possible` 2,
`Subsidy_Available` 2.

## FINDING 1 — both continuous features are left-censored at a floor, and the floor is a strong signal

| col | floor | rows at floor | label rate at floor | label rate off floor |
|---|---|---|---|---|
| `Annual_Income_USD` | 30,000.0 | 61,605 (9.21%) | **0.04834** | 0.18820 |
| `Daily_Commute_km` | 5.0 | 144,280 (21.58%) | **0.18443** | 0.17195 |

The income floor is the single biggest effect found: being at exactly 30,000
drops P(buy) to about a quarter of the base rate. The generator clipped a
left tail into a point mass, so the floor group is a *mixture* of distinct
underlying populations and behaves nothing like its neighbours at 30,001.

**Action:** explicit `is_income_floor` / `is_commute_floor` indicator features,
and exclude the floor rows when fitting any smooth/binned encoding over income
so they don't contaminate the lowest bin.

## FINDING 2 — "income roundness" is NOT an independent signal (contradicts public repos)

Several public S6E9 repos report gains from roundness / digit features. Measured here:

- `income % 1000 == 0` → frac 0.0922, label rate 0.04435
- rows at the 30,000 floor → frac 0.0921

These are **the same rows**. Roundness is the floor indicator wearing a disguise,
not a second discovery. Adding both is redundant.

- Income digit count: 5 digits → 0.13058, 6 digits → 0.29182. This is just a
  threshold at income = 100,000, which any tree finds unaided. Not magic.

## FINDING 3 — no duplicate or leak surface at all

- Exact duplicate feature-rows within train: **0**
- Within test: **0**
- Feature-rows shared between train and test: **0**

Kills the whole duplicate-matching / train-test-overlap class of exploits.
Cheap, definitive negative result — do not spend time here.

## FINDING 4 — commute is 70.7% non-integer
Non-integer label rate 0.17391 vs integer 0.17642. The integer group is mostly
the 5.0 floor, so this is largely Finding 1 again. Weak as a standalone feature.

## Structural consequence
Only two features have enough cardinality to justify target encoding. The other
eleven are small enough that their *interactions* are the natural encoding
targets instead — GBDTs get single low-card splits for free, so any gain there
has to come from combinations.
