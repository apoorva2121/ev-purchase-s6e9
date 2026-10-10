# S6E9 — full idea backlog, with status

Every idea considered this project. `DONE` = measured, number given. `RUNNING`/`QUEUED` = in flight.
`DEAD` = ruled out by a measurement, with the measurement. Reference: lgb42 honest-ES OOF 0.94574,
exp05 LB 0.94615. Paired SD 0.00008, so the accept bar is +0.00016 on both splits.

## A. Encoding of the two high-cardinality columns  (the only thing that has ever worked)
| Idea | Status | Result |
|---|---|---|
| Nested per-exact-value TE, m=5/20/100 | DONE | **+0.00346** — the whole competition |
| Binned TE 250/1000/5000 added to exact | DONE | +0.00012 / +0.00010 on two splits — unproven, kept |
| Binned TE *instead of* exact | DONE | −0.00150 |
| Smoothing sweep m ∈ {1,500} | DONE | −0.00000 |
| Hierarchical shrink toward bin-250 prior | DONE | +0.00002 |
| WOE / logit-space credibility shrinkage | DONE | +0.00001 |
| LightGBM native categorical on income | DONE | −0.00126 |
| CatBoost ordered target statistics, no hand TE | DONE | −0.00260 |
| Exact-pair TE (income × subsidy/concern/anxiety) | DONE | −0.00026 |
| Centred-window rates, widths 2–200 (windows only) | DONE | 0.94564 (−0.00010) |
| Centred-window rates + exact TE | RUNNING | — |
| Quantisation ladder inc/10, /50, /500 | QUEUED | — |
| Semi-supervised TE (test rows, soft labels) | QUEUED | the one that adds data |
| TE on **all 13 columns** (published recipe pairs this with depth 5) | QUEUED | — |
| Test-side TE fitted on 100% of train | QUEUED | step1 crashed, needs fix |
| 10-fold outer (90% per TE fit instead of 80%) | QUEUED | — |
| More inner folds (64% → 72% for training rows) | TODO | — |
| TE bagged over several inner seeds | TODO | — |

## B. Features
| Idea | Status | Result |
|---|---|---|
| Frequency / count encoding | DONE | in baseline |
| Floor indicators (income 30000, commute 5.0) | DONE | in baseline |
| Distance to nearest original income | DONE | in baseline |
| Ratios, products, sums, differences | DONE | −0.00006 |
| Group aggregations, deviation-from-group, z-scores | DONE | −0.00009 |
| Source-table count, lift (0.5–541x), novelty flags | DONE | +0.00001 |
| Recovered probit score as a column | DONE | 0.00000 |
| KMeans cluster id + centroid distances | DONE | −0.00012 |
| kNN out-of-fold target mean | DONE | −0.00012 |
| Partial source-row matching (n columns matched) | DONE | +0.00001 |
| One-hot low-cardinality | DONE | −0.00002 |
| Label vs native categorical | DONE | −0.00001 |
| Digit-power features (each decimal digit) | QUEUED | — |
| Floors re-encoded as NaN (they are mode-fill placeholders) | QUEUED | — |
| IsolationForest + kNN-distance outlier scores | QUEUED | — |
| PCA / rotations | DEAD | the *optimal* diagonal (probit score) gave 0.00000 |
| Model-based imputation of the floor rows (MICE etc.) | DEAD | income R²=0.016 from other cols — imputer returns the mean |
| Marking hard/uncertain rows | DEAD | function of y; unavailable at test |

## C. Models and tuning
| Idea | Status | Result |
|---|---|---|
| LightGBM depth 5, colsample 0.30, lr 0.02 | DONE | baseline |
| XGBoost / CatBoost / HistGradientBoosting | DONE | 0.94571 / 0.94570 / 0.94556 |
| MLP, 5 seeds | DONE | 0.94429 |
| MLP with 13,214-entry income embedding | DONE | 0.94429, rank-corr 0.991 — most diverse, still no blend weight |
| max_bin 255 vs 1024 | DONE | 0.00000 — settles a question nobody had isolated |
| min_data_in_leaf 50 / 200 | DONE | −0.00002 / −0.00000 |
| learning rate 0.01 | DONE | +0.00002 |
| Interaction constraints | DONE | +0.00004 |
| GOSS / linear_tree / DART | DONE | −0.00013 / −0.00001 / −0.00045 |
| cat_smooth, min_data_per_group, max_cat_threshold | DONE | +0.00001 |
| Monotone constraints, extra_trees | DEAD | published −0.0015 / −0.0059 |
| scale_pos_weight, SMOTE | DEAD | published −0.002 to −0.01 for AUC |
| Optuna on the frozen split | DEAD | would compound selection overfitting; dropped on review |

## D. Ensembling
| Idea | Status | Result |
|---|---|---|
| 10-seed bag, equal rank weights | DONE | +0.00003; seed sd is 0.00001 — no variance to reduce |
| Equal-weight 6-member cross-library | DONE | OOF +0.00005 → **LB −0.00003** |
| Equal-weight 12-member | DONE | LB 0.94612 |
| Hill-climbed weights on OOF | DEAD | fits OOF noise; dropped on review |
| 56-member logistic stack | DEAD | OOF +0.00019, LB −0.00004; offset +0.00013 vs population +0.00032 — 3 sd off |

## E. Post-processing and strategy
| Idea | Status | Result |
|---|---|---|
| Deterministic-region clamps (4 regions, 3,733 test rows) | DONE | +0.00001 |
| Public-split rank-band probing (what the "top 38" does) | REFUSED | fits the public 20%; provably useless on private |
| Final pick = max(A,B) on private → pick for diversity | NOTED | changes the selection rule |

## F. Validation and process
| Idea | Status |
|---|---|
| Frozen split, paired bootstrap (sd 0.00008) | DONE |
| Shuffled-label canary, ES disabled | DONE — 0.50174 / 0.50149 / 0.49962 |
| Second split (seed 7) for every conclusion | DONE |
| Honest ES (rounds from a train slice, then refit) | DONE — bias was +0.00005 |
| Adversarial validation | DONE — 0.49940, transductive features licensed |
| Null importance, permutation importance, IV | DONE — only `Gender` is dead |
| Unit test: encoder vs independent recomputation | DONE — exact to 1e-12, no own-label dependence |
| Data-integrity / garbage-value audit | RUNNING |
