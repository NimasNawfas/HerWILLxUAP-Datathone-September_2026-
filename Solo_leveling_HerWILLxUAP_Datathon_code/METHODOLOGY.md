# Methodology

| component                             | architecture                                                                | params                                               | folds                | hyperparams                                                                                                                                                   |
|:--------------------------------------|:----------------------------------------------------------------------------|:-----------------------------------------------------|:---------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TF-IDF + LinearSVC (Platt-calibrated) | word 1-2g + char_wb 3-5g TF-IDF (≤120,000 feats), linear SVM                | ≈0.4M weights x3 calibrated copies                   | 5                    | C=0.1, class_weight=balanced, sigmoid calibration cv=3                                                                                                        |
| XLM-RoBERTa-large                     | xlm-roberta-large + classification head                                     | 560M total (256M embeddings, 303M layers, 1.1M head) | 4                    | epochs=4, batch=16x2, max_len=128, lr(top)=2e-05, LLRD=0.95, head_lr x2.0, wd=0.01, warmup=0.08, label_smoothing=0.05, EMA=None, fp16=True, noise-aware=False |
| MuRIL-base-cased                      | google/muril-base-cased + classification head                               | 238M total (152M embeddings, 85M layers, 0.6M head)  | 5                    | epochs=3, batch=32x1, max_len=128, lr(top)=3e-05, LLRD=0.9, head_lr x2.0, wd=0.01, warmup=0.08, label_smoothing=0.05, EMA=0.999, fp16=True, noise-aware=True  |
| mDeBERTa-v3-base                      | microsoft/mdeberta-v3-base + classification head                            | 279M total (193M embeddings, 85M layers, 0.6M head)  | 5                    | epochs=3, batch=32x1, max_len=128, lr(top)=3e-05, LLRD=0.9, head_lr x2.0, wd=0.01, warmup=0.08, label_smoothing=0.05, EMA=0.999, fp16=True, noise-aware=True  |
| Stacker                               | multinomial LogReg / HistGradientBoosting on log-probs + text meta-features | a few hundred coefficients / 250 depth-3 trees       | 5 (CV over OOF rows) | best=stack_hgb, per-class thresholds by grid search (nested-CV validated)                                                                                     |

## Validation protocol
* Every model is trained with K-fold cross-validation over all labeled rows; each row's prediction comes from a model that never saw it (out-of-fold, OOF).
* The stacker and the per-class thresholds are tuned only on these OOF predictions, and the reported estimates are nested-CV (thresholds chosen on
  some rows, scored on held-out rows), so they are not tuned on the rows they are scored on.
* Test predictions are the average over the fold models of each backbone.

## Text meta-features (stacker input)
Log character length, log word count, script indicators (Bangla / Latin / mixed / other), counts of `!` and `?`, upper-case ratio, emoji count,
and URL / @user flags. No feature is derived from the `id` column.

## Add-ons
* **Exact-duplicate text lookup:** ≈3% of test comments have an identical text (after lower-casing and stripping punctuation, URLs and @mentions) in train.
  On train twins the majority twin label was more accurate than the model (0.875 vs 0.790), and applying it improved OOF Macro-F1 by ≈ +0.002, so it is applied.
* **Per-script thresholds (Bangla vs Latin):** tested and rejected; nested-CV Macro-F1 went from 0.5999 to 0.5960 (−0.0039).
