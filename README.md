# MSDS 451 — Programming Assignment 1
## Predicting next-day return direction for Palantir (PLTR)

This repository contains my submission for **MSDS 451 Financial Engineering, Programming Assignment 1** . I replicate the WTI jump-start methodology on a different asset — Palantir Technologies (ticker PLTR) — that I am personally interested in as an investor.

The pipeline:

1. Pull daily OHLCV data for PLTR from Yahoo! Finance (`yfinance`).
2. Engineer 15 lag/EMA-based features in Polars.
3. Run all-possible-subsets logistic regression and pick a feature subset by AIC.
4. Train an XGBoost classifier under five-fold `TimeSeriesSplit` cross-validation (10-day gap).
5. Tune hyperparameters via `RandomizedSearchCV` (100 iterations).
6. Report ROC AUC, confusion matrix, classification report, and feature importance.

---

## Repository contents

| File | Purpose |
| --- | --- |
| `getdata_pltr.py` | Downloads PLTR daily prices from Yahoo! Finance and writes `msds_getdata_yfinance_pltr.csv`. |
| `pltr_pa1_analysis.py` | Full analysis pipeline: feature engineering → AIC subset selection → XGBoost tuning → evaluation. Produces all figures. |
| `msds_getdata_yfinance_pltr.csv` | Raw daily OHLCV for PLTR (2020-09-30 → 2026-04-19). |
| `pltr-with-computed-features.csv` | Engineered-feature DataFrame, written as a checkpoint by the analysis script. |
| `analysis_output.log` | Captured stdout from the most recent analysis run. |
| `figures/` | PNG plots: correlation heat map, ROC curve, confusion matrix, feature importance. |
| `PLTR_PA1_research_report.docx` | Research report (Word). |
| `PLTR_PA1_research_report.pdf` | Research report (PDF). |

The original course materials referenced in the assignment brief are also included:
`451_pa1_jump_start_v001.{ipynb,py}`, `451_pa1_technical_report_v002.pdf`, `getdata_yfinance.{ipynb,py}`, and `msds_getdata_yfinance_aapl.csv`.

---

## Requirements

- Python 3.9 or newer
- Packages: `yfinance`, `polars`, `pyarrow`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `scipy`, `python-docx`

Install everything with:

```bash
python3 -m pip install --user yfinance polars pyarrow numpy scikit-learn xgboost matplotlib seaborn scipy python-docx
```

---

## How to run

From the repository root:

```bash
# 1. (Optional) re-download PLTR daily prices
python3 getdata_pltr.py

# 2. Run the full feature-engineering and modeling pipeline
python3 pltr_pa1_analysis.py

# 3. Rebuild the Word research report from the figures
python3 build_report.py
```

`pltr_pa1_analysis.py` runs end to end in a couple of minutes on a modern laptop. The all-possible-subsets stage evaluates 32,767 logistic regressions; this is the slow step.

The script prints progress to stdout and writes:

- `pltr-with-computed-features.csv` — the engineered DataFrame
- `figures/pltr_correlation_heatmap.png`
- `figures/pltr_roc_curve.png`
- `figures/pltr_confusion_matrix.png`
- `figures/pltr_feature_importance.png`

---

## Headline results

- **Selected feature subset (top frequencies in the 10 lowest-AIC subsets):** `HMLLag2`, `HMLLag1`, `OMCLag3`, `VolumeLag2`, `OMCLag1`. Notably, none of `CloseLag1–3` and none of the EMAs survived selection — a different mix than the WTI reference subset.
- **Baseline XGBoost five-fold time-series CV accuracy:** 0.506 ± 0.036 (essentially the majority-class rate).
- **Tuned XGBoost CV accuracy:** 0.521.
- **In-sample ROC AUC after refitting on the full data:** 0.997 — clear evidence of training-set overfit, not held-out skill.
- **In-sample confusion matrix:** 669 TN / 17 FP / 21 FN / 683 TP (97.3% in-sample accuracy).

The training-vs-CV gap is itself the headline finding: XGBoost can memorize this dataset, but the lagged price/volume features provide little reliable directional signal for PLTR. The research report discusses next steps (non-price features, three-level targets, true forward holdout, related tickers).

---

## Reproducibility notes

- All random-state-dependent code uses `random_state = 2025` (matching the jump-start).
- The data window is fixed in `getdata_pltr.py` (`START_DATE = "2020-09-30"`, `END_DATE = "2026-04-19"`); change those constants and re-run if you want a different window.
- `polars` and `xgboost` versions tested: `polars==1.36.1`, `xgboost==2.1.4`.
