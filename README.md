# F1 Pit Stop Duration Prediction

Predicting Formula 1 pit stop duration using machine learning on 17 seasons of historical data (1994–2010).

**Author:** Saurabh Gupta — IIT (ISM) Dhanbad, B.Tech Mineral & Metallurgical Engineering  
**GitHub:** [Saurabh6266](https://github.com/Saurabh6266)

---

## Problem Statement

Can machine learning predict how long an F1 pit stop will take, using only information available before the stop — the team, driver, circuit, stop number, lap, and historical performance averages?

This project builds an end-to-end ML pipeline on the [Pitstop Pulse dataset](https://www.kaggle.com/datasets/eshummalik/pitstop-pulse-formula-1-performance-data) to answer that question across two prediction tasks:

- **Task A — Regression:** Predict exact pit stop duration in seconds
- **Task B — Classification:** Classify a stop as fast, normal, or slow

---

## Dataset

- **Source:** Kaggle — Pitstop Pulse: Formula 1 Performance Data
- **Coverage:** 9,921 pit stop events across 288 races (1994–2010)
- **Columns:** Race, Driver, Constructor, Lap, Stop number, Duration (Time), Cumulative time (Total)
- **Key structural feature:** Refueling was permitted from 1994–2009 (stops averaging ~30s, range 8–120s), banned from 2010 onward (tyre-only stops averaging ~25s).

Download `pitstops.csv` from Kaggle and place it in the project root before running.

---

## Project Structure
```
f1-pitstop-duration-prediction/
├── f1_pitstop_prediction.ipynb   # Main notebook (all sections)
├── README.md
├── requirements.txt
└── .gitignore
```
---

## Notebook Sections

| Section | Description |
|---|---|
| 1 — Data Loading | Shape, dtypes, initial inspection |
| 2 — Time Parsing | Convert string durations to float seconds |
| 3 — Leakage Audit | Identify and document `Total_sec` and `TimeOfDay` as leakage |
| 4 — Preprocessing | Drop nulls, trim extreme outliers (>120s) |
| 5 — EDA | Era split, year trend, stop number patterns, constructor analysis |
| 6 — Feature Engineering | 12 features including rolling historical averages |
| 7 — Train/Val/Test Split | Time-aware split by season year |
| 8 — Classification Targets | Binary and 3-class targets from training quantiles |
| 9 — Baselines | Mean, median, constructor-average |
| 10 — Regression (Task A) | Linear → Ridge → Lasso → KNN → DT → RF → GBM → XGBoost → LightGBM → MLP |
| 11 - Era-Specific Regression | For refueling era: train on 1994-2007, val on 2008, test on 2009 |
| 12 — Hyperparameter Tuning | RandomizedSearchCV with TimeSeriesSplit |
| 13 — SHAP Interpretability | Summary, waterfall, dependence, SHAP vs Gini comparison |
| 14 — Error Analysis | Residual plots, worst predictions, learning curves |
| 15 — Model Stacking | RF + XGBoost + LightGBM with Ridge meta-learner |
| 16 — Classification (Task B) | 9 classifiers, ROC curves, confusion matrices, 3-class XGBoost, calibration |
| 17 — Final Summary | Full leaderboards and comparison charts |
| 18 — Conclusions | Key findings, performance summary, model limitations |
| 19 — Future Work | Fuel load data, Ergast API merge, two-stage modelling |

---

## Results

| Task | Best Model | Metric | Score |
|---|---|---|---|
| Regression (Task A) | Gradient Boosting (sklearn) | R² | 0.019 |
| Regression (Task A) | Gradient Boosting (sklearn) | RMSE | 4.743s |
| Binary Classification (Task B) | SVM (RBF) | AUC-ROC | 0.644 |
| Binary Classification (Task B) | SVM (RBF) | Accuracy | 83.4% |
| 3-Class Classification (Task B) | XGBoost | Macro F1 | 0.49 |

> **Note:** R² values are constrained by covariate shift across the refueling-era boundary (test set is 84% fast stops vs 50% in training) and the absence of fuel load data, which was the primary determinant of stop duration in the refueling era. Era-specific evaluation on 2009-only data (no distribution shift) yields substantially better performance.

---

## Key Findings

- The refueling era boundary (pre-2010 vs 2010) is the dominant structural feature, creating a clear regime shift in stop duration that the model must bridge.
- sklearn Gradient Boosting achieved the best regression R² (0.019), outperforming XGBoost and LightGBM — an unusual result explained by early stopping on a distribution-shifted validation set causing the XGBoost variants to underfit.
- Within each era, constructor historical average outranks driver history in SHAP importance — pit stops are a crew operation, not a driver operation.
- Distribution shift between the refueling era (training) and no-refueling era (test) limits achievable R² without fuel load data, which was the primary determinant of stop duration in the refueling era.
- High classification accuracy (SVM: 83.4%) with low AUC-ROC (0.64) is a textbook signature of class imbalance — 84% of the test set is "fast" stops, making accuracy a misleading metric.
- SHAP and Gini importance disagree most on high-cardinality encoded features (circuit_encoded, driver_hist_avg), where SHAP provides the more reliable ranking.

---

## Setup

```bash
git clone https://github.com/Saurabh6266/f1-pitstop-duration-prediction
cd f1-pitstop-duration-prediction
pip install -r requirements.txt
# Download pitstops.csv from Kaggle and place it here
jupyter notebook f1_pitstop_prediction.ipynb
```

---

## References

- Dataset: [Pitstop Pulse on Kaggle](https://www.kaggle.com/datasets/eshummalik/pitstop-pulse-formula-1-performance-data)
- Inspiration: [Before the Lights Go Out — UC Berkeley ML Project](https://sites.google.com/berkeley.edu/team15-predict-f1-race)
- Lundberg & Lee (2017) — [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) (SHAP paper)
- Bergstra & Bengio (2012) — [Random Search for Hyper-parameter Optimization](https://jmlr.org/papers/v13/bergstra12a.html)
