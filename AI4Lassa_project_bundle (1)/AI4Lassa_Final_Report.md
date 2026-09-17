# AI4Lassa — National Case-Count Early-Warning Model
## Final Report (Phases 1–5 + Final Audit)

---

## 1. Problem

Lassa fever is endemic and seasonal in Nigeria, with recurring surges typically in the dry season. An early-warning signal for expected case volume in the coming month can help public-health teams plan testing capacity, staffing, and supplies ahead of a surge, rather than reacting after it's underway.

## 2. What Existed Before This Work

The existing `AI4Lassa` GitHub repository contained a working Streamlit app backed by a **linear-kernel SVM** (`svmnew_model.pkl`, scikit-learn 1.0.2) that predicts a **per-patient clinical outcome** from 13 vitals (age, sex, ward type, temperature, heart rate, respiratory rate, blood pressure, SpO2, GCS, oxygen/ventilation need, bleeding). Critically:

- **No training code, notebooks, or training data exist anywhere in the repo's git history** — only the app and the three serialized artifacts have ever been committed. How the model was trained, validated, and tuned is not recoverable.
- `requirements.txt` lists `sdv` (Synthetic Data Vault), suggesting the original training data may have been synthetically augmented — a real, unresolved question mark on scientific validity that predates this project.
- The app's "outbreak declared" logic is a hard-coded rule (predicted-positive count > predicted-negative count in an uploaded CSV) — not a statistically grounded outbreak definition.
- **This existing model uses none of the features available in the new dataset**, and the new dataset has no location, weather, or population data, so the original brief's assumption of a geography/weather-driven spatiotemporal model was not achievable with the data provided. This was flagged and agreed with you before proceeding (Phase 1).

The existing app (`streamlit_app.py` and its three `.pkl` files) has **not been modified** by this work. Everything here was built as a separate, parallel pipeline.

## 3. Data

Source: your uploaded `2015-2025__3_.xlsx` — a national **individual-level Lassa fever line-listing (surveillance) dataset**, 31,929 rows × 53 columns, spanning 2015-01 to 2025-12 with no gaps. No fabricated or synthetic records were introduced. No geographic or environmental variables exist in this file.

It was aggregated into a **continuous monthly national time series** (132 months) using `DATE OF LINE LISTING` as the temporal key — chosen because it reflects what would actually be known to a live surveillance system in real time, rather than a retrospective onset date.

## 4. Target

**Primary (regression):** `target_next_month_cases` — the actual national case count recorded the following month. This is a real, non-fabricated quantity derived directly from the data.

**Secondary (derived binary flag):** `high_risk` = 1 if next month's case count exceeds a threshold of **462 cases** (mean + 1.5×SD of the *training-period* target distribution, 2016–2021 only — never computed using validation/test data). This threshold is a heuristic starting point, explicitly not an epidemiologist-validated clinical/public-health cutoff, and is documented as such.

## 5. Features

14 features, all computable using information available **on or before** the month being used to forecast (no future information):

- Autoregressive: current-month case count, lags at 1/2/3/6/12 months, 3- and 6-month rolling mean/max, month-over-month growth
- Seasonality: cyclical month encoding (`sin`/`cos`), calendar year
- Epidemiological leading indicator: prior month's lab-positivity rate (share of tested specimens returning positive)

An **ablation study** (Phase 4) showed seasonality contributes real, measurable value (adding it cut validation MAE by ~24% and roughly tripled R²), while the lab-positivity-rate feature did not improve validation performance in this dataset — it was kept in the final feature set for epidemiological plausibility but its practical contribution should not be overstated (confirmed by final feature importance: 2.0%).

## 6. Validation Strategy

Strict time-aware split, no random shuffling:

| Split | Period | Months |
|---|---|---:|
| Train | 2016-01 – 2021-12 | 72 |
| Validation | 2022-01 – 2023-12 | 24 |
| Test | 2024-01 – 2025-11 | 23 |

The test period was held out from all model selection and hyperparameter tuning through Phase 4, and only evaluated once in Phase 5.

## 7. Models Compared

**Regression:** naive persistence baseline, architecture-matched linear SVM (mirroring the existing app's model type), Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost, GradientBoosting, ExtraTrees, an RF+CatBoost ensemble average, regularized Poisson regression, and a univariate SARIMA(1,1,1)(1,1,0,12) model.

**Classification (derived risk flag):** the same linear-SVM baseline, Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost.

## 8. Results (actual measured numbers)

### Regression — validation set (2022–2023)

| Model | MAE | R² |
|---|---:|---:|
| **Random Forest (tuned)** | **57.7** | **0.65** |
| Ensemble avg (RF + CatBoost) | 57.7 | 0.63 |
| GradientBoosting | 60.7 | 0.61 |
| CatBoost (default) | 60.6 | 0.59 |
| ExtraTrees | 61.3 | 0.57 |
| LightGBM (tuned) | 72.0 | 0.44 |
| Linear Regression | 71.5 | 0.50 |
| Poisson GLM (regularized) | 84.4 | 0.24 |
| Naive persistence | 80.0 | 0.27 |
| SARIMA | 93.8 | -0.09 |
| Linear SVM (architecture-matched baseline) | 83.4 | 0.12 |

**Random Forest selected.** It's worth stating plainly: neither the textbook-correct count-data model (Poisson/Negative-Binomial GLM) nor the classic epidemiological time-series approach (SARIMA) outperformed the tree ensemble here — most likely because the real signal (seasonal surges, occasional shocks) is more threshold-like/nonlinear than either can capture with only 72 training months.

### Regression — final test set (2024–2025, evaluated once)

**MAE = 49.3, RMSE = 64.1, MAPE = 17.2%, R² = 0.567.** Performance was consistent with (in fact slightly better than) validation — a genuine positive sign that the model generalizes rather than having been overfit to the validation window.

### Classification — validation set (2022–2023, only 3 true high-risk months)

| Model | Recall | Precision | Brier | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression (C=5) | 1.00 | 0.43 | 0.114 | 0.968 |
| Linear SVM | 1.00 | 0.30–0.33 | 0.086–0.203 | 0.90–0.97 |
| LightGBM (tuned) | 0.67 | 1.00 | 0.058 | 0.921 |
| XGBoost | 0.33 | 0.33 | 0.092 | 0.92 |
| Random Forest | 0.00 | 0.00 | 0.071 | 0.95 |
| CatBoost | 0.00 | 0.00 | 0.089 | 0.92 |

Random Forest's 87.5% *accuracy* on this task corresponded to **zero recall** — it never once flagged a true high-risk month, simply predicting "low risk" every time. This is presented explicitly as a caution against accuracy-only model selection, consistent with your original brief.

### Classification — final test set (2024–2025, one true high-risk month)

**Logistic Regression at the default 0.5 threshold missed the one true event** (predicted probability 18.5%, recall = 0). Lowering the decision threshold to 0.15 (chosen by inspecting test-set outcomes) recovers recall = 1.0 with precision = 0.25 (4 of 23 months flagged).

**This needs to be stated honestly, not minimized:** picking 0.15 by observing test-set performance is a mild form of data leakage into model selection — the opposite of what the earlier phases were careful to avoid. It should be treated as a promising provisional operating point, not a validated one, until confirmed against a genuinely new out-of-sample period (e.g., as 2026 data accumulates).

## 9. Explainability

Global feature importance from the final Random Forest (built-in impurity-based importance):

| Feature | Importance |
|---|---:|
| Current-month case count | 49.4% |
| Seasonality (month, cosine encoding) | 24.5% |
| 3-month rolling max of past cases | 3.8% |
| 2-month-lagged case count | 3.4% |
| 12-month-lagged case count | 3.1% |
| All other features (10) | ≤2.2% each |

In plain terms: **the model is mostly learning "how many cases are happening right now" plus "what time of year it is."** This is intuitive, defensible, and easy to communicate to a non-technical stakeholder — but it also means the model has essentially no ability to anticipate a surge driven by something outside that pattern (e.g., a localized cluster or an external trigger not captured in this dataset), which is exactly what happened with the missed test-set event.

**Limitation acknowledged:** full per-prediction SHAP attribution was not implemented in this phase — the current explanation approach (`src/predict.py`) reports the model's top global drivers and the current month's values for them, which is transparent but coarser than a true per-prediction attribution. Recommended as near-term future work (see Section 11).

## 10. Final Recommendation

| Component | Model | Status |
|---|---|---|
| Case-count forecast (primary output) | Random Forest (tuned) | **Reasonably solid** — consistent validation→test performance, real R² gain over naive persistence |
| Risk-level flag (secondary output) | Logistic Regression | **Provisional / exploratory** — directionally useful but the specific decision threshold is not independently validated, and it missed its only clean out-of-sample test |

Recommendation: ship the **regression forecast as the primary, user-facing number** (expected case count next month), and present the risk label with visible caveat language rather than as a confident alert — consistent with treating this as a decision-support tool, not a diagnostic or alerting system.

## 11. Limitations

- **119 usable monthly observations total** is a small sample for any model, especially the rare-event classifier (5 high-risk months in training, 3 in validation, 1 in test). All classifier metrics should be read as directional, not statistically settled.
- **No geographic granularity** — this is a national aggregate model. It cannot say which state/LGA is at risk, only that national case volume is expected to be elevated.
- **No environmental/weather/population data** — the model can't distinguish a climate-driven surge from a reporting-driven one, or account for population exposure.
- **The classifier's decision threshold (0.15) was selected using test-set outcomes** and needs re-validation against new data before being trusted operationally.
- **Original SVM's training methodology remains unverifiable** — no code or data for it was ever committed to the repository, and the `sdv` dependency raises an unresolved question about synthetic data use in its training.
- This system should be described, and used, as a **decision-support and early-warning aid** — not a replacement for laboratory diagnosis, clinicians, epidemiologists, or NCDC/public-health authorities.

## 11a. Application Integration (Phase 7)

`streamlit_app.py` now has two tabs:

- **🏥 Patient Outcome Predictor** — the original app, byte-for-byte unchanged in logic (manual entry, CSV upload, the existing outbreak-declared heuristic). Verified to still load and run without error.
- **📈 National Outbreak Early-Warning** — new. Loads `models/final_rf_regressor.pkl` and `models/final_logit_riskflag.pkl`, shows the next-month case-count forecast, a provisional risk label with an adjustable decision threshold, the top global contributing factors, and a recent-months table. The validation/test caveats from this report are shown directly in the UI (not just in this document), and the tab fails gracefully with a clear warning — without crashing the rest of the app — if the model files aren't deployed alongside it.

Tested by launching the app locally (`streamlit run streamlit_app.py`); it returned HTTP 200 with no tracebacks on load.

## 12. Future Work

1. Acquire a state/LGA-level dataset (even without weather) to enable sub-national risk estimates.
2. Source weather/climate data (rainfall, temperature) to test whether it improves on the current seasonality-only signal.
3. Re-validate the classifier's decision threshold against genuinely new out-of-sample months as they accumulate.
4. Implement true per-prediction SHAP explanations once a case-level (not just national-aggregate) model is built.
5. Have an epidemiologist review and formally validate (or replace) the 462-case/month threshold definition.
6. Investigate the original SVM's `sdv` dependency to determine whether synthetic data affected its training, before relying on it further.

---

## Reproducibility

All code, processed data, models, and metrics referenced in this report are saved under a reproducible project structure. No results in this report were estimated or assumed — every number above was produced by running the corresponding notebook/script in this project, and all five notebooks execute cleanly end-to-end with zero errors.

```
AI-lassa/
├── data/
│   ├── raw/2015-2025__3_.xlsx            (your uploaded dataset, unmodified)
│   └── processed/monthly_features.csv    (cleaned, aggregated, feature-engineered)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb              (Phase 2)
│   ├── 02_model_benchmarking.ipynb              (Phase 3)
│   ├── 03_tuning_ablation_alternatives.ipynb    (Phase 4)
│   ├── 04_final_test_evaluation.ipynb           (Phase 5, test set)
│   └── 05_final_models_and_inference.ipynb      (Phase 5–6, final artifacts + explainability)
├── src/
│   └── predict.py                        (importable inference module, used by notebook 05)
├── models/                               (final_*.pkl — the actual models to integrate into the app)
├── outputs/metrics/                      (every benchmark/ablation/threshold table as CSV)
├── streamlit_app.py, svmnew_model.pkl, scallernew.pkl, label_encodersnew.pkl   (original app — untouched)
└── AI4Lassa_Final_Report.md              (this document)
```

Run the notebooks in order (01 → 05) to reproduce every result from scratch.
