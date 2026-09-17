# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time  # 👈 Added for delay

BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()

# ----------------------------
# Helpers
# ----------------------------
def safe_map_binary_series(s):
    """Map common binary representations to 0/1, else return NaN for invalids."""
    def map_val(v):
        if pd.isna(v):
            return np.nan
        v_str = str(v).strip().lower()
        if v_str in {"1", "yes", "y", "true", "t"}:
            return 1
        if v_str in {"0", "no", "n", "false", "f"}:
            return 0
        # numeric-like
        try:
            if float(v) == 1.0:
                return 1
            if float(v) == 0.0:
                return 0
        except Exception:
            pass
        return np.nan
    return s.map(map_val)

def normalize_sex_series(s):
    """Normalize sex to 'Male'/'Female' where possible."""
    def norm(v):
        if pd.isna(v):
            return v
        v_str = str(v).strip().lower()
        if v_str in {"m", "male"}:
            return "Male"
        if v_str in {"f", "female"}:
            return "Female"
        return str(v).strip()
    return s.map(norm)

def normalize_ward_series(s):
    """Normalize ward values to 'Ward'/'ICU'."""
    def norm(v):
        if pd.isna(v):
            return v
        v_str = str(v).strip().lower()
        if v_str in {"ward", "w"}:
            return "Ward"
        if "icu" in v_str:
            return "ICU"
        return str(v).strip()
    return s.map(norm)

# ----------------------------
# Load artifacts (safe path)
# ----------------------------
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    scaler_path = os.path.join(base, "scallernew.pkl")
    enc_path = os.path.join(base, "label_encodersnew.pkl")
    model_path = os.path.join(base, "svmnew_model.pkl")

    missing = []
    for p in (scaler_path, enc_path, model_path):
        if not os.path.exists(p):
            missing.append(os.path.basename(p))
    if missing:
        raise FileNotFoundError(f"Required files not found in app folder: {missing}")

    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(enc_path)  # expected dict: {"Sex":le, "WardType":le, "Outcome":le}
    model = joblib.load(model_path)
    return scaler, label_encoders, model

# Try loading artifacts; show friendly error if not present
try:
    scaler, label_encoders, model = load_artifacts()
except Exception as e:
    scaler = label_encoders = model = None
    st.error(f"Model files load error: {e}")
    st.stop()

# ----------------------------
# Expected features (exactly as in your CSV)
# ----------------------------
expected_features = [
    "Age", "Sex", "WardType",
    "Temperature_C", "HeartRate_bpm", "RespRate_bpm",
    "SystolicBP_mmHg", "DiastolicBP_mmHg", "SpO2_%",
    "GCS", "OxygenNeeded", "VentilationNeeded", "BleedingPresent"
]

st.set_page_config(page_title="AI4Lassa", layout="wide")
st.title("🏥 AI4Lassa")

tab_patient, tab_forecast = st.tabs(
    ["🏥 Patient Outcome Predictor", "📈 National Outbreak Early-Warning"]
)

with tab_patient:
    st.markdown(
        "This tool predicts **Outcome** for an individual patient using the original "
        "trained SVM model. Fill values manually or upload a CSV that matches the "
        "template (downloadable below)."
    )

    # ----------------------------
    # Manual entry
    # ----------------------------
    with st.expander("✍️ Manual Entry"):
        with st.form("manual_form"):
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1)
            sex = st.selectbox("Sex", ["Male", "Female"])
            ward_type = st.selectbox("Ward Type", ["Ward", "ICU"])
            temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, value=80, step=1)
            resp_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=5, max_value=60, value=18, step=1)
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=120, step=1)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80, step=1)
            spo2 = st.number_input("SpO₂ (%)", min_value=50, max_value=100, value=97, step=1)
            gcs = st.number_input("Glasgow Coma Scale (GCS)", min_value=3, max_value=15, value=15, step=1)

            oxygen_needed = st.radio("Oxygen Needed?", ("No", "Yes"), index=0)
            ventilation_needed = st.radio("Ventilation Needed?", ("No", "Yes"), index=0)
            bleeding_present = st.radio("Bleeding Present?", ("No", "Yes"), index=0)

            submit_manual = st.form_submit_button("🔮 Predict Outcome")

        if submit_manual:
            with st.spinner("⏳ Running prediction... Please wait."):
                time.sleep(3)  # 👈 simulate delay
                try:
                    bin_map = {"Yes": 1, "No": 0}
                    row = pd.DataFrame([[age, sex, ward_type, temperature, heart_rate,
                                         resp_rate, systolic_bp, diastolic_bp, spo2, gcs,
                                         bin_map[oxygen_needed], bin_map[ventilation_needed],
                                         bin_map[bleeding_present]]], columns=expected_features)

                    row["Sex"] = normalize_sex_series(row["Sex"])
                    row["WardType"] = normalize_ward_series(row["WardType"])

                    row["Sex"] = label_encoders["Sex"].transform(row["Sex"])
                    row["WardType"] = label_encoders["WardType"].transform(row["WardType"])

                    Xs = scaler.transform(row.astype(float))
                    pred = model.predict(Xs)[0]
                    label = label_encoders["Outcome"].inverse_transform([pred])[0]
                    st.success(f"✅ Predicted Outcome for lassa: **{label}**")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # ----------------------------
    # CSV upload
    # ----------------------------
    # CSV upload
    # ----------------------------
    with st.expander("📂 CSV Upload"):
        uploaded_file = st.file_uploader("Upload CSV (must contain exact headers)", type=["csv"])
        template_df = pd.DataFrame(columns=expected_features)
        st.download_button("📥 Download CSV Template", template_df.to_csv(index=False).encode("utf-8"),
                           "template.csv", "text/csv")

        if uploaded_file is not None:
            with st.spinner("⏳ Processing CSV... Please wait."):
                time.sleep(3)  # 👈 simulate delay
                try:
                    df_uploaded = pd.read_csv(uploaded_file)
                    df_uploaded.columns = [c.strip() for c in df_uploaded.columns]

                    df_uploaded["Sex"] = normalize_sex_series(df_uploaded["Sex"])
                    df_uploaded["WardType"] = normalize_ward_series(df_uploaded["WardType"])
                    for col in ["OxygenNeeded", "VentilationNeeded", "BleedingPresent"]:
                        df_uploaded[col] = safe_map_binary_series(df_uploaded[col])

                    df_uploaded["Sex"] = label_encoders["Sex"].transform(df_uploaded["Sex"])
                    df_uploaded["WardType"] = label_encoders["WardType"].transform(df_uploaded["WardType"])

                    Xs = scaler.transform(df_uploaded[expected_features].astype(float))
                    preds = model.predict(Xs)
                    df_uploaded["PredictedOutcome"] = label_encoders["Outcome"].inverse_transform(preds)

                    st.success("✅ Predictions done. Preview:")
                    st.dataframe(df_uploaded.head(10))
                    st.download_button("📥 Download Predictions",
                                       df_uploaded.to_csv(index=False).encode("utf-8"),
                                       "predictions.csv", "text/csv")

                    # ✅ Dynamic outbreak check
                    outcome_counts = df_uploaded["PredictedOutcome"].value_counts()
                    st.info("📊 Outcome Counts:")
                    st.write(outcome_counts)

                    if "Positive" in outcome_counts and "Negative" in outcome_counts:
                        pos_count = outcome_counts.get("Positive", 0)
                        neg_count = outcome_counts.get("Negative", 0)

                        if pos_count > neg_count:
                            st.error(f"🚨 Lassa Outbreak Declared! Positive cases ({pos_count}) exceed negative cases ({neg_count}).")
                        else:
                            st.success(f"✅ No outbreak detected. Negative cases ({neg_count}) are higher.")
                    else:
                        st.warning("⚠️ Could not find 'Positive'/'Negative' labels in predictions. "
                                   "Please check your label encoder classes.")

                except Exception as e:
                    st.error(f"CSV Prediction failed: {e}")

# ----------------------------------------------------------------------------
# National Outbreak Early-Warning tab (new — added on top of the original app)
# ----------------------------------------------------------------------------
FORECAST_FEATURES = [
    "case_count", "case_count_lag1", "case_count_lag2", "case_count_lag3",
    "case_count_lag6", "case_count_lag12",
    "case_count_roll3_mean", "case_count_roll6_mean", "case_count_roll3_max",
    "case_growth_lag1", "positivity_rate_lag1",
    "month_sin", "month_cos", "year",
]


@st.cache_resource
def load_forecast_artifacts():
    """Loads the national early-warning model. Returns None if the model
    files aren't present, so the rest of the app keeps working either way."""
    model_dir = os.path.join(BASE_DIR, "models")
    data_path = os.path.join(BASE_DIR, "data", "processed", "monthly_features.csv")
    importance_path = os.path.join(BASE_DIR, "outputs", "metrics", "rf_feature_importance.csv")

    required = [
        os.path.join(model_dir, "final_rf_regressor.pkl"),
        os.path.join(model_dir, "final_logit_riskflag.pkl"),
        os.path.join(model_dir, "final_riskflag_scaler.pkl"),
        os.path.join(model_dir, "final_config.pkl"),
        data_path,
    ]
    if any(not os.path.exists(p) for p in required):
        return None

    rf = joblib.load(os.path.join(model_dir, "final_rf_regressor.pkl"))
    logit = joblib.load(os.path.join(model_dir, "final_logit_riskflag.pkl"))
    fc_scaler = joblib.load(os.path.join(model_dir, "final_riskflag_scaler.pkl"))
    config = joblib.load(os.path.join(model_dir, "final_config.pkl"))
    monthly_df = pd.read_csv(data_path)

    if os.path.exists(importance_path):
        importances = pd.read_csv(importance_path, index_col=0).iloc[:, 0]
    else:
        importances = pd.Series(dtype=float)

    return rf, logit, fc_scaler, config, monthly_df, importances


def forecast_from_row(row, rf, logit, fc_scaler, config, importances, decision_threshold):
    Xdf = pd.DataFrame([row[FORECAST_FEATURES].values], columns=FORECAST_FEATURES)
    forecast = float(rf.predict(Xdf)[0])
    proba = float(logit.predict_proba(fc_scaler.transform(Xdf))[0, 1])
    risk_level = (
        "HIGH" if proba >= decision_threshold
        else ("ELEVATED" if proba >= decision_threshold * 0.5 else "LOW")
    )
    top_feats = importances.sort_values(ascending=False).head(3) if len(importances) else pd.Series(dtype=float)
    factors = [f"{feat} = {row[feat]:.2f}  (global importance {imp:.0%})" for feat, imp in top_feats.items()]
    return forecast, proba, risk_level, factors


with tab_forecast:
    st.markdown(
        "Forecasts **national next-month Lassa fever case volume** from historical "
        "case counts and seasonality. Built and validated separately from the patient "
        "tool above — see caveats below before treating this as an alert system."
    )

    forecast_artifacts = load_forecast_artifacts()

    if forecast_artifacts is None:
        st.warning(
            "⚠️ Early-warning model files not found alongside this app "
            "(expected a `models/` folder with `final_rf_regressor.pkl`, "
            "`final_logit_riskflag.pkl`, `final_riskflag_scaler.pkl`, `final_config.pkl`, "
            "and `data/processed/monthly_features.csv`). "
            "Make sure these were committed to the repo alongside `streamlit_app.py`."
        )
    else:
        rf, logit, fc_scaler, config, monthly_df, importances = forecast_artifacts

        st.info(
            "**Read before relying on this tool:** the case-count forecast tested well "
            "on 2024–2025 held-out data (MAE ≈ 49 cases, R² ≈ 0.57). The HIGH/ELEVATED/LOW "
            "risk label is more provisional — it missed the one true high-risk month in "
            "that same test period at the default threshold, and its decision threshold "
            "was chosen after inspecting that test result rather than independently "
            "validated. Treat the risk label as a soft, exploratory signal, and the case-count "
            "forecast as the primary, more trustworthy output. This is decision-support, not "
            "a diagnosis or a confirmed outbreak declaration."
        )

        st.subheader("Forecast for the next available month")

        monthly_df = monthly_df.sort_values("month_ts")
        latest_row = monthly_df.iloc[-1]

        threshold = st.slider(
            "Risk-flag decision threshold (probability)",
            min_value=0.05, max_value=0.95, value=0.15, step=0.05,
            help=(
                "Lower = flags more months as risky (higher recall, more false alarms). "
                "0.5 was the default used in validation; 0.15 is the provisional, "
                "recall-favoring value discussed in the project report — neither is "
                "independently confirmed on new data yet."
            ),
        )

        if st.button("🔮 Generate Forecast", type="primary"):
            with st.spinner("Running forecast..."):
                forecast, proba, risk_level, factors = forecast_from_row(
                    latest_row, rf, logit, fc_scaler, config, importances, threshold
                )

            st.markdown(f"**Based on data through:** {latest_row['month']}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Forecast — next month's national case count", f"{round(forecast)}")
            with col2:
                st.metric("Provisional outbreak probability", f"{proba:.1%}")

            if risk_level == "HIGH":
                st.error(f"🚨 Risk level: **{risk_level}** (provisional — see caveats above)")
            elif risk_level == "ELEVATED":
                st.warning(f"⚠️ Risk level: **{risk_level}** (provisional — see caveats above)")
            else:
                st.success(f"✅ Risk level: **{risk_level}** (provisional — see caveats above)")

            if factors:
                st.markdown("**Main contributing factors (predictive association, not causation):**")
                for f in factors:
                    st.markdown(f"- {f}")

        with st.expander("📊 Recent monthly case counts"):
            st.dataframe(
                monthly_df[["month", "case_count", "positivity_rate"]].tail(12).reset_index(drop=True)
            )

        with st.expander("ℹ️ About this model"):
            st.markdown(
                f"""
- **Model:** {config.get('regression_model', 'Random Forest Regressor')} for case-count forecast;
  {config.get('classifier_model', 'Logistic Regression')} for the risk flag.
- **Trained on:** {config.get('train_period', '—')} + {config.get('validation_period', '—')}
- **Tested on:** {config.get('test_period', '—')} (held out from all model selection)
- **Test performance:** MAE = {config.get('test_regression_mae', '—')},
  R² = {config.get('test_regression_r2', '—')}
- **Risk threshold definition:** {config.get('risk_threshold_definition', '—')}
- **Known limitation:** {config.get('classifier_threshold_caveat', '—')}

Full methodology, benchmarking, and honest limitations are in `AI4Lassa_Final_Report.md`
and the `notebooks/` folder in this repository.
                """
            )
