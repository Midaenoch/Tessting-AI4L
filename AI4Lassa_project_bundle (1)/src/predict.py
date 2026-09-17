"""
AI4Lassa — predict.py

Produces a 1-month-ahead national case-count forecast, a provisional
risk label, and a plain-language explanation of the main contributing
factors, for the most recent month in the processed feature table
(or any row passed in).

This is a decision-support output, not a diagnosis or a guarantee.
"""
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = "/home/claude/AI-lassa/models"
DATA_PATH = "/home/claude/AI-lassa/data/processed/monthly_features.csv"


def load_artifacts():
    rf = joblib.load(f"{MODEL_DIR}/final_rf_regressor.pkl")
    logit = joblib.load(f"{MODEL_DIR}/final_logit_riskflag.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/final_riskflag_scaler.pkl")
    config = joblib.load(f"{MODEL_DIR}/final_config.pkl")
    importances = pd.read_csv(f"{MODEL_DIR}/../outputs/metrics/rf_feature_importance.csv",
                               index_col=0).iloc[:, 0]
    return rf, logit, scaler, config, importances


def predict_next_month(row: pd.Series, rf, logit, scaler, config, importances,
                        decision_threshold: float = None):
    features = config["features"]
    X = row[features].values.reshape(1, -1)
    Xdf = pd.DataFrame(X, columns=features)

    forecast = float(rf.predict(Xdf)[0])
    proba_high_risk = float(logit.predict_proba(scaler.transform(Xdf))[0, 1])

    threshold = decision_threshold if decision_threshold is not None \
        else config["classifier_decision_threshold_default"]
    risk_level = "HIGH" if proba_high_risk >= threshold else \
                 ("ELEVATED" if proba_high_risk >= threshold * 0.5 else "LOW")

    # Simple, transparent explanation: report the top-3 globally-important
    # features and this month's actual value, not a per-prediction SHAP
    # attribution (SHAP recommended as future work — see README).
    top_feats = importances.head(3)
    explanation = []
    for feat, imp in top_feats.items():
        explanation.append(f"{feat} = {row[feat]:.2f} (global importance {imp:.1%})")

    return {
        "forecast_next_month_cases": round(forecast),
        "outbreak_probability_provisional": round(proba_high_risk, 3),
        "risk_level_provisional": risk_level,
        "decision_threshold_used": threshold,
        "top_contributing_factors": explanation,
        "note": "Risk level is provisional and based on limited historical high-risk "
                "events (see README limitations). This is a decision-support signal, "
                "not a diagnosis or confirmed outbreak declaration.",
    }


if __name__ == "__main__":
    rf, logit, scaler, config, importances = load_artifacts()
    df = pd.read_csv(DATA_PATH)
    latest_row = df.iloc[-1]
    result = predict_next_month(latest_row, rf, logit, scaler, config, importances,
                                 decision_threshold=0.15)
    print(f"Forecast based on month: {latest_row['month']}")
    for k, v in result.items():
        print(f"  {k}: {v}")
