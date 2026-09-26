"""
Standalone TMRM Heart Disease Prediction Runner
Works with both single patient input and bulk CSV batch predictions!
"""

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
MODEL_PATH = CURRENT_DIR / "tmrm_heart_disease_production.joblib"
if not MODEL_PATH.exists():
    MODEL_PATH = CURRENT_DIR / "models" / "tmrm_heart_disease_production.joblib"

def predict_single_patient(age=62, sex=1, cp=4, trestbps=150, chol=285, fbs=1,
                           restecg=2, thalach=125, exang=1, oldpeak=2.8, slope=2, ca=2, thal=7):
    print("=" * 65)
    print("      TMRM CLINICAL DECISION SUPPORT SYSTEM (PREDICTION)")
    print("=" * 65)
    
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]
    features = bundle["feature_names"]

    patient_dict = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    df_in = pd.DataFrame([patient_dict])[features]
    X_scaled = scaler.transform(df_in)

    pred = int(model.predict(X_scaled)[0])
    probs = model.predict_proba(X_scaled)[0]
    novelty = float(model.get_epistemic_novelty(X_scaled)[0])

    print(f"Patient Data Input:")
    print(f"  * Age: {age} | Sex: {'Male' if sex==1 else 'Female'} | BP: {trestbps} mmHg | Chol: {chol} mg/dl")
    print(f"  * Chest Pain Type: {cp} | Max Heart Rate: {thalach} bpm | Calcified Vessels: {ca}")
    print("-" * 65)
    if pred == 1:
        print(f"DIAGNOSIS RESULT : [ALERT] HEART DISEASE RISK DETECTED")
        print(f"Confidence       : {probs[1] * 100:.2f}% Probability of Disease")
    else:
        print(f"DIAGNOSIS RESULT : [SAFE] HEALTHY / NO CORONARY HEART DISEASE")
        print(f"Confidence       : {probs[0] * 100:.2f}% Probability Safe")

    print(f"Epistemic Safety : {'SAFE_TO_PREDICT' if novelty < 3.0 else 'UNFAMILIAR / DOCTOR REVIEW REQUIRED'} (Novelty Score: {novelty:.2f})")
    print("=" * 65)


def predict_batch_csv(csv_path="sample_patients.csv"):
    print("=" * 80)
    print(f"BATCH INFERENCE ON: {csv_path}")
    print("=" * 80)
    df = pd.read_csv(csv_path)
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]
    features = bundle["feature_names"]

    X = df[features]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    df["Diagnosis"] = ["Heart Disease Risk" if p == 1 else "Healthy" for p in preds]
    df["Risk_Probability_%"] = np.round(probs * 100, 2)

    result_cols = ["patient_id", "Diagnosis", "Risk_Probability_%", "age", "trestbps", "chol", "cp"]
    print(df[[c for c in result_cols if c in df.columns]].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].endswith(".csv"):
        predict_batch_csv(sys.argv[1])
    else:
        # Run demo on sample high-risk patient
        predict_single_patient()
        print("\nTip: You can also run batch predictions on CSV files by typing:")
        print("     python predict_patient.py sample_patients.csv\n")
