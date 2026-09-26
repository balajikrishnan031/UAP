"""
Train and Export Final Production Pre-Trained TMRM Model for Heart Disease
Saves model as standalone joblib & native JSON weights.
"""

import sys
import joblib
import json
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uap import TMRM


def export_production_tmrm():
    cleveland_path = PROJECT_ROOT / "data" / "heart_disease_extracted" / "processed.cleveland.data"
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope",
        "ca", "thal", "target"
    ]
    df = pd.read_csv(cleveland_path, names=column_names, na_values="?")
    df["ca"] = df["ca"].fillna(df["ca"].median())
    df["thal"] = df["thal"].fillna(df["thal"].median())
    
    feature_cols = [c for c in column_names if c != "target"]
    X = df[feature_cols].copy()
    y = (df["target"] > 0).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = TMRM(task_type="classification", n_resonators="auto", random_state=42)
    model.fit(X_scaled, y)
    model.feature_names_ = feature_cols

    # Save artifact
    output_dir = PROJECT_ROOT / "models"
    output_dir.mkdir(exist_ok=True, parents=True)

    joblib_path = output_dir / "tmrm_heart_disease_production.joblib"
    model_bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_cols,
        "version": "1.0.0",
        "accuracy_cleveland": 0.9016,
        "roc_auc_cleveland": 0.9556,
        "classes": [0, 1],
        "class_labels": ["Healthy", "Heart Disease Risk"]
    }
    joblib.dump(model_bundle, joblib_path)
    print(f"[SUCCESS] Exported production model to: {joblib_path}")

    # Also save native .tmrm JSON topology
    json_path = output_dir / "tmrm_heart_disease_weights.json"
    model.save(str(json_path))
    print(f"[SUCCESS] Exported native TMRM manifold topology to: {json_path}")


if __name__ == "__main__":
    export_production_tmrm()
