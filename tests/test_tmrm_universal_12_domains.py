"""
Universal 12-Domain Benchmark & Scientific Audit:
Topological Manifold Resonant Machine (TMRM) vs Industry Prediction Models
========================================================================
Datasets:
1. Heart Disease (Clinical Cardiology) -> Classification
2. Breast Cancer (Cellular Diagnostics) -> Classification
3. Clinical Survival (High-Risk Mortality) -> Classification
4. Credit Risk (FinTech Default) -> Classification
5. Fraud Detection (Imbalanced Transactions) -> Classification
6. Telecom Churn (Customer Retention) -> Classification
7. IoT Maintenance (Industrial Sensor Failure) -> Classification
8. Cyber Intrusion (Network Attack Classification) -> Classification
9. Agriculture Crop (Smart Farming Yield) -> Classification
10. Energy Grid (Utility Load Demand) -> Regression
11. Air Quality (Environmental AQI / Particulate) -> Regression
12. California Housing (Macroeconomic Median Price) -> Regression
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from uap.models.novel_tmrm import TopologicalManifoldResonantMachine


DOMAINS = [
    {"name": "Heart Disease (Cardiology)", "file": "data/real_world/heart_disease.csv", "task": "classification"},
    {"name": "Breast Cancer (Diagnostics)", "file": "data/real_world/breast_cancer.csv", "task": "classification"},
    {"name": "Clinical Survival (Medical Risk)", "file": "data/real_world/clinical_survival.csv", "task": "classification"},
    {"name": "Credit Risk (FinTech Default)", "file": "data/real_world/credit_risk.csv", "task": "classification"},
    {"name": "Fraud Detection (Transactions)", "file": "data/real_world/fraud_detection.csv", "task": "classification"},
    {"name": "Telecom Churn (Customer Retention)", "file": "data/real_world/telecom_churn.csv", "task": "classification"},
    {"name": "IoT Maintenance (Sensors)", "file": "data/real_world/iot_maintenance.csv", "task": "classification"},
    {"name": "Cyber Intrusion (Cybersecurity)", "file": "data/real_world/cyber_intrusion.csv", "task": "classification"},
    {"name": "Agriculture Crop (Smart Farming)", "file": "data/real_world/agriculture_crop.csv", "task": "classification"},
    {"name": "Energy Grid (Power Demand)", "file": "data/real_world/energy_grid.csv", "task": "regression"},
    {"name": "Air Quality (Environmental AQI)", "file": "data/real_world/air_quality.csv", "task": "regression"},
    {"name": "California Housing (Real Estate)", "file": "data/real_world/california_housing.csv", "task": "regression"},
]


def run_benchmark():
    results = []

    print("=" * 105, flush=True)
    print("STARTING UNIVERSAL 12-DOMAIN REAL-WORLD BENCHMARK: TMRM vs INDUSTRY MODELS", flush=True)
    print("=" * 105, flush=True)

    for item in DOMAINS:
        name = item["name"]
        file_path = item["file"]
        task = item["task"]

        if not os.path.exists(file_path):
            print(f"Skipping {name}: file not found.", flush=True)
            continue

        df = pd.read_csv(file_path)
        # Standardize target column detection
        target_col = "target" if "target" in df.columns else df.columns[-1]

        # Use up to 1000 samples for swift, responsive benchmark execution
        if len(df) > 1000:
            df = df.sample(1000, random_state=42)

        X_df = df.drop(columns=[target_col])
        y_ser = df[target_col]

        # Clean features to numeric
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(X_df[col].median() if len(X_df[col].dropna()) > 0 else 0)

        X_arr = X_df.values
        # Clean target
        if task == "classification":
            y_arr = pd.to_numeric(y_ser, errors='coerce').fillna(0).values.astype(int)
        else:
            y_arr = pd.to_numeric(y_ser, errors='coerce').fillna(y_ser.median()).values.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.25, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if task == "classification":
            # 1. TMRM (Novel Algorithm)
            t0 = time.time()
            tmrm = TopologicalManifoldResonantMachine(task_type="classification", n_resonators="auto", random_state=42)
            tmrm.fit(X_train, y_train)
            tmrm_time = time.time() - t0
            tmrm_preds = tmrm.predict(X_test)
            tmrm_score = f1_score(y_test, tmrm_preds, average="macro")

            # 2. Gradient Boosting
            gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
            gb.fit(X_train, y_train)
            gb_preds = gb.predict(X_test)
            gb_score = f1_score(y_test, gb_preds, average="macro")

            # 3. Random Forest
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X_train, y_train)
            rf_preds = rf.predict(X_test)
            rf_score = f1_score(y_test, rf_preds, average="macro")

            # 4. Logistic Regression
            lr = LogisticRegression(max_iter=300, random_state=42)
            try:
                lr.fit(X_train_scaled, y_train)
                lr_preds = lr.predict(X_test_scaled)
                lr_score = f1_score(y_test, lr_preds, average="macro")
            except Exception:
                lr_score = 0.0

            metric_name = "Macro-F1"

        else:
            # Regression Task
            t0 = time.time()
            tmrm = TopologicalManifoldResonantMachine(task_type="regression", n_resonators="auto", random_state=42)
            tmrm.fit(X_train, y_train)
            tmrm_time = time.time() - t0
            tmrm_preds = tmrm.predict(X_test)
            tmrm_score = r2_score(y_test, tmrm_preds)

            # Gradient Boosting
            gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
            gb.fit(X_train, y_train)
            gb_preds = gb.predict(X_test)
            gb_score = r2_score(y_test, gb_preds)

            # Random Forest
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_train, y_train)
            rf_preds = rf.predict(X_test)
            rf_score = r2_score(y_test, rf_preds)

            # Ridge
            rg = Ridge(random_state=42)
            rg.fit(X_train_scaled, y_train)
            rg_preds = rg.predict(X_test_scaled)
            rg_score = r2_score(y_test, rg_preds)
            lr_score = rg_score

            metric_name = "R^2 Score"

        # Winner
        scores = {"TMRM": tmrm_score, "GradientBoosting": gb_score, "RandomForest": rf_score, "Linear/Ridge": lr_score}
        winner = max(scores, key=scores.get)

        res_dict = {
            "Domain": name,
            "Task": task,
            "Metric": metric_name,
            "TMRM": round(tmrm_score, 4),
            "GradientBoosting": round(gb_score, 4),
            "RandomForest": round(rf_score, 4),
            "Linear_Baseline": round(lr_score, 4),
            "Winner": winner,
            "TMRM_vs_GB": round(tmrm_score - gb_score, 4),
            "TMRM_vs_RF": round(tmrm_score - rf_score, 4),
            "TMRM_Speed_s": round(tmrm_time, 2)
        }
        results.append(res_dict)
        print(f"[{name:<35}] -> TMRM: {tmrm_score:.4f} | GB: {gb_score:.4f} | RF: {rf_score:.4f} | Linear: {lr_score:.4f} | Winner: {winner}", flush=True)

    res_df = pd.DataFrame(results)
    os.makedirs("models", exist_ok=True)
    res_df.to_csv("models/tmrm_vs_all_models_12_domain_benchmark.csv", index=False)
    print("\n" + "=" * 105, flush=True)
    print("12-DOMAIN BENCHMARK COMPLETE. Saved to models/tmrm_vs_all_models_12_domain_benchmark.csv", flush=True)
    print("=" * 105, flush=True)


if __name__ == "__main__":
    run_benchmark()
