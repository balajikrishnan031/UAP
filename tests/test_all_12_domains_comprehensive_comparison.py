"""
UAP 4.0 Comprehensive Universal Benchmark & Real-World Instance Audit across ALL 12 Domains.
1. Trains and evaluates UAP 4.0 vs 6 baseline models/libraries:
   - Scikit-Learn Random Forest
   - DMLC XGBoost
   - Microsoft LightGBM
   - Support Vector Machine (SVC/SVR)
   - Linear Baseline (LogisticRegression / Ridge)
   - k-Nearest Neighbors (KNN)
2. Displays side-by-side Real Individual Instance Predictions (Actual Ground Truth vs UAP Prediction).
3. Automatically audits performance across all 12 domains.
"""

from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from data.dataset_loader import (
    fetch_heart_disease, fetch_breast_cancer, fetch_financial_credit_risk,
    fetch_smart_agriculture, fetch_iot_predictive_maintenance,
    fetch_california_housing_data, fetch_energy_grid, fetch_telecom_churn,
    fetch_cyber_intrusion, fetch_fraud_detection, fetch_air_quality,
    fetch_clinical_survival
)
from uap.engine import UAPEngine
from uap.core.contracts import UserUtilitySpec

DOMAINS = [
    {
        "id": 1,
        "name": "Diagnostic Oncology",
        "dataset": "breast_cancer.csv",
        "fetcher": fetch_breast_cancer,
        "target": "target",
        "task": "classification",
        "target_labels": {0: "Malignant (Cancer)", 1: "Benign (Healthy)"},
        "max_train": 450,
        "key_features": ["mean radius", "mean texture", "mean perimeter", "worst area"]
    },
    {
        "id": 2,
        "name": "Clinical Cardiology",
        "dataset": "heart_disease.csv",
        "fetcher": fetch_heart_disease,
        "target": "target",
        "task": "classification",
        "target_labels": {0: "Healthy Heart", 1: "Heart Disease Present"},
        "max_train": 250,
        "key_features": ["age", "sex", "cp", "trestbps", "chol", "thalach"]
    },
    {
        "id": 3,
        "name": "Banking & Credit Risk",
        "dataset": "credit_risk.csv",
        "fetcher": fetch_financial_credit_risk,
        "target": "target",
        "task": "classification",
        "target_labels": {0: "Credit Approved / Low Risk", 1: "Loan Default Risk"},
        "max_train": 800,
        "key_features": ["fin_feature_1", "fin_feature_2", "fin_feature_3", "fin_feature_4"]
    },
    {
        "id": 4,
        "name": "Fintech & Fraud Detection",
        "dataset": "fraud_detection.csv",
        "fetcher": fetch_fraud_detection,
        "target": "is_fraud",
        "task": "classification",
        "target_labels": {0: "Legitimate Transaction", 1: "Fraudulent Transaction"},
        "max_train": 2500,
        "key_features": ["ratio_to_median_price", "distance_from_home_km", "used_chip"]
    },
    {
        "id": 5,
        "name": "Industrial IoT Maintenance",
        "dataset": "iot_maintenance.csv",
        "fetcher": fetch_iot_predictive_maintenance,
        "target": "target",
        "task": "classification",
        "target_labels": {0: "Machine Operating Normally", 1: "Equipment Failure Imminent"},
        "max_train": 3000,
        "key_features": ["air_temperature_K", "rotational_speed_rpm", "torque_Nm", "tool_wear_min"]
    },
    {
        "id": 6,
        "name": "Telecom Customer Churn",
        "dataset": "telecom_churn.csv",
        "fetcher": fetch_telecom_churn,
        "target": "churn",
        "task": "classification",
        "target_labels": {0: "Retained Customer", 1: "Customer Churned"},
        "max_train": 2000,
        "key_features": ["tenure_months", "monthly_charges", "customer_service_calls"]
    },
    {
        "id": 7,
        "name": "Smart Agriculture Recommendation",
        "dataset": "agriculture_crop.csv",
        "fetcher": fetch_smart_agriculture,
        "target": "target",
        "task": "classification",
        "target_labels": {0: "Low/Unsuitable Yield", 1: "Optimal High Yield"},
        "max_train": 1800,
        "key_features": ["nitrogen_N", "phosphorus_P", "potassium_K", "rainfall_mm", "soil_ph"]
    },
    {
        "id": 8,
        "name": "Cybersecurity Intrusion Detection",
        "dataset": "cyber_intrusion.csv",
        "fetcher": fetch_cyber_intrusion,
        "target": "is_intrusion",
        "task": "classification",
        "target_labels": {0: "Normal Network Traffic", 1: "Malicious Intrusion Attack"},
        "max_train": 2500,
        "key_features": ["failed_logins", "src_bytes", "serror_rate", "logged_in"]
    },
    {
        "id": 9,
        "name": "Medical Clinical Survival",
        "dataset": "clinical_survival.csv",
        "fetcher": fetch_clinical_survival,
        "target": "death_event",
        "task": "classification",
        "target_labels": {0: "Surviving Patient", 1: "High Mortality Event"},
        "max_train": 240,
        "key_features": ["age", "ejection_fraction", "serum_creatinine", "time"]
    },
    {
        "id": 10,
        "name": "Macroeconomic Housing Valuation",
        "dataset": "california_housing.csv",
        "fetcher": fetch_california_housing_data,
        "target": "MedHouseVal",
        "task": "regression",
        "target_labels": None,
        "max_train": 2500,
        "key_features": ["MedInc", "HouseAge", "AveRooms", "Latitude", "Longitude"]
    },
    {
        "id": 11,
        "name": "Clean Energy Grid Load",
        "dataset": "energy_grid.csv",
        "fetcher": fetch_energy_grid,
        "target": "grid_load_mw",
        "task": "regression",
        "target_labels": None,
        "max_train": 2500,
        "key_features": ["lag_1h_load_mw", "lag_24h_load_mw", "temperature_c", "hour_of_day"]
    },
    {
        "id": 12,
        "name": "Environmental Air Quality (AQI)",
        "dataset": "air_quality.csv",
        "fetcher": fetch_air_quality,
        "target": "pm2_5_aqi",
        "task": "regression",
        "target_labels": None,
        "max_train": 2000,
        "key_features": ["ambient_temp_c", "relative_humidity", "wind_speed_ms", "traffic_density_idx"]
    }
]

def run_comprehensive_audit():
    print("=" * 110)
    print(">>> UAP 4.0 UNIVERSAL BENCHMARK & REAL INDIVIDUAL PREDICTION AUDIT (ALL 12 DOMAINS)")
    print("=" * 110)

    comparison_results = []
    instance_prediction_records = []

    for idx, domain_cfg in enumerate(DOMAINS, 1):
        d_name = domain_cfg["name"]
        d_code = domain_cfg["dataset"]
        target_col = domain_cfg["target"]
        is_cls = domain_cfg["task"] == "classification"

        print(f"\n[{idx}/12] EVALUATING DOMAIN: {d_name} ({d_code})")
        print("-" * 90)

        # 1. Load Data
        df = domain_cfg["fetcher"]()
        df = df.dropna()

        # Split Train / Holdout Test
        stratify = df[target_col] if is_cls and df[target_col].nunique() < 10 else None
        train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=stratify)

        if len(train_df) > domain_cfg["max_train"]:
            train_df = train_df.sample(domain_cfg["max_train"], random_state=42)
        if len(test_df) > 500:
            test_df = test_df.sample(500, random_state=42)

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col].values

        # 2. Train UAP 4.0
        t0 = time.time()
        engine = UAPEngine(n_trials=6, timeout_sec=15, utility_spec=UserUtilitySpec(risk_tolerance="medium"))
        engine.fit(train_df, target_column=target_col, dataset_name=d_code)
        uap_train_time = round(time.time() - t0, 2)

        uap_preds = engine.predict(X_test)

        # 3. Train Competitor Baseline Models
        # Fill missing values for baseline models
        X_tr_fill = X_train.fillna(X_train.mean(numeric_only=True)).fillna(0)
        X_te_fill = X_test.fillna(X_train.mean(numeric_only=True)).fillna(0)

        models_to_test = {}
        if is_cls:
            models_to_test["Random Forest (sklearn)"] = RandomForestClassifier(n_estimators=100, random_state=42)
            models_to_test["XGBoost (DMLC)"] = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
            models_to_test["LightGBM (Microsoft)"] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            models_to_test["Linear (Logistic Regression)"] = LogisticRegression(max_iter=500, random_state=42)
            models_to_test["SVM (RBF Kernel)"] = SVC(probability=True, random_state=42)
            models_to_test["KNN (k=5)"] = KNeighborsClassifier(n_neighbors=5)
        else:
            models_to_test["Random Forest (sklearn)"] = RandomForestRegressor(n_estimators=100, random_state=42)
            models_to_test["XGBoost (DMLC)"] = XGBRegressor(n_estimators=100, random_state=42)
            models_to_test["LightGBM (Microsoft)"] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            models_to_test["Linear (Ridge)"] = Ridge(random_state=42)
            models_to_test["SVM (SVR)"] = SVR()
            models_to_test["KNN (k=5)"] = KNeighborsRegressor(n_neighbors=5)

        competitor_scores = {}
        for m_name, model in models_to_test.items():
            try:
                model.fit(X_tr_fill, y_train)
                m_preds = model.predict(X_te_fill)
                if is_cls:
                    score = accuracy_score(y_test, m_preds)
                    f1 = f1_score(y_test, m_preds, average="macro", zero_division=0)
                    competitor_scores[m_name] = {"acc": score, "f1": f1}
                else:
                    score = r2_score(y_test, m_preds)
                    rmse = root_mean_squared_error(y_test, m_preds)
                    competitor_scores[m_name] = {"r2": score, "rmse": rmse}
            except Exception as e:
                competitor_scores[m_name] = {"acc": 0.0, "f1": 0.0, "r2": 0.0, "rmse": 999.0}

        # 4. Evaluate UAP metrics
        if is_cls:
            uap_acc = accuracy_score(y_test, uap_preds)
            uap_f1 = f1_score(y_test, uap_preds, average="macro", zero_division=0)
            uap_score_str = f"Acc: {uap_acc*100:.2f}% | F1: {uap_f1:.4f}"
            
            # Find best competitor
            best_comp_name = max(competitor_scores.keys(), key=lambda k: competitor_scores[k]["f1"])
            best_comp_f1 = competitor_scores[best_comp_name]["f1"]
            best_comp_acc = competitor_scores[best_comp_name]["acc"]
            best_comp_str = f"{best_comp_name} (Acc: {best_comp_acc*100:.2f}%, F1: {best_comp_f1:.4f})"
            delta_f1 = (uap_f1 - best_comp_f1)
            advantage_str = f"{delta_f1:+0.4f} F1" if delta_f1 >= 0 else f"{delta_f1:+0.4f} F1"
        else:
            uap_r2 = r2_score(y_test, uap_preds)
            uap_rmse = root_mean_squared_error(y_test, uap_preds)
            uap_score_str = f"R2: {uap_r2:.4f} | RMSE: {uap_rmse:.4f}"

            best_comp_name = max(competitor_scores.keys(), key=lambda k: competitor_scores[k]["r2"])
            best_comp_r2 = competitor_scores[best_comp_name]["r2"]
            best_comp_rmse = competitor_scores[best_comp_name]["rmse"]
            best_comp_str = f"{best_comp_name} (R2: {best_comp_r2:.4f}, RMSE: {best_comp_rmse:.4f})"
            delta_r2 = (uap_r2 - best_comp_r2)
            advantage_str = f"{delta_r2:+0.4f} R2"

        print(f"  * UAP 4.0 Architecture : {engine.optimizer.winning_model_name}")
        print(f"  * UAP 4.0 Holdout Score : {uap_score_str}")
        print(f"  * Best Baseline Model   : {best_comp_str}")
        print(f"  * UAP Relative Delta   : {advantage_str}")

        comparison_results.append({
            "Domain": d_name,
            "Dataset": d_code,
            "Task": "Classification" if is_cls else "Regression",
            "UAP_Model": engine.optimizer.winning_model_name,
            "UAP_Score": uap_score_str,
            "Best_Competitor": best_comp_name,
            "Competitor_Score": f"F1: {best_comp_f1:.4f}" if is_cls else f"R2: {best_comp_r2:.4f}",
            "UAP_Advantage": advantage_str,
        })

        # 5. Extract 3 Real Unseen Test Records and Predict with Reliability Card (Show True Ground Truth vs UAP)
        print(f"\n  [REAL INSTANCE PREDICTIONS FOR: {d_name}]")
        print("  " + "-" * 86)
        n_samples_to_show = min(3, len(test_df))
        for sample_i in range(n_samples_to_show):
            raw_row = X_test.iloc[sample_i]
            true_val = y_test[sample_i]
            
            # Predict with Reliability Card
            card = engine.predict_with_reliability_card(raw_row)
            pred_val = card.prediction

            # Human-readable labels
            if is_cls and domain_cfg["target_labels"]:
                true_label_str = f"{true_val} [{domain_cfg['target_labels'].get(int(true_val), 'Unknown')}]"
                pred_label_str = f"{int(pred_val)} [{domain_cfg['target_labels'].get(int(pred_val), 'Unknown')}]"
                is_correct = (int(pred_val) == int(true_val))
                match_str = "[CORRECT MATCH]" if is_correct else "[MISCLASSIFICATION]"
            else:
                true_label_str = f"{true_val:.3f}"
                pred_label_str = f"{pred_val:.3f}"
                pct_err = abs(pred_val - true_val) / (abs(true_val) + 1e-5) * 100
                match_str = f"[Abs Error: {abs(pred_val - true_val):.3f} ({pct_err:.1f}%)]"

            # Key feature snippet
            feat_snippets = [f"{col}={round(float(raw_row[col]), 2)}" for col in domain_cfg["key_features"] if col in raw_row]
            feat_str = ", ".join(feat_snippets[:3])

            print(f"  Record #{sample_i+1}:")
            print(f"    Features        : {feat_str}")
            print(f"    ACTUAL VALUE    : {true_label_str}")
            print(f"    UAP PREDICTION  : {pred_label_str} -> {match_str}")
            print(f"    Trust Decision  : {card.decision} | Confidence: {card.confidence_score*100:.1f}% | OOD: {card.ood_risk}")

            instance_prediction_records.append({
                "Domain": d_name,
                "Record_ID": f"{d_code}_sample_{sample_i+1}",
                "Features_Summary": feat_str,
                "Actual_Ground_Truth": true_label_str,
                "UAP_Predicted": pred_label_str,
                "Match_Result": match_str,
                "Safety_Decision": card.decision,
                "Confidence": f"{card.confidence_score*100:.1f}%"
            })
        print("  " + "-" * 86)

    # ------------------------------------------------------------------------
    # FINAL EXECUTIVE TABLES
    # ------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print(">>> FINAL MASTER BENCHMARK TABLE: UAP 4.0 VS ALL INDUSTRY BASELINE LIBRARIES")
    print("=" * 110)
    cmp_df = pd.DataFrame(comparison_results)
    print(cmp_df.to_string(index=False))

    print("\n" + "=" * 110)
    print(">>> ACTUAL GROUND TRUTH VS UAP PREDICTIONS (INDIVIDUAL INSTANCE AUDIT)")
    print("=" * 110)
    inst_df = pd.DataFrame(instance_prediction_records)
    print(inst_df.to_string(index=False))

    # Save CSV summaries for persistence
    cmp_df.to_csv("models/uap_vs_all_libraries_benchmark_summary.csv", index=False)
    inst_df.to_csv("models/uap_real_individual_predictions_audit.csv", index=False)
    print("\n[Audit Complete] Saved benchmark and instance prediction reports to models/")

if __name__ == "__main__":
    run_comprehensive_audit()
