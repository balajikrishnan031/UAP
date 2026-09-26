"""
Master Multi-Domain Real-World Training Pipeline for UAP 2.0.
Trains, evaluates, stress-tests, and persists production models across 6 real-world domains:
  1. Healthcare (Cardiology): UCI Cleveland Heart Disease
  2. Oncology (Cell Biology): Breast Cancer Wisconsin
  3. Banking & Credit Risk: German Credit Default
  4. Smart Agriculture: Soil Nutrients & Climate Crop Recommendation
  5. Industrial IoT: AI4I Equipment Predictive Maintenance
  6. Macroeconomics / Real Estate: California Housing Census
Populates the Meta-Learning Strategy Memory (MLS-Memory) for zero-shot strategy recall.
"""

from pathlib import Path
import time
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, root_mean_squared_error

from uap.engine import UAPEngine
from uap.core.contracts import ProblemType
from uap.core.memory import MetaLearningMemory
from data.dataset_loader import prepare_all_domains, DATA_DIR


MODELS_DIR = Path(__file__).resolve().parent / "models"
MEMORY_PATH = str(MODELS_DIR / "meta_learning_memory.json")


def ensure_models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def train_domain(
    domain_name: str,
    dataset_code: str,
    target_col: str,
    is_classification: bool,
    immutable_features: List[str],
    direction_constraints: Dict[str, str],
    max_train_samples: int = 5000,
    n_trials: int = 8,
    timeout_sec: int = 25,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"TRAINING DOMAIN: {domain_name.upper()} ({dataset_code})")
    print("=" * 80)

    train_path = DATA_DIR / f"{dataset_code}_train.csv"
    test_path = DATA_DIR / f"{dataset_code}_test.csv"
    if not train_path.exists():
        prepare_all_domains()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Subsample if large for fast iterative training
    if len(train_df) > max_train_samples:
        train_df = train_df.sample(max_train_samples, random_state=42)
    if len(test_df) > 1000:
        test_df = test_df.sample(1000, random_state=42)

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print(f"Loaded Train: {train_df.shape} | Holdout Test: {test_df.shape}")

    # Initialize UAP Master Engine with Meta-Learning Memory
    memory = MetaLearningMemory(memory_file=MEMORY_PATH)
    engine = UAPEngine(n_trials=n_trials, timeout_sec=timeout_sec, memory=memory)

    t0 = time.time()
    engine.fit(
        df=train_df,
        target_column=target_col,
        domain_tag=domain_name,
        dataset_name=dataset_code,
    )
    elapsed = time.time() - t0

    # Holdout Test Evaluation
    preds = engine.predict(X_test)
    probs = engine.predict_proba(X_test)
    prob_pos = probs[:, 1] if (probs is not None and probs.shape[1] > 1) else preds

    metrics = {}
    if is_classification:
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(y_test, prob_pos)
        except Exception:
            auc = float("nan")
        metrics = {"Accuracy": acc, "Macro_F1": f1, "ROC_AUC": auc}
        print(f"\nHoldout Test Performance: Accuracy={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")
    else:
        r2 = r2_score(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        metrics = {"R2_Score": r2, "RMSE": rmse}
        print(f"\nHoldout Test Performance: R2={r2:.4f} | RMSE={rmse:.4f}")

    rds = engine.ema.performance_metrics.robustness_decay_score
    print(f"Robustness Decay Score (RDS): {rds:.4f} (Under Noise & Perturbations)")
    print(f"Winning Pipeline Strategy  : {engine.ema.winning_model_name}")

    # Safety: Test Abstention Gate on normal sample
    test_sample = X_test.iloc[0]
    safe_eval = engine.predict_safe(test_sample)
    print(f"Sample 0 Safety Gate Decision: {safe_eval['status']} (Gate: {safe_eval['gate_decision']}, Action: {safe_eval['human_action']})")

    # Actionable Recourse for classification domains
    recourse_info = "N/A (Regression)"
    if is_classification and len(preds) > 0:
        # Pick unfavorable sample if possible
        unfav_indices = [i for i, p in enumerate(preds) if p == 1]
        target_flip = 0
        if not unfav_indices:
            unfav_indices = [0]
            target_flip = 1 - int(preds[0])

        unfav_sample = X_test.iloc[unfav_indices[0]]
        recourse = engine.generate_feasible_recourse(
            sample=unfav_sample,
            target_outcome=target_flip,
            immutable_features=immutable_features,
            direction_constraints=direction_constraints,
        )
        recourse_info = f"{recourse['status']} (MAD Cost: {recourse.get('total_mad_cost')})"
        print(f"Feasible Counterfactual Recourse: {recourse_info}")
        if recourse["status"] == "RECOURSE_FOUND":
            for act in recourse["minimal_action_plan"][:2]:
                print(f"  * {act['feature']}: {act['original_value']} -> {act['suggested_value']} (delta: {act['change_delta']:+})")

    # Serialize Model Artifact
    save_path = str(MODELS_DIR / f"{dataset_code}_uap.joblib")
    engine.save(save_path)
    print(f"[Serialized] Saved production model -> {save_path}")

    return {
        "Domain": domain_name,
        "Code": dataset_code,
        "Problem": "Classification" if is_classification else "Regression",
        "Train_Rows": len(train_df),
        "Features": train_df.shape[1] - 1,
        "Winning_Model": engine.ema.winning_model_name,
        "Primary_Score": round(metrics.get("Accuracy", metrics.get("R2_Score", 0.0)), 4),
        "Score_Metric": "Accuracy" if is_classification else "R2",
        "RDS": round(rds, 4),
        "Training_Time_Sec": round(elapsed, 2),
        "Safety_Gate": safe_eval["status"],
        "Recourse_Status": recourse_info,
        "Model_Path": save_path,
    }


def main():
    ensure_models_dir()
    prepare_all_domains()

    domains = [
        {
            "domain_name": "Healthcare (Cardiology)",
            "dataset_code": "heart_disease",
            "target_col": "target",
            "is_classification": True,
            "immutable_features": ["age", "sex"],
            "direction_constraints": {"trestbps": "negative_only", "chol": "negative_only"},
            "max_train_samples": 300,
        },
        {
            "domain_name": "Oncology (Cell Pathology)",
            "dataset_code": "breast_cancer",
            "target_col": "target",
            "is_classification": True,
            "immutable_features": [],
            "direction_constraints": {},
            "max_train_samples": 600,
        },
        {
            "domain_name": "Banking & Finance (Credit Risk)",
            "dataset_code": "credit_risk",
            "target_col": "target",
            "is_classification": True,
            "immutable_features": ["age"],
            "direction_constraints": {"credit_amount": "negative_only"},
            "max_train_samples": 1000,
        },
        {
            "domain_name": "Smart Agriculture (Crop Yield)",
            "dataset_code": "agriculture_crop",
            "target_col": "target",
            "is_classification": True,
            "immutable_features": [],
            "direction_constraints": {},
            "max_train_samples": 2000,
        },
        {
            "domain_name": "Industrial IoT (Predictive Maint.)",
            "dataset_code": "iot_maintenance",
            "target_col": "target",
            "is_classification": True,
            "immutable_features": ["air_temperature_K"],
            "direction_constraints": {"tool_wear_min": "negative_only"},
            "max_train_samples": 4000,
        },
        {
            "domain_name": "Economics (Real Estate)",
            "dataset_code": "california_housing",
            "target_col": "MedHouseVal",
            "is_classification": False,
            "immutable_features": ["Latitude", "Longitude"],
            "direction_constraints": {},
            "max_train_samples": 5000,
        },
        {
            "domain_name": "Clean Energy & Utilities",
            "dataset_code": "energy_grid",
            "target_col": "grid_load_mw",
            "is_classification": False,
            "immutable_features": ["hour_of_day", "is_weekend"],
            "direction_constraints": {"lag_1h_load_mw": "negative_only"},
            "max_train_samples": 3600,
        },
        {
            "domain_name": "Telecom & SaaS (Retention)",
            "dataset_code": "telecom_churn",
            "target_col": "churn",
            "is_classification": True,
            "immutable_features": ["tenure_months"],
            "direction_constraints": {"monthly_charges": "negative_only", "customer_service_calls": "negative_only"},
            "max_train_samples": 2800,
        },
        {
            "domain_name": "Cybersecurity & Defense",
            "dataset_code": "cyber_intrusion",
            "target_col": "is_intrusion",
            "is_classification": True,
            "immutable_features": ["protocol_type"],
            "direction_constraints": {"failed_logins": "negative_only", "serror_rate": "negative_only"},
            "max_train_samples": 4000,
        },
        {
            "domain_name": "Fintech & Banking (Fraud)",
            "dataset_code": "fraud_detection",
            "target_col": "is_fraud",
            "is_classification": True,
            "immutable_features": ["distance_from_home_km"],
            "direction_constraints": {"transaction_amount": "negative_only"},
            "max_train_samples": 4800,
        },
        {
            "domain_name": "Environmental Science (AQI)",
            "dataset_code": "air_quality",
            "target_col": "pm2_5_aqi",
            "is_classification": False,
            "immutable_features": ["ambient_temp_c", "relative_humidity"],
            "direction_constraints": {"carbon_monoxide_co": "negative_only", "traffic_density_idx": "negative_only"},
            "max_train_samples": 2400,
        },
        {
            "domain_name": "Medical Biostatistics (Survival)",
            "dataset_code": "clinical_survival",
            "target_col": "death_event",
            "is_classification": True,
            "immutable_features": ["age", "sex"],
            "direction_constraints": {"serum_creatinine": "negative_only", "smoking": "negative_only"},
            "max_train_samples": 250,
        },
    ]

    results = []
    print("\n" + "#" * 80)
    print("STARTING UAP 3.0 MULTI-DOMAIN UNIVERSAL TRAINING SUITE (12 DOMAINS)")
    print("#" * 80)

    for cfg in domains:
        res = train_domain(
            domain_name=cfg["domain_name"],
            dataset_code=cfg["dataset_code"],
            target_col=cfg["target_col"],
            is_classification=cfg["is_classification"],
            immutable_features=cfg["immutable_features"],
            direction_constraints=cfg["direction_constraints"],
            max_train_samples=cfg["max_train_samples"],
        )
        results.append(res)

    # Display Benchmark Table
    summary_df = pd.DataFrame(results)
    print("\n\n" + "=" * 90)
    print("UAP 2.0 MULTI-DOMAIN PRODUCTION BENCHMARK SUMMARY")
    print("=" * 90)
    cols = ["Domain", "Problem", "Features", "Winning_Model", "Score_Metric", "Primary_Score", "RDS", "Safety_Gate", "Training_Time_Sec"]
    print(summary_df[cols].to_string(index=False))

    # Save summary report to CSV
    report_csv = MODELS_DIR / "multi_domain_benchmark_report.csv"
    summary_df.to_csv(report_csv, index=False)
    print(f"\n[Saved] Detailed benchmark summary saved to {report_csv}")

    # Inspect Meta-Learning Memory
    memory = MetaLearningMemory(memory_file=MEMORY_PATH)
    print(f"[MLS-Memory] Knowledge Bank now holds {len(memory)} dataset experiences in {MEMORY_PATH}")


if __name__ == "__main__":
    main()
