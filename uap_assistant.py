"""
UAP 2.0 Multi-Domain Decision Intelligence & Diagnostic Assistant.
Interactive domain intelligence interface:
  - Query any of the 6 production domain models (Cardiology, Oncology, Credit, Agriculture, IoT, Real Estate)
  - Run Selective Prediction with the Abstention Gate G(x)
  - Inspect TreeSHAP Feature Attributions
  - Calculate Feasible Counterfactual Recourse (FACE)
  - Inspect the Meta-Learning Strategy Memory Bank
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from uap.engine import UAPEngine
from uap.core.memory import MetaLearningMemory
from data.dataset_loader import DATA_DIR


MODELS_DIR = Path(__file__).resolve().parent / "models"
MEMORY_PATH = str(MODELS_DIR / "meta_learning_memory.json")

DOMAINS = {
    "heart_disease": {
        "title": "Cardiology Healthcare (Heart Disease Risk)",
        "model_file": "heart_disease_uap.joblib",
        "test_file": "heart_disease_test.csv",
        "target_col": "target",
        "immutable": ["age", "sex"],
        "directions": {"trestbps": "negative_only", "chol": "negative_only"},
    },
    "breast_cancer": {
        "title": "Diagnostic Oncology (Breast Cancer Cellular Pathology)",
        "model_file": "breast_cancer_uap.joblib",
        "test_file": "breast_cancer_test.csv",
        "target_col": "target",
        "immutable": [],
        "directions": {},
    },
    "credit_risk": {
        "title": "Banking & Finance (Credit Card Default Risk)",
        "model_file": "credit_risk_uap.joblib",
        "test_file": "credit_risk_test.csv",
        "target_col": "target",
        "immutable": ["age"],
        "directions": {"credit_amount": "negative_only"},
    },
    "agriculture_crop": {
        "title": "Smart Agriculture (Soil & Climate Crop Recommendation)",
        "model_file": "agriculture_crop_uap.joblib",
        "test_file": "agriculture_crop_test.csv",
        "target_col": "target",
        "immutable": [],
        "directions": {},
    },
    "iot_maintenance": {
        "title": "Industrial IoT (Predictive Maintenance & Machine Failure)",
        "model_file": "iot_maintenance_uap.joblib",
        "test_file": "iot_maintenance_test.csv",
        "target_col": "target",
        "immutable": ["air_temperature_K"],
        "directions": {"tool_wear_min": "negative_only"},
    },
    "california_housing": {
        "title": "Macroeconomic Real Estate (California Housing Value)",
        "model_file": "california_housing_uap.joblib",
        "test_file": "california_housing_test.csv",
        "target_col": "MedHouseVal",
        "immutable": ["Latitude", "Longitude"],
        "directions": {},
    },
    "energy_grid": {
        "title": "Clean Energy & Utilities (Grid Load Demand Forecasting)",
        "model_file": "energy_grid_uap.joblib",
        "test_file": "energy_grid_test.csv",
        "target_col": "grid_load_mw",
        "immutable": ["hour_of_day", "is_weekend"],
        "directions": {"lag_1h_load_mw": "negative_only"},
    },
    "telecom_churn": {
        "title": "Telecom & SaaS (Customer Retention & Churn)",
        "model_file": "telecom_churn_uap.joblib",
        "test_file": "telecom_churn_test.csv",
        "target_col": "churn",
        "immutable": ["tenure_months"],
        "directions": {"monthly_charges": "negative_only", "customer_service_calls": "negative_only"},
    },
    "cyber_intrusion": {
        "title": "Cybersecurity & Defense (Network Intrusion Threat)",
        "model_file": "cyber_intrusion_uap.joblib",
        "test_file": "cyber_intrusion_test.csv",
        "target_col": "is_intrusion",
        "immutable": ["protocol_type"],
        "directions": {"failed_logins": "negative_only", "serror_rate": "negative_only"},
    },
    "fraud_detection": {
        "title": "Fintech & Banking (Card Transaction Fraud)",
        "model_file": "fraud_detection_uap.joblib",
        "test_file": "fraud_detection_test.csv",
        "target_col": "is_fraud",
        "immutable": ["distance_from_home_km"],
        "directions": {"transaction_amount": "negative_only"},
    },
    "air_quality": {
        "title": "Environmental Science (Urban PM2.5 Pollution Index)",
        "model_file": "air_quality_uap.joblib",
        "test_file": "air_quality_test.csv",
        "target_col": "pm2_5_aqi",
        "immutable": ["ambient_temp_c", "relative_humidity"],
        "directions": {"carbon_monoxide_co": "negative_only", "traffic_density_idx": "negative_only"},
    },
    "clinical_survival": {
        "title": "Medical Biostatistics (Clinical Survival & Time-to-Event)",
        "model_file": "clinical_survival_uap.joblib",
        "test_file": "clinical_survival_test.csv",
        "target_col": "death_event",
        "immutable": ["age", "sex"],
        "directions": {"serum_creatinine": "negative_only", "smoking": "negative_only"},
    },
}


def list_domains():
    print("\n" + "=" * 75)
    print("UAP 2.0 MULTI-DOMAIN INTELLIGENCE PORTFOLIO")
    print("=" * 75)
    for code, info in DOMAINS.items():
        model_exists = (MODELS_DIR / info["model_file"]).exists()
        status_tag = "[READY - MODEL SERIALIZED]" if model_exists else "[NOT TRAINED YET]"
        print(f"  * {code:<20}: {info['title']} {status_tag}")
    print()


def show_memory_status():
    mem = MetaLearningMemory(memory_file=MEMORY_PATH)
    print("\n" + "=" * 75)
    print(f"META-LEARNING STRATEGY MEMORY (Experiences: {len(mem)})")
    print("=" * 75)
    if len(mem) == 0:
        print("  Memory bank is empty. Run 'python train_all_domains.py' to populate.")
        return

    for i, r in enumerate(mem.records, 1):
        print(f"  {i}. Domain: {r['domain']:<32} | Dataset: {r['dataset_name']:<18} | Winning: {r['winning_model_name']}")
        print(f"     Score: {r.get('holdout_score', r.get('validation_score'))} | Samples: {r['samples']} | Features: {r['features']}")
    print()


def analyze_domain_sample(domain_code: str, sample_index: int = 0, inject_ood: bool = False):
    if domain_code not in DOMAINS:
        print(f"[Error] Unknown domain code: '{domain_code}'. Choose from: {list(DOMAINS.keys())}")
        return

    info = DOMAINS[domain_code]
    model_path = MODELS_DIR / info["model_file"]
    if not model_path.exists():
        print(f"[Error] Model file {model_path} does not exist. Train it first via 'python train_all_domains.py'.")
        return

    test_path = DATA_DIR / info["test_file"]
    if not test_path.exists():
        print(f"[Error] Holdout test file {test_path} does not exist.")
        return

    # Load Model & Holdout Data
    print(f"\n[Loading] Loading {info['title']} from {model_path.name}...")
    engine = UAPEngine.load(str(model_path))
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=[info["target_col"]])
    y_test = test_df[info["target_col"]]

    idx = min(sample_index, len(X_test) - 1)
    sample = X_test.iloc[idx].copy()
    actual_label = y_test.iloc[idx]

    if inject_ood:
        print("[Stress-Test] Injecting synthetic Out-Of-Distribution perturbation (+15.0 delta)...")
        sample = sample + 15.0

    print("\n" + "=" * 75)
    print(f"DIAGNOSTIC CONSULTATION: {info['title']}")
    print(f"Sample Record #{idx} | Target Column: '{info['target_col']}' (Actual: {actual_label})")
    print("=" * 75)

    # 1. Prediction & Abstention Gate
    safe_eval = engine.predict_safe(sample)
    print(f"\n[1. ABSTENTION SAFETY GATE G(x)]")
    print(f"  * Safety Status   : {safe_eval['status']}")
    print(f"  * Gate Decision   : {safe_eval['gate_decision']} (1 = Safe, 0 = Abstain)")
    print(f"  * Final Prediction: {safe_eval['prediction']}")
    print(f"  * Confidence Score: {safe_eval.get('confidence_score', 'N/A')}")
    print(f"  * OOD Shift Dist  : {safe_eval.get('ood_distance', 'N/A')}")
    print(f"  * Data Quality    : {safe_eval.get('quality_score', 'N/A')}")
    print(f"  * Recommended Action: {safe_eval['human_action']}")
    if safe_eval.get("reasons"):
        print(f"  * Safety Reasons  : {', '.join(safe_eval['reasons'])}")

    # 1.5 Conformal Risk Certification
    try:
        conformal_cert = engine.predict_conformal(sample, alpha=0.05)
        print(f"\n[1.5 CONFORMAL RISK CERTIFICATION (FINITE-SAMPLE GUARANTEE)]")
        print(f"  * Coverage Guarantee : {conformal_cert['coverage_guarantee']}")
        if "prediction_set" in conformal_cert:
            print(f"  * Guaranteed Set C(x): {conformal_cert['prediction_set']} (Size: {conformal_cert['set_size']})")
            print(f"  * Conformal Status   : {conformal_cert['status']} (Action: {conformal_cert['action']})")
        else:
            print(f"  * 95% Interval Bound : [{conformal_cert['lower_bound']} , {conformal_cert['upper_bound']}] (Margin: +/-{conformal_cert['conformal_margin']})")
    except Exception as e:
        print(f"  [Conformal Notice] Conformal certification: {e}")

    # 2. XAI Feature Importance
    try:
        explanation = engine.explain_instance(sample)
        top_contribs = explanation.get("contributions", {})
        sorted_contribs = sorted(top_contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        print(f"\n[2. TREESHAP EXPLAINABILITY (TOP DRIVING FACTORS)]")
        for f, val in sorted_contribs:
            impact_sign = "Increases Risk" if val > 0 else "Decreases Risk"
            print(f"  * {f:<22}: Attribution = {val:+7.4f} ({impact_sign}) | Feature Val = {sample.get(f, 'N/A')}")
    except Exception as e:
        print(f"  [XAI Notice] Attribution calculation skipped: {e}")

    # 2.5 Causal SCM Drivers
    try:
        dag = engine.get_causal_dag()
        drivers = dag.get("direct_target_drivers", {})
        if drivers:
            print(f"\n[2.5 CAUSAL SCM DIRECT DRIVERS (PEARL'S STRUCTURAL MODEL)]")
            sorted_drivers = sorted(drivers.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            for f, w in sorted_drivers:
                print(f"  * {f:<22}: Causal Weight = {w:+7.4f}")
    except Exception as e:
        pass

    # 3. Feasible Actionable Recourse (FACE)
    if "Classification" in str(type(engine.optimizer.trained_models.get(engine.ema.winning_model_name, None))) or safe_eval["prediction"] in [0, 1]:
        print(f"\n[3. FEASIBLE COUNTERFACTUAL RECOURSE (ACTION PLAN)]")
        current_pred = safe_eval["prediction"]
        if current_pred in [0, 1]:
            target_flip = 1 - int(current_pred)
            recourse = engine.generate_feasible_recourse(
                sample=sample,
                target_outcome=target_flip,
                immutable_features=info["immutable"],
                direction_constraints=info["directions"],
            )
            print(f"  * Status        : {recourse['status']}")
            print(f"  * Objective     : Flip outcome from {current_pred} -> {target_flip}")
            if recourse["status"] == "RECOURSE_FOUND":
                print(f"  * Total Cost    : {recourse['total_mad_cost']:.4f} MAD units")
                print("  * Prescriptive Step-by-Step Action Plan:")
                for step_i, act in enumerate(recourse["minimal_action_plan"], 1):
                    print(f"    Step {step_i}: Adjust '{act['feature']}' from {act['original_value']} to {act['suggested_value']} (delta: {act['change_delta']:+})")
            else:
                print(f"  * Reason        : {recourse.get('message', 'No feasible recourse path within manifold')}")
        else:
            print("  * System abstained on this sample; recourse cannot be computed for unknown inputs.")

    print("\n" + "=" * 75 + "\n")


def interactive_repl():
    print("\n" + "#" * 75)
    print("WELCOME TO UAP 2.0 DECISION INTELLIGENCE ASSISTANT")
    print("Type 'help' for commands, 'domains' to list, or 'quit' to exit.")
    print("#" * 75)

    while True:
        try:
            cmd = input("\n[UAP-CLI] > ").strip()
            if not cmd:
                continue
            if cmd in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            elif cmd in ["help", "h"]:
                print("\nAvailable Commands:")
                print("  domains                     - List all 6 real-world domains")
                print("  memory                      - Inspect Meta-Learning Strategy Memory")
                print("  consult <domain> <idx>      - Run safety, inference, SHAP & recourse")
                print("  stress <domain> <idx>       - Run with Out-of-Distribution test injection")
                print("  quit                        - Exit assistant\n")
            elif cmd == "domains":
                list_domains()
            elif cmd == "memory":
                show_memory_status()
            elif cmd.startswith("consult"):
                parts = cmd.split()
                if len(parts) < 2:
                    print("Usage: consult <domain_code> [sample_idx]")
                else:
                    d_code = parts[1]
                    s_idx = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                    analyze_domain_sample(d_code, sample_index=s_idx, inject_ood=False)
            elif cmd.startswith("stress"):
                parts = cmd.split()
                if len(parts) < 2:
                    print("Usage: stress <domain_code> [sample_idx]")
                else:
                    d_code = parts[1]
                    s_idx = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                    analyze_domain_sample(d_code, sample_index=s_idx, inject_ood=True)
            else:
                print(f"Unknown command: '{cmd}'. Type 'help' for options.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="UAP 2.0 Multi-Domain Decision Assistant")
    parser.add_argument("--list-domains", action="store_true", help="List all available domains")
    parser.add_argument("--memory-status", action="store_true", help="Inspect Meta-Learning Strategy Memory")
    parser.add_argument("--domain", type=str, help="Domain code to analyze (e.g. heart_disease, iot_maintenance)")
    parser.add_argument("--sample", type=int, default=0, help="Sample index in holdout set to analyze")
    parser.add_argument("--stress-ood", action="store_true", help="Inject out-of-distribution shift to test Abstention Gate")
    parser.add_argument("--stream-scale", type=int, nargs="?", const=5_000_000, default=None, help="Demonstrate infinite/astronomical online streaming convergence (default 5,000,000 samples)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive REPL mode")

    args = parser.parse_args()

    if args.list_domains:
        list_domains()
    elif args.memory_status:
        show_memory_status()
    elif args.stream_scale is not None:
        from train_infinite_stream import InfiniteStreamingEngine
        trainer = InfiniteStreamingEngine(n_features=12, chunk_size=100_000)
        trainer.train_asymptotic_stream(target_demonstration_samples=args.stream_scale)
    elif args.domain:
        analyze_domain_sample(args.domain, sample_index=args.sample, inject_ood=args.stress_ood)
    elif args.interactive or len(sys.argv) == 1:
        interactive_repl()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
