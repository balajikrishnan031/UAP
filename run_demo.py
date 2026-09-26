"""
End-to-End Demonstration of the Universal Adaptive Prediction (UAP) Framework.
Runs Module 1 -> Module 2 -> Module 3 -> Module 4 and displays all inter-module artifacts.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from uap.engine import UAPEngine
from synthetic_data.generator import UAPSyntheticDataGenerator


def main():
    print("=" * 70)
    print("      UNIVERSAL ADAPTIVE PREDICTION (UAP) FRAMEWORK DEMO")
    print("=" * 70)

    # 1. Generate Challenge Dataset (Imbalanced, Non-Linear Binary Classification)
    print("\n[Step 0] Generating Challenge Synthetic Dataset...")
    generator = UAPSyntheticDataGenerator()
    df = generator.generate_imbalanced_non_linear_classification(
        n_samples=2500,
        n_features=12,
        minority_ratio=0.04,  # Severe 4% imbalance
        label_noise=0.03,
        random_state=42,
    )
    print(f"Dataset Shape: {df.shape}")
    print(f"Target Distribution:\n{df['target'].value_counts(normalize=True).round(4)}")

    # 2. Initialize and Train UAP Closed-Loop Engine
    print("\n[Step 1] Initializing UAP Engine and Running Closed-Loop Pipeline...")
    engine = UAPEngine(n_trials=8, timeout_sec=30)
    engine.fit(df, target_column="target")

    # 3. Inspect Module 1: DCV Artifact
    print("\n" + "=" * 70)
    print("ARTIFACT 1: DATASET CHARACTERISTIC VECTOR (DCV) [Module 1 Output]")
    print("=" * 70)
    print(json.dumps(engine.dcv.to_dict(), indent=2))

    # 4. Inspect Module 2: MSC Artifact
    print("\n" + "=" * 70)
    print("ARTIFACT 2: MODEL STRATEGY CONFIGURATION (MSC) [Module 2 Output]")
    print("=" * 70)
    print(json.dumps(engine.msc.to_dict(), indent=2))

    # 5. Inspect Module 3: EMA Artifact
    print("\n" + "=" * 70)
    print("ARTIFACT 3: EVALUATED MODEL ARTIFACT (EMA) [Module 3 Output]")
    print("=" * 70)
    print(json.dumps(engine.ema.to_dict(), indent=2))

    # 6. Module 4: Explainability (TreeSHAP)
    print("\n" + "=" * 70)
    print("MODULE 4 - PART 1: GLOBAL SHAP FEATURE IMPORTANCE")
    print("=" * 70)
    X = df.drop(columns=["target"])
    global_importance = engine.explain_global(X.head(50))
    for feat, imp in list(global_importance.items())[:6]:
        print(f"  * {feat:<18}: {imp:.4f}")

    sample_instance = X.iloc[0]
    print("\nLOCAL INSTANCE EXPLANATION (Sample 0):")
    explanation = engine.explain_instance(sample_instance)
    print(f"  Base Expected Value: {explanation['base_value']}")
    print(f"  Top Positive Driver: {explanation.get('top_positive_driver')}")
    print(f"  Top Negative Driver: {explanation.get('top_negative_driver')}")

    # 7. Module 4 - Part 3: The Abstention Gate (Selective Prediction)
    print("\n" + "=" * 70)
    print("MODULE 4 - PART 3: THE ABSTENTION GATE G(x) [SELECTIVE PREDICTION]")
    print("=" * 70)
    # Test A: In-distribution sample
    safe_eval_normal = engine.predict_safe(sample_instance)
    print("Normal Sample Safety Evaluation:")
    print(f"  Gate Decision : {safe_eval_normal['status']} (Code: {safe_eval_normal['gate_decision']})")
    print(f"  Confidence    : {safe_eval_normal['confidence_score']} | OOD Dist: {safe_eval_normal['ood_distance']} | Quality: {safe_eval_normal['quality_score']}")
    print(f"  Human Action  : {safe_eval_normal['human_action']}")

    # Test B: Corrupted Out-of-Distribution Sample
    corrupted_sample = sample_instance.copy()
    corrupted_sample.iloc[:3] += 20.0  # Extreme sensor spike / OOD shift
    safe_eval_corrupted = engine.predict_safe(corrupted_sample)
    print("\nExtreme OOD Sample Safety Evaluation:")
    print(f"  Gate Decision : {safe_eval_corrupted['status']} (Code: {safe_eval_corrupted['gate_decision']})")
    print(f"  Prediction    : {safe_eval_corrupted['prediction']}")
    print(f"  OOD Distance  : {safe_eval_corrupted['ood_distance']}")
    print(f"  Human Action  : {safe_eval_corrupted['human_action']}")
    print(f"  Failure Reason: {safe_eval_corrupted['reasons']}")

    # 8. Module 4 - Part 5: Feasible Counterfactual Engine (FACE)
    print("\n" + "=" * 70)
    print("MODULE 4 - PART 5: FEASIBLE COUNTERFACTUAL ENGINE (FACE)")
    print("=" * 70)
    immutable_cols = [list(X.columns)[0], list(X.columns)[1]]  # e.g. Age, History
    print(f"Freezing Immutable Features (delta_i = 0): {immutable_cols}")
    all_preds = engine.predict(X)
    unfav_indices = [i for i, p in enumerate(all_preds) if p == 0]
    recourse_target_sample = X.iloc[unfav_indices[0]] if unfav_indices else sample_instance

    recourse_plan = engine.generate_feasible_recourse(
        sample=recourse_target_sample,
        target_outcome=1,
        immutable_features=immutable_cols,
    )
    print(f"Recourse Status   : {recourse_plan['status']}")
    if recourse_plan["status"] == "RECOURSE_FOUND":
        print(f"Total MAD Cost    : {recourse_plan['total_mad_cost']}")
        print("Actionable Interventions:")
        for act in recourse_plan["minimal_action_plan"]:
            print(f"  * {act['feature']}: change from {act['original_value']} -> {act['suggested_value']} (delta: {act['change_delta']:+})")

    # 8. Module 4: Real-Time Drift Monitoring (PSI)
    print("\n" + "=" * 70)
    print("MODULE 4 - PART 4: REAL-TIME DATA DRIFT MONITORING (PSI)")
    print("=" * 70)
    # Test A: In-distribution batch
    in_dist_batch = X.sample(300, random_state=123)
    drift_res_a = engine.monitor_drift(in_dist_batch)
    print(f"In-Distribution Stream PSI: {drift_res_a['overall_psi']} | Status: {drift_res_a['status']}")

    # Test B: Shifted Out-of-Distribution batch (simulating sensor shift / demographic drift)
    shifted_batch = in_dist_batch.copy()
    shifted_batch[list(X.columns)[:3]] += 3.5
    drift_res_b = engine.monitor_drift(shifted_batch)
    print(f"Shifted OOD Stream PSI    : {drift_res_b['overall_psi']} | Status: {drift_res_b['status']}")
    print(f"Alert Features Detected   : {drift_res_b['alert_features']}")

    print("\n" + "=" * 70)
    print("[SUCCESS] UAP DEMONSTRATION EXECUTED SUCCESSFULLY ACROSS ALL 4 MODULES!")
    print("=" * 70)


if __name__ == "__main__":
    main()
