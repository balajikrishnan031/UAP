"""
UAP 2.0 Real-World CLI Prediction & Recourse Tool.
Performs instant, production inference on real input records using persisted model weights:
  1. Loads serialized UAP model from disk in milliseconds.
  2. Evaluates input sample through the Abstention Gate G(x) (Selective Prediction).
  3. Emits prediction, calibrated confidence score, and OOD distance.
  4. Automatically computes Feasible Counterfactual Recourse if the outcome is unfavorable.
"""

import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from uap.engine import UAPEngine


def parse_args():
    parser = argparse.ArgumentParser(description="UAP 2.0 Production Real-World Prediction CLI")
    parser.add_argument(
        "--model",
        type=str,
        default="models/heart_disease_uap.joblib",
        help="Path to serialized UAP model file (*.joblib)"
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="JSON string of feature key-value pairs for single record scoring"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="CSV file path to score multiple records"
    )
    parser.add_argument(
        "--target_goal",
        type=int,
        default=0,
        help="Desired target outcome for recourse optimization (e.g. 0 = healthy)"
    )
    return parser.parse_args()


def score_single_record(engine: UAPEngine, features: dict, target_goal: int):
    sample = pd.Series(features)
    print("\n" + "=" * 70)
    print("UAP 2.0 PRODUCTION INFERENCE AUDIT")
    print("=" * 70)
    print(f"Target Feature Schema: {len(engine.feature_names)} features registered")

    # Ensure all required features are present
    missing_cols = [c for c in engine.feature_names if c not in sample]
    if missing_cols:
        print(f"[Warning] Missing features in input: {missing_cols}. Missing values will be imputed with zero.")
        for col in missing_cols:
            sample[col] = 0.0

    sample = sample[engine.feature_names]

    # 1. Evaluate through Abstention Gate
    gate_eval = engine.predict_safe(sample)

    print("\n[Stage 1: The Abstention Gate G(x)]")
    print(f"  Decision Status : {gate_eval['status']}")
    print(f"  Gate Code G(x)  : {gate_eval['gate_decision']}")
    print(f"  OOD Distance    : {gate_eval['ood_distance']} (Threshold: {engine.abstention_gate.tau_ood})")
    print(f"  Confidence Score: {gate_eval['confidence_score']}")
    print(f"  Instance Quality: {gate_eval['quality_score']}")
    print(f"  Human Action    : {gate_eval['human_action']}")

    if gate_eval["status"] == "ABSTAIN":
        print(f"\n[ALERT] Model execution halted by Abstention Gate!")
        print(f"Reasons: {gate_eval['reasons']}")
        print("Record redirected to human medical/domain expert inspection.")
        return

    # 2. Prediction Verdict
    print("\n[Stage 2: Prediction Verdict]")
    pred_val = gate_eval["prediction"]
    print(f"  Predicted Outcome: {pred_val}")

    # 3. Feasible Actionable Recourse
    if pred_val != target_goal:
        print(f"\n[Stage 3: Feasible Actionable Recourse (Target: {target_goal})]")
        # Identify common immutable features
        immutable = [f for f in ["age", "sex", "race", "HouseAge"] if f in engine.feature_names]
        print(f"  Freezing Immutable Features: {immutable}")

        recourse = engine.generate_feasible_recourse(
            sample=sample,
            target_outcome=target_goal,
            immutable_features=immutable,
        )

        print(f"  Recourse Status: {recourse['status']}")
        if recourse["status"] == "RECOURSE_FOUND":
            print(f"  Normalized MAD Cost: {recourse['total_mad_cost']}")
            print("  Prescriptive Interventions to Reverse Outcome:")
            for act in recourse["minimal_action_plan"]:
                print(f"    * {act['feature']}: change from {act['original_value']} -> {act['suggested_value']} (delta: {act['change_delta']:+})")
        else:
            print(f"  {recourse.get('message', 'Outcome could not be reversed within plausible bounds.')}")
    else:
        print("\n[Stage 3: Actionable Recourse]")
        print("  Current outcome already matches target favorable state. No intervention required.")


def score_csv_file(engine: UAPEngine, test_path: str):
    df = pd.read_csv(test_path)
    target_col = engine.target_name
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df
        y = None

    print(f"\nScoring batch of {len(X)} records from: {test_path}")
    preds = engine.predict(X)
    probs = engine.predict_proba(X)

    # Evaluate Abstention Gate across batch
    abstain_count = 0
    for idx in range(len(X)):
        eval_res = engine.predict_safe(X.iloc[idx])
        if eval_res["status"] == "ABSTAIN":
            abstain_count += 1

    print(f"Batch Scoring Complete:")
    print(f"  * Total Processed : {len(X)}")
    print(f"  * Safe Emitted    : {len(X) - abstain_count} ({((len(X) - abstain_count)/len(X))*100:.1f}%)")
    print(f"  * Abstained / OOD : {abstain_count} ({(abstain_count/len(X))*100:.1f}%)")

    if y is not None:
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y, preds)
        print(f"  * Batch Accuracy  : {acc:.4f}")


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model file not found at {model_path}. Please run 'python train_real_models.py' first.")
        sys.exit(1)

    print(f"Loading trained production model from: {model_path}...")
    engine = UAPEngine.load(str(model_path))

    if args.sample:
        features = json.loads(args.sample)
        score_single_record(engine, features, args.target_goal)
    elif args.test_file:
        score_csv_file(engine, args.test_file)
    else:
        # Default: pick the first record from the holdout test set
        default_test = Path("data/real_world/heart_disease_test.csv")
        if default_test.exists():
            test_df = pd.read_csv(default_test)
            patient = test_df.iloc[0].to_dict()
            if "target" in patient:
                del patient["target"]
            print(f"No sample provided. Testing with first patient record from {default_test}:")
            score_single_record(engine, patient, args.target_goal)
        else:
            print("Please specify --sample '<json>' or --test_file <path.csv>")


if __name__ == "__main__":
    main()
