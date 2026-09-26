"""
UAP 2.0 Production Training Pipeline on Real-World Datasets.
Trains, evaluates, and serializes production-grade models on:
  1. UCI Cleveland Heart Disease (Cardiology Risk Classification)
  2. Breast Cancer Wisconsin (Diagnostic Oncology Classification)
  3. California Housing (Macroeconomic Real Estate Regression)
Saves serialized models and manifolds to 'models/*.joblib'.
"""

from pathlib import Path
import json
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, root_mean_squared_error

from uap.engine import UAPEngine
from data.dataset_loader import prepare_all_datasets, DATA_DIR


MODELS_DIR = Path(__file__).resolve().parent / "models"


def ensure_models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def train_production_heart_disease_model() -> str:
    print("\n" + "=" * 75)
    print("TRAINING PRODUCTION MODEL 1: UCI CLEVELAND HEART DISEASE (CARDIOLOGY)")
    print("=" * 75)

    train_path = DATA_DIR / "heart_disease_train.csv"
    test_path = DATA_DIR / "heart_disease_test.csv"
    if not train_path.exists():
        prepare_all_datasets()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    print(f"Loaded Train: {train_df.shape} | Holdout Test: {test_df.shape}")
    print(f"Disease Class Balance:\n{train_df['target'].value_counts(normalize=True).round(3)}")

    # Train UAP Engine
    t0 = time.time()
    engine = UAPEngine(n_trials=8, timeout_sec=25)
    engine.fit(train_df, target_column="target")
    elapsed = time.time() - t0

    # Test Evaluation
    preds = engine.predict(X_test)
    probs = engine.predict_proba(X_test)
    prob_pos = probs[:, 1] if probs is not None else preds

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    auc = roc_auc_score(y_test, prob_pos)

    print("\nHoldout Test Performance:")
    print(f"  * Accuracy: {acc:.4f}")
    print(f"  * Macro F1: {f1:.4f}")
    print(f"  * ROC-AUC : {auc:.4f}")
    print(f"  * RDS     : {engine.ema.performance_metrics.robustness_decay_score:.4f}")
    print(f"  * Strategy: {engine.ema.winning_model_name}")

    # Test Abstention Gate on real patients
    safe_normal = engine.predict_safe(X_test.iloc[0])
    print(f"\nReal Patient Sample 0 Gate Decision: {safe_normal['status']} (Action: {safe_normal['human_action']})")

    # Test Recourse for patient with disease (target = 0 -> healthy)
    disease_indices = [i for i, p in enumerate(preds) if p == 1]
    if disease_indices:
        unfavorable_patient = X_test.iloc[disease_indices[0]]
        recourse = engine.generate_feasible_recourse(
            sample=unfavorable_patient,
            target_outcome=0,  # Flip to healthy
            immutable_features=["age", "sex"],  # Cannot change age or biological sex
            direction_constraints={"trestbps": "negative_only", "chol": "negative_only"}
        )
        print(f"\nRecourse to Reduce Heart Disease Risk:")
        print(f"  Status: {recourse['status']} | Total MAD Cost: {recourse.get('total_mad_cost')}")
        if recourse["status"] == "RECOURSE_FOUND":
            for act in recourse["minimal_action_plan"]:
                print(f"  * {act['feature']}: change from {act['original_value']} -> {act['suggested_value']} (delta: {act['change_delta']:+})")

    # Serialize Model
    ensure_models_dir()
    save_path = str(MODELS_DIR / "heart_disease_uap.joblib")
    engine.save(save_path)
    return save_path


def train_production_breast_cancer_model() -> str:
    print("\n" + "=" * 75)
    print("TRAINING PRODUCTION MODEL 2: BREAST CANCER WISCONSIN (DIAGNOSTIC ONCOLOGY)")
    print("=" * 75)

    train_path = DATA_DIR / "breast_cancer_train.csv"
    test_path = DATA_DIR / "breast_cancer_test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    print(f"Loaded Train: {train_df.shape} | Holdout Test: {test_df.shape}")

    engine = UAPEngine(n_trials=8, timeout_sec=25)
    engine.fit(train_df, target_column="target")

    preds = engine.predict(X_test)
    probs = engine.predict_proba(X_test)
    prob_pos = probs[:, 1] if probs is not None else preds

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    auc = roc_auc_score(y_test, prob_pos)

    print("\nHoldout Test Performance:")
    print(f"  * Accuracy: {acc:.4f}")
    print(f"  * Macro F1: {f1:.4f}")
    print(f"  * ROC-AUC : {auc:.4f}")
    print(f"  * RDS     : {engine.ema.performance_metrics.robustness_decay_score:.4f}")

    save_path = str(MODELS_DIR / "breast_cancer_uap.joblib")
    engine.save(save_path)
    return save_path


def train_production_california_housing_model() -> str:
    print("\n" + "=" * 75)
    print("TRAINING PRODUCTION MODEL 3: CALIFORNIA HOUSING (REAL ESTATE REGRESSION)")
    print("=" * 75)

    train_path = DATA_DIR / "california_housing_train.csv"
    test_path = DATA_DIR / "california_housing_test.csv"
    # Take a 5000-sample balanced subset for fast training
    train_df = pd.read_csv(train_path).sample(5000, random_state=42)
    test_df = pd.read_csv(test_path).sample(1000, random_state=42)
    X_test = test_df.drop(columns=["MedHouseVal"])
    y_test = test_df["MedHouseVal"]

    print(f"Loaded Train: {train_df.shape} | Holdout Test: {test_df.shape}")

    engine = UAPEngine(n_trials=8, timeout_sec=25)
    engine.fit(train_df, target_column="MedHouseVal")

    preds = engine.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)

    print("\nHoldout Test Performance:")
    print(f"  * R2 Score : {r2:.4f}")
    print(f"  * RMSE ($): {rmse:.4f}")
    print(f"  * RDS      : {engine.ema.performance_metrics.robustness_decay_score:.4f}")

    save_path = str(MODELS_DIR / "california_housing_uap.joblib")
    engine.save(save_path)
    return save_path


def main():
    ensure_models_dir()
    p1 = train_production_heart_disease_model()
    p2 = train_production_breast_cancer_model()
    p3 = train_production_california_housing_model()

    print("\n" + "=" * 75)
    print("[SUCCESS] ALL 3 REAL PRODUCTION MODELS TRAINED & PERSISTED!")
    print("=" * 75)
    print(f"  1. Heart Disease Model    : {p1}")
    print(f"  2. Breast Cancer Model    : {p2}")
    print(f"  3. California Housing Model: {p3}")


if __name__ == "__main__":
    main()
