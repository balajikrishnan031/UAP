"""
UCI Heart Disease (Cleveland 303 instances) Strict Benchmark:
Comparing 12-Feature & 13-Feature configurations across:
1. SVM
2. Random Forest
3. XGBoost
4. Logistic Regression
5. TMRM (Topological Manifold Resonance Machine)
6. UAP (Universal Autonomous Pipeline)

Metrics: Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, ROC-AUC, 5-Fold Cross Validation.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uap import TMRM


def load_cleveland_data():
    raw_path = PROJECT_ROOT / "data" / "heart_disease_extracted" / "processed.cleveland.data"
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope",
        "ca", "thal", "target"
    ]
    df = pd.read_csv(raw_path, names=column_names, na_values="?")
    # Impute missing values with median (ca and thal have small missing counts in Cleveland)
    df["ca"] = df["ca"].fillna(df["ca"].median())
    df["thal"] = df["thal"].fillna(df["thal"].median())
    # Binary classification target: 0 vs 1
    df["target"] = (df["target"] > 0).astype(int)
    return df


def calculate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except:
        roc = 0.5
    
    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall (Sensitivity)": rec,
        "Specificity": spec,
        "F1-Score": f1,
        "ROC-AUC": roc,
        "Confusion_Matrix": (tn, fp, fn, tp)
    }


def run_benchmark(feature_cols, config_name):
    print("=" * 80)
    print(f"BENCHMARK: {config_name} ({len(feature_cols)} Features | 303 Samples)")
    print(f"Features: {feature_cols}")
    print("=" * 80)

    df = load_cleveland_data()
    X = df[feature_cols].copy()
    y = df["target"].copy()

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Preprocessing: Standard Scaler fitted only on Train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "SVM (RBF Kernel)": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "TMRM (Topological Resonance)": TMRM(task_type="classification", n_resonators="auto", random_state=42),
    }

    results = []

    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Test Inference
        y_pred = model.predict(X_test_scaled)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, "predict_probability"):
            y_prob = model.predict_probability(X_test_scaled)[:, 1]
        else:
            y_prob = y_pred

        metrics = calculate_metrics(y_test, y_pred, y_prob)

        # 5-Fold Stratified Cross Validation Accuracy
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_val_s = sc.transform(X_val)
            
            # clone/fresh model
            if "TMRM" in name:
                m_cv = TMRM(task_type="classification", n_resonators="auto", random_state=42)
            elif "SVM" in name:
                m_cv = SVC(probability=True, random_state=42)
            elif "Random Forest" in name:
                m_cv = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            elif "XGBoost" in name:
                m_cv = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
            else:
                m_cv = LogisticRegression(random_state=42, max_iter=1000)

            m_cv.fit(X_tr_s, y_tr)
            val_pred = m_cv.predict(X_val_s)
            cv_scores.append(accuracy_score(y_val, val_pred))

        metrics["5-Fold CV Mean"] = np.mean(cv_scores)
        metrics["5-Fold CV Std"] = np.std(cv_scores)
        results.append((name, metrics))

    # Print Table
    print(f"\n{'Model':<30} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'Spec':<8} | {'F1':<8} | {'ROC-AUC':<8} | {'5-Fold CV':<12}")
    print("-" * 105)
    for name, m in results:
        cv_str = f"{m['5-Fold CV Mean']*100:.1f} +- {m['5-Fold CV Std']*100:.1f}%"
        print(f"{name:<30} | {m['Accuracy']*100:>6.2f}%   | {m['Precision']*100:>6.2f}%   | {m['Recall (Sensitivity)']*100:>6.2f}%   | {m['Specificity']*100:>6.2f}% | {m['F1-Score']:>6.4f} | {m['ROC-AUC']:>6.4f} | {cv_str:<12}")

    print("\nConfusion Matrices [TN, FP, FN, TP]:")
    for name, m in results:
        tn, fp, fn, tp = m["Confusion_Matrix"]
        print(f"  * {name:<30}: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("\n")


if __name__ == "__main__":
    # 1. 12-Feature Configuration (Dropping 'thal' as requested)
    features_12 = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca"
    ]
    run_benchmark(features_12, "12-FEATURE CLEVELAND CONFIGURATION (WITHOUT 'thal')")

    # 2. 13-Feature Configuration (Full UCI Standard with 'thal')
    features_13 = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    run_benchmark(features_13, "13-FEATURE CLEVELAND CONFIGURATION (WITH 'thal')")
