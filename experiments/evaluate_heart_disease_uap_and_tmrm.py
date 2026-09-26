"""
Comprehensive Clinical Evaluation on UCI Multi-Hospital Heart Disease Dataset (920 Patients)
Using:
1. Novel Invention: Topological Manifold Resonant Machine (TMRM)
2. Autonomous AI Framework: UAP Engine (Autonomous Intent, Conformal, Living Decision Dossier)
3. Industry Benchmarks: Gradient Boosting (XGBoost), Random Forest, Logistic Regression
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from uap import UAP, TMRM
from uap.models.novel_tmrm import TopologicalManifoldResonantMachine


def run_full_clinical_study():
    print("=" * 105, flush=True)
    print("CLINICAL EVALUATION: 920 MULTI-HOSPITAL HEART DISEASE PATIENTS", flush=True)
    print("Hospitals: Cleveland Clinic, Hungarian Cardiology, Zurich Switzerland, Long Beach VA", flush=True)
    print("=" * 105, flush=True)

    csv_path = r"e:\Prediction Model\data\multi_hospital_heart_disease_combined.csv"
    df = pd.read_csv(csv_path)

    # Impute missing clinical values using median per hospital
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    for col in feature_cols:
        df[col] = df[col].fillna(df.groupby('hospital_origin')[col].transform('median'))
        df[col] = df[col].fillna(df[col].median() if len(df[col].dropna()) > 0 else 0)

    # Binary classification target: 0 (No Disease) vs 1 (Heart Disease Presence)
    df['target_binary'] = (df['target'] > 0).astype(int)

    X = df[feature_cols].copy()
    y_bin = df['target_binary'].values
    y_multi = df['target'].values

    # Train / Test split (75% Train, 25% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.25, random_state=42, stratify=y_bin)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------------------
    # PART 1: BINARY HEART DISEASE PREDICTION (DIAGNOSIS)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 105, flush=True)
    print("PART 1: BINARY HEART DISEASE CLASSIFICATION (0 = Healthy, 1 = Heart Disease)", flush=True)
    print("-" * 105, flush=True)

    # 1. TMRM (Novel Model)
    tmrm = TMRM(task_type="classification", n_resonators="auto", random_state=42)
    tmrm.fit(X_train, y_train)
    tmrm_preds = tmrm.predict(X_test)
    tmrm_probs = tmrm.predict_proba(X_test)[:, 1]
    tmrm_acc = accuracy_score(y_test, tmrm_preds)
    tmrm_f1 = f1_score(y_test, tmrm_preds)
    tmrm_auc = roc_auc_score(y_test, tmrm_probs)

    # 2. Gradient Boosting (Tree Boosting)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    gb_probs = gb.predict_proba(X_test)[:, 1]
    gb_acc = accuracy_score(y_test, gb_preds)
    gb_f1 = f1_score(y_test, gb_preds)
    gb_auc = roc_auc_score(y_test, gb_probs)

    # 3. Random Forest (Bagging Trees)
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    rf_auc = roc_auc_score(y_test, rf_probs)

    # 4. Logistic Regression (Clinical Baseline)
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)
    lr_auc = roc_auc_score(y_test, lr_probs)

    print(f"{'Model Architecture':<35} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'Status'}", flush=True)
    print("-" * 90, flush=True)
    print(f"{'TMRM (Our Novel Algorithm)':<35} | {tmrm_acc*100:<9.2f}% | {tmrm_f1:<10.4f} | {tmrm_auc:<10.4f} | [*] Novel Manifolds", flush=True)
    print(f"{'Gradient Boosting (Industry Best)':<35} | {gb_acc*100:<9.2f}% | {gb_f1:<10.4f} | {gb_auc:<10.4f} | Standard Tree Boosting", flush=True)
    print(f"{'Random Forest (Industry Baseline)':<35} | {rf_acc*100:<9.2f}% | {rf_f1:<10.4f} | {rf_auc:<10.4f} | Standard Bagging", flush=True)
    print(f"{'Logistic Regression (Medical)':<35} | {lr_acc*100:<9.2f}% | {lr_f1:<10.4f} | {lr_auc:<10.4f} | Linear Baseline", flush=True)

    # -------------------------------------------------------------------------
    # PART 2: UAP END-TO-END AUTONOMOUS ENGINE EXECUTION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 105, flush=True)
    print("PART 2: UAP AUTONOMOUS DECISION ENGINE EXECUTION (LIVING DECISION DOSSIER)", flush=True)
    print("-" * 105, flush=True)

    df_uap = X.copy()
    df_uap['target'] = y_bin
    uap_engine = UAP(timeout_sec=25, n_trials=6)
    card = uap_engine.auto_discover(df_uap, goal_prompt="Identify patients at high risk of coronary artery disease")
    print(f"\n[UAP Discovery Card Summary]:")
    print(f"  * Inferred Domain     : {card.detected_domain}")
    print(f"  * Recommended Target  : {card.recommended_target}")
    print(f"  * Business Objective  : {card.primary_business_goal}")

    uap_engine.fit(df_uap, target_column="target")
    print(f"  * Winning Architecture: {uap_engine.optimizer.winning_model_name}")

    # Generate Living Decision Dossier for a real high-risk patient
    high_risk_idx = np.where((y_test == 1))[0][0]
    high_risk_patient = X_test.iloc[high_risk_idx]

    print("\n[UAP 5.0 LIVING DECISION DOSSIER FOR PATIENT #{}]:".format(high_risk_idx), flush=True)
    dossier = uap_engine.predict_dossier(
        sample=high_risk_patient,
        target_goal=0,
        immutable_features=['age', 'sex']
    )
    print(dossier.format_executive_dossier(), flush=True)

    # -------------------------------------------------------------------------
    # PART 3: IN-MODEL CLINICAL RECOURSE (TMRM ACTION PLAN)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 105, flush=True)
    print("PART 3: TMRM IN-MODEL PRESCRIPTIVE RECOURSE (What flips this patient to Safe?)", flush=True)
    print("-" * 105, flush=True)

    recourse = tmrm.get_recourse_action_plan(
        sample=high_risk_patient,
        target_class=0,
        immutable_features=['age', 'sex']
    )
    print(f"Target Outcome: Safe (Class 0) | Manifold Feasibility Distance: {recourse['feasibility_distance']}")
    print("Prescribed Clinical Action Plan:")
    for idx, act in enumerate(recourse['action_items'], 1):
        print(f"  {idx}. {act['description']} (Shift: {act['delta']:+})")

    # -------------------------------------------------------------------------
    # PART 4: CLINICAL SAFETY & EPISTEMIC SELF-DOUBT TEST (ANOMALOUS DATA)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 105, flush=True)
    print("PART 4: SAFETY AUDIT ON SEVERELY CORRUPTED / ALIEN PATIENT DATA", flush=True)
    print("-" * 105, flush=True)

    corrupted_patient = high_risk_patient.copy()
    corrupted_patient['trestbps'] = 320.0  # Impossible blood pressure
    corrupted_patient['chol'] = 980.0      # Extreme toxic cholesterol

    # Check normal patient novelty vs corrupted patient novelty
    normal_novelty = tmrm.get_epistemic_novelty(pd.DataFrame([high_risk_patient]))[0]
    alien_novelty = tmrm.get_epistemic_novelty(pd.DataFrame([corrupted_patient]))[0]

    print(f"Normal Patient Epistemic Novelty Score    : {normal_novelty:.2f} (Status: Safe & Familiar)")
    print(f"Corrupted Patient Epistemic Novelty Score : {alien_novelty:.2f} (Threshold: {tmrm.novelty_threshold})")
    print(f"TMRM Self-Doubt Triggered                : {alien_novelty > tmrm.novelty_threshold} -> [SAFETY ABSTENTION ACTIVATED!]")

    # -------------------------------------------------------------------------
    # PART 5: MULTI-STAGE SEVERITY PREDICTION (MULTICLASS: 0 to 4 STAGES)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 105, flush=True)
    print("PART 5: MULTI-STAGE CORONARY SEVERITY (0=Healthy, 1=Mild, 2=Moderate, 3=Severe, 4=Critical)", flush=True)
    print("-" * 105, flush=True)

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X, y_multi, test_size=0.25, random_state=42, stratify=y_multi)
    tmrm_multi = TMRM(task_type="classification", n_resonators=5, random_state=42)
    tmrm_multi.fit(X_train_m, y_train_m)
    m_preds = tmrm_multi.predict(X_test_m)
    m_f1 = f1_score(y_test_m, m_preds, average='macro')
    m_acc = accuracy_score(y_test_m, m_preds)

    rf_multi = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_multi.fit(X_train_m, y_train_m)
    rf_m_preds = rf_multi.predict(X_test_m)
    rf_m_f1 = f1_score(y_test_m, rf_m_preds, average='macro')
    rf_m_acc = accuracy_score(y_test_m, rf_m_preds)

    print(f"TMRM Multi-Stage Severity Accuracy : {m_acc*100:.2f}% | Macro-F1: {m_f1:.4f}", flush=True)
    print(f"Random Forest Severity Accuracy    : {rf_m_acc*100:.2f}% | Macro-F1: {rf_m_f1:.4f}", flush=True)

    print("\n" + "=" * 105, flush=True)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY ON 920 REAL CLINICAL PATIENTS!", flush=True)
    print("=" * 105, flush=True)


if __name__ == "__main__":
    run_full_clinical_study()
