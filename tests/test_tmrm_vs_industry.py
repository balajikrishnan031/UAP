"""
Head-to-head comparison test between:
1. Novel Invention: Topological Manifold Resonant Machine (TMRM)
2. Industry Standard 1: XGBoost (or Gradient Boosting)
3. Industry Standard 2: Random Forest
4. Industry Standard 3: Logistic Regression
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_moons

from uap.models.novel_tmrm import TopologicalManifoldResonantMachine


def test_synthetic_complex_manifold():
    print("=" * 80)
    print("TEST 1: Highly Non-Linear Curved Manifold (Two Interlocking Moons)")
    print("Trees struggle with diagonal/curved non-orthogonal splits!")
    print("=" * 80)

    X, y = make_moons(n_samples=1000, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 1. TMRM (Our Novel Algorithm)
    tmrm = TopologicalManifoldResonantMachine(n_resonators_per_class=6, random_state=42)
    tmrm.fit(X_train, y_train)
    tmrm_preds = tmrm.predict(X_test)
    tmrm_probs = tmrm.predict_proba(X_test)[:, 1]
    tmrm_acc = accuracy_score(y_test, tmrm_preds)
    tmrm_f1 = f1_score(y_test, tmrm_preds)
    tmrm_auc = roc_auc_score(y_test, tmrm_probs)

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    rf_auc = roc_auc_score(y_test, rf_probs)

    # 3. Gradient Boosting (Tree Boosting)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    gb_probs = gb.predict_proba(X_test)[:, 1]
    gb_acc = accuracy_score(y_test, gb_preds)
    gb_f1 = f1_score(y_test, gb_preds)
    gb_auc = roc_auc_score(y_test, gb_probs)

    # 4. Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)
    lr_auc = roc_auc_score(y_test, lr_probs)

    print(f"{'Model':<35} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 75)
    print(f"{'TMRM (Our Novel Algorithm)':<35} | {tmrm_acc:.4f}     | {tmrm_f1:.4f}     | {tmrm_auc:.4f}")
    print(f"{'Random Forest (Industry Standard)':<35} | {rf_acc:.4f}     | {rf_f1:.4f}     | {rf_auc:.4f}")
    print(f"{'Gradient Boosting (Industry Standard)':<35} | {gb_acc:.4f}     | {gb_f1:.4f}     | {gb_auc:.4f}")
    print(f"{'Logistic Regression (Baseline)':<35} | {lr_acc:.4f}     | {lr_f1:.4f}     | {lr_auc:.4f}")

    print("\n[TMRM IN-MODEL COGNITIVE CAPABILITIES - IMPOSSIBLE IN OTHER MODELS]:")
    # Inherent Novelty Test (Self-Doubt)
    ood_sample = np.array([[10.0, 10.0]])  # Far away from data distribution
    novelty = tmrm.get_epistemic_novelty(ood_sample)[0]
    print(f"1. Epistemic Novelty for normal sample: {tmrm.get_epistemic_novelty(X_test[:1])[0]:.2f}")
    print(f"2. Epistemic Novelty for Out-of-Distribution sample: {novelty:.2f} (Self-Doubt Flagged: {novelty > 2.5})")

    # Analytical Closed-form Recourse Test
    sample = X_test[0]
    recourse = tmrm.get_recourse_gradient(sample, target_class=1 if y_test[0] == 0 else 0)
    print(f"3. In-Model Recourse Distance to Target: {recourse['manifold_distance_to_target']:.4f}")
    print(f"4. Top Actionable Feature to change outcome: Feature Index {recourse['top_actionable_feature_idx']}")


def test_real_dataset():
    print("\n" + "=" * 80)
    print("TEST 2: Real-World Clinical / Healthcare Dataset (Heart Disease Risk)")
    print("=" * 80)
    df = pd.read_csv("data/real_world/heart_disease.csv")
    target_col = [c for c in df.columns if "target" in c.lower() or "heart" in c.lower() or "disease" in c.lower()][0]
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # TMRM
    tmrm = TopologicalManifoldResonantMachine(n_resonators_per_class=8, random_state=42)
    tmrm.fit(X_train, y_train)
    tmrm_preds = tmrm.predict(X_test)
    tmrm_probs = tmrm.predict_proba(X_test)[:, 1]
    tmrm_acc = accuracy_score(y_test, tmrm_preds)
    tmrm_f1 = f1_score(y_test, tmrm_preds)
    tmrm_auc = roc_auc_score(y_test, tmrm_probs)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    rf_auc = roc_auc_score(y_test, rf_probs)

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    gb_probs = gb.predict_proba(X_test)[:, 1]
    gb_acc = accuracy_score(y_test, gb_preds)
    gb_f1 = f1_score(y_test, gb_preds)
    gb_auc = roc_auc_score(y_test, gb_probs)

    print(f"{'Model':<35} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 75)
    print(f"{'TMRM (Our Novel Algorithm)':<35} | {tmrm_acc:.4f}     | {tmrm_f1:.4f}     | {tmrm_auc:.4f}")
    print(f"{'Random Forest (Industry Standard)':<35} | {rf_acc:.4f}     | {rf_f1:.4f}     | {rf_auc:.4f}")
    print(f"{'Gradient Boosting (Industry Standard)':<35} | {gb_acc:.4f}     | {gb_f1:.4f}     | {gb_auc:.4f}")


if __name__ == "__main__":
    test_synthetic_complex_manifold()
    test_real_dataset()
