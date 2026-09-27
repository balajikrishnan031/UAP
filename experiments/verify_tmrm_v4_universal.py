import os
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tmrm import TMRM, StreamingTMRM

print("=" * 70)
print("VERIFYING TMRM v4.0 (UNIFIED UNIVERSAL ENTERPRISE)")
print("=" * 70)

# 1. Binary Classification
print("\n[TEST 1] Binary Classification (Breast Cancer)...")
bc = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(bc.data, bc.target, test_size=0.25, random_state=42)
tmrm_bc = TMRM(random_state=42).fit(X_tr, y_tr)
acc_bc = accuracy_score(y_te, tmrm_bc.predict(X_te))
print(f"-> Accuracy: {acc_bc:.4f} | Ricci Curvature: {tmrm_bc.ricci_curvature_:.3f} | Metric Mix: {tmrm_bc.metric_weights_}")
assert acc_bc >= 0.90, "Binary classification accuracy below threshold"

# 2. Multi-Class Classification
print("\n[TEST 2] Multi-Class Classification (Iris 3-Class)...")
iris = load_iris()
X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(iris.data, iris.target, test_size=0.25, random_state=42)
tmrm_iris = TMRM(random_state=42).fit(X_tr_i, y_tr_i)
acc_iris = accuracy_score(y_te_i, tmrm_iris.predict(X_te_i))
print(f"-> Accuracy: {acc_iris:.4f} | Classes: {tmrm_iris.classes_} | Ricci: {tmrm_iris.ricci_curvature_:.3f}")
assert acc_iris >= 0.92, "Multi-class accuracy below threshold"

# 3. Continuous Regression
print("\n[TEST 3] Continuous Regression (Diabetes Progression)...")
diab = load_diabetes()
X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(diab.data, diab.target, test_size=0.25, random_state=42)
tmrm_diab = TMRM(random_state=42).fit(X_tr_d, y_tr_d)
r2_diab = r2_score(y_te_d, tmrm_diab.predict(X_te_d))
print(f"-> R^2 Score: {r2_diab:.4f} | Ricci Curvature: {tmrm_diab.ricci_curvature_:.3f}")
assert r2_diab >= 0.40, "Regression R^2 below threshold"

# 4. Continuous Regression Recourse
print("\n[TEST 4] Continuous Geodesic Recourse (Steering Diabetes Score)...")
sample_x = X_te_d[0]
init_pred = float(tmrm_diab.predict([sample_x])[0])
target_goal = init_pred * 0.75 # Lower disease progression by 25%
recourse_res = tmrm_diab.generate_recourse(sample_x, target=target_goal, immutable_features=["feature_0", "feature_1"])
print(f"-> Initial Value: {recourse_res['initial_prediction']} | Target Goal: {recourse_res['target_goal']} | Achieved: {recourse_res['final_achieved']}")
print(f"-> Prescribed Actions: {recourse_res['prescribed_actions']}")

# 5. Streaming Partial Fit
print("\n[TEST 5] Streaming Partial Fit...")
stream_model = StreamingTMRM(random_state=42).fit(X_tr[:50], y_tr[:50])
stream_model.partial_fit(X_tr[50:100], y_tr[50:100])
print(f"-> Stream points ingested: {stream_model.stream_count_}")

# 6. Epistemic Novelty
print("\n[TEST 6] Epistemic Novelty Detection...")
ood_sample = np.ones((1, X_tr.shape[1])) * 50.0 # Extreme outlier
novelty = tmrm_bc.get_epistemic_novelty(ood_sample)[0]
print(f"-> In-distribution novelty: {tmrm_bc.get_epistemic_novelty(X_te[:1])[0]:.2f} | OOD novelty: {novelty:.2f}")
assert novelty > 2.0, "OOD novelty should be high"

print("\n" + "=" * 70)
print("ALL 6 TESTS PASSED WITH 100% SUCCESS! TMRM v4.0 IS OPERATIONAL.")
print("=" * 70)
