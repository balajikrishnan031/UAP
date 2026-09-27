import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.datasets import load_diabetes, load_wine, make_friedman1
from sklearn.metrics import accuracy_score, r2_score
from tmrm import TMRM

print("=" * 100)
print("TMRM v4.0 MASTER 11-FIELD CHAMPIONSHIP BENCHMARK AUDIT")
print("Head-to-Head Comparison: TMRM vs Random Forest vs Gradient Boosting")
print("=" * 100)

records = []

def run_field(field_no, domain_name, dataset_name, X, y, is_clf=True):
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    
    # Stratified split for classification, standard split for regression
    strat = y_arr if (is_clf and len(np.unique(y_arr)) <= 10) else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_arr, y_arr, test_size=0.25, random_state=42, stratify=strat
    )
    
    N, D = X_arr.shape
    
    # 1. Random Forest
    t0 = time.time()
    if is_clf:
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        rf_score = accuracy_score(y_te, rf.predict(X_te))
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        rf_score = r2_score(y_te, rf.predict(X_te))
    rf_time = time.time() - t0

    # 2. Gradient Boosting
    t0 = time.time()
    if is_clf:
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        gb_score = accuracy_score(y_te, gb.predict(X_te))
    else:
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        gb_score = r2_score(y_te, gb.predict(X_te))
    gb_time = time.time() - t0

    # 3. TMRM v4.0
    t0 = time.time()
    task = "classification" if is_clf else "regression"
    tm = TMRM(task_type=task, random_state=42).fit(X_tr, y_tr)
    if is_clf:
        tm_score = accuracy_score(y_te, tm.predict(X_te))
    else:
        tm_score = r2_score(y_te, tm.predict(X_te))
    tm_time = time.time() - t0

    metric = "Acc" if is_clf else "R^2"
    best_competitor = max(rf_score, gb_score)
    winner = "TMRM" if tm_score >= best_competitor else ("RF" if rf_score >= gb_score else "GB")
    diff = tm_score - best_competitor
    
    row = {
        "No": field_no,
        "Domain": domain_name,
        "Dataset": dataset_name,
        "N": N,
        "D": D,
        "Metric": metric,
        "TMRM": round(tm_score, 4),
        "RandomForest": round(rf_score, 4),
        "GradientBoosting": round(gb_score, 4),
        "Winner": winner,
        "Margin_vs_Best": f"{diff:+.4f}",
        "TMRM_Curvature": round(tm.ricci_curvature_, 3)
    }
    records.append(row)
    print(f"[{field_no:02d}/11] {domain_name:<24} | TMRM: {tm_score:.4f} | RF: {rf_score:.4f} | GB: {gb_score:.4f} => Winner: {winner} ({diff:+.4f})")

# 1. Cardiology Diagnostics
df_heart = pd.read_csv('data/real_world/heart_disease.csv')
run_field(1, "Cardiology Diagnostics", "Cleveland Heart Disease", df_heart.iloc[:, :-1].values, df_heart.iloc[:, -1].values, is_clf=True)

# 2. Clinical ICU Survival
df_clin = pd.read_csv('data/real_world/clinical_survival.csv')
run_field(2, "Clinical ICU Survival", "Heart Failure Patient Triage", df_clin.iloc[:, :-1].values, df_clin.iloc[:, -1].values, is_clf=True)

# 3. Metabolic Disease
d = load_diabetes()
run_field(3, "Metabolic Disease", "Clinical Diabetes Progression", d.data, d.target, is_clf=False)

# 4. Industrial IoT & Turbines
df_iot = pd.read_csv('data/real_world/iot_maintenance.csv').dropna()
run_field(4, "Industrial Turbomachinery", "Turbofan Telemetry (IoT)", df_iot.iloc[:2000, :-1].values, df_iot.iloc[:2000, -1].values, is_clf=True)

# 5. Chemical Spectrometry
w = load_wine()
run_field(5, "Chemical Spectrometry", "Wine Cultivar Biomarkers", w.data, w.target, is_clf=True)

# 6. Environmental Atmospheric Hydrology
df_air = pd.read_csv('data/real_world/air_quality.csv').dropna()
run_field(6, "Atmospheric Hydrology", "Air Quality PM2.5 Dynamics", df_air.iloc[:2000, :-1].values, df_air.iloc[:2000, -1].values, is_clf=False)

# 7. Telecom SaaS Business
df_churn = pd.read_csv('data/real_world/telecom_churn.csv').dropna()
run_field(7, "Telecom SaaS Business", "Customer Lifetime Churn", df_churn.iloc[:2000, :-1].values, df_churn.iloc[:2000, -1].values, is_clf=True)

# 8. Oncology Medicine
df_bc = pd.read_csv('data/real_world/breast_cancer.csv').dropna()
run_field(8, "Oncology Medicine", "Breast Cancer Cytology", df_bc.iloc[:, :-1].values, df_bc.iloc[:, -1].values, is_clf=True)

# 9. Financial Banking Credit
df_cred = pd.read_csv('data/real_world/credit_risk.csv').dropna()
run_field(9, "Financial Banking Credit", "German Credit Default Risk", df_cred.iloc[:, :-1].values, df_cred.iloc[:, -1].values, is_clf=True)

# 10. Housing Econometrics
df_cal = pd.read_csv('data/real_world/california_housing.csv').dropna()
run_field(10, "Housing Econometrics", "California Price Surface", df_cal.iloc[:2500, :-1].values, df_cal.iloc[:2500, -1].values, is_clf=False)

# 11. Complex Non-Linear Physics
X_f, y_f = make_friedman1(n_samples=1200, n_features=10, noise=1.0, random_state=42)
run_field(11, "Non-Linear Physics", "Friedman-1 Energy Field", X_f, y_f, is_clf=False)

# Summary
df_report = pd.DataFrame(records)
print("\n" + "=" * 100)
print("FINAL 11-FIELD CHAMPIONSHIP BENCHMARK SUMMARY TABLE")
print("=" * 100)
print(df_report.to_string(index=False))
print("=" * 100)

os.makedirs("models", exist_ok=True)
out_csv = "models/master_11_field_championship_report.csv"
df_report.to_csv(out_csv, index=False)
print(f"Master benchmark report saved to: {out_csv}")
