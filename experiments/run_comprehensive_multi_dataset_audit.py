import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_wine, load_digits
from sklearn.metrics import accuracy_score, r2_score
from tmrm import TMRM

print("=" * 90)
print("COMPREHENSIVE MULTI-DOMAIN DIVERSE DATASET AUDIT (TMRM vs RANDOM FOREST vs GRADIENT BOOSTING)")
print("=" * 90)

audit_results = []

def evaluate_dataset(domain_name, dataset_name, X, y, is_classification=True):
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    
    # Train-Test Split (75% train, 25% test)
    X_tr, X_te, y_tr, y_te = train_test_split(X_arr, y_arr, test_size=0.25, random_state=42)
    
    n_samples, n_feats = X_arr.shape
    unique_classes = len(np.unique(y_arr)) if is_classification else "Continuous"
    
    if is_classification:
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        tmrm = TMRM(task_type="classification", random_state=42).fit(X_tr, y_tr)
        
        rf_score = accuracy_score(y_te, rf.predict(X_te))
        gb_score = accuracy_score(y_te, gb.predict(X_te))
        tmrm_score = accuracy_score(y_te, tmrm.predict(X_te))
        metric_name = "Accuracy"
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        tmrm = TMRM(task_type="regression", random_state=42).fit(X_tr, y_tr)
        
        rf_score = r2_score(y_te, rf.predict(X_te))
        gb_score = r2_score(y_te, gb.predict(X_te))
        tmrm_score = r2_score(y_te, tmrm.predict(X_te))
        metric_name = "R^2"
        
    winner = "TMRM" if tmrm_score >= max(rf_score, gb_score) else ("RF" if rf_score >= gb_score else "GB")
    margin = tmrm_score - max(rf_score, gb_score)
    
    record = {
        "Domain": domain_name,
        "Dataset": dataset_name,
        "N": n_samples,
        "D": n_feats,
        "Classes/Target": unique_classes,
        "TMRM": round(tmrm_score, 4),
        "RandomForest": round(rf_score, 4),
        "GradientBoosting": round(gb_score, 4),
        "Winner": winner,
        "Delta_vs_Best": f"{margin:+.4f}"
    }
    audit_results.append(record)
    print(f"[{domain_name.upper()}] {dataset_name}: TMRM={tmrm_score:.4f} | RF={rf_score:.4f} | GB={gb_score:.4f} => Winner: {winner}")

# -------------------------------------------------------------
# 1. ACOUSTICS / DEFENSE PHYSICS: Sonar Mines vs Rocks
# -------------------------------------------------------------
url_sonar = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'
try:
    df_sonar = pd.read_csv(url_sonar, header=None)
    X_sonar = df_sonar.iloc[:, :-1].values
    y_sonar = (df_sonar.iloc[:, -1] == 'M').astype(int).values
    evaluate_dataset("Acoustics & Defense", "Sonar (Mines vs Rocks)", X_sonar, y_sonar, is_classification=True)
except Exception as e:
    print("Sonar load error:", e)

# -------------------------------------------------------------
# 2. CHEMISTRY & SPECTROMETRY: Wine Biomarkers (Multi-Class 3)
# -------------------------------------------------------------
wine = load_wine()
evaluate_dataset("Chemistry & Wine", "Wine Chemical Cultivars", wine.data, wine.target, is_classification=True)

# -------------------------------------------------------------
# 3. SPATIAL IMAGING / COMPUTER VISION: Optical Digits (10-Class)
# -------------------------------------------------------------
digits = load_digits()
# Use subset of 600 digits for fast execution
evaluate_dataset("Spatial Vision", "Optical Handwritten Digits (0-9)", digits.data[:800], digits.target[:800], is_classification=True)

# -------------------------------------------------------------
# 4. SENSORY & MATERIAL PHYSICS: Red Wine Quality (Regression)
# -------------------------------------------------------------
url_wine_qual = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
try:
    df_wq = pd.read_csv(url_wine_qual, sep=';')
    X_wq = df_wq.iloc[:, :-1].values
    y_wq = df_wq.iloc[:, -1].values.astype(float)
    evaluate_dataset("Food Science / Sensory", "Red Wine Quality Scoring", X_wq, y_wq, is_classification=False)
except Exception as e:
    print("Wine quality error:", e)

# -------------------------------------------------------------
# 5. CARDIOLOGY / CLINICAL SURVIVAL: Heart Failure Survival
# -------------------------------------------------------------
path_clin = "data/real_world/clinical_survival.csv"
if os.path.exists(path_clin):
    df_clin = pd.read_csv(path_clin)
    X_clin = df_clin.iloc[:, :-1].values
    y_clin = df_clin.iloc[:, -1].values
    evaluate_dataset("Clinical Survival", "Heart Failure Clinical Triage", X_clin, y_clin, is_classification=True)

# -------------------------------------------------------------
# 6. BANKING & CREDIT RISK: German Credit
# -------------------------------------------------------------
path_cred = "data/real_world/credit_risk.csv"
if os.path.exists(path_cred):
    df_cred = pd.read_csv(path_cred)
    X_cred = df_cred.iloc[:, :-1].values
    y_cred = df_cred.iloc[:, -1].values
    evaluate_dataset("Financial Banking", "German Credit Default Risk", X_cred, y_cred, is_classification=True)

# -------------------------------------------------------------
# 7. INDUSTRIAL IOT & TURBOMACHINERY: IoT Turbofan Maintenance
# -------------------------------------------------------------
path_iot = "data/real_world/iot_maintenance.csv"
if os.path.exists(path_iot):
    df_iot = pd.read_csv(path_iot).dropna()
    X_iot = df_iot.iloc[:1500, :-1].values
    y_iot = df_iot.iloc[:1500, -1].values
    evaluate_dataset("Industrial IoT", "Turbofan Predictive Maintenance", X_iot, y_iot, is_classification=True)

# -------------------------------------------------------------
# 8. AGRONOMY & SOIL SCIENCE: Crop Yield
# -------------------------------------------------------------
path_agri = "data/real_world/agriculture_crop.csv"
if os.path.exists(path_agri):
    df_agri = pd.read_csv(path_agri)
    X_agri = df_agri.iloc[:, :-1].values
    y_agri = df_agri.iloc[:, -1].values
    evaluate_dataset("Agronomy & Agriculture", "Crop Soil Suitability", X_agri, y_agri, is_classification=True)

# -------------------------------------------------------------
# Summary Table
# -------------------------------------------------------------
df_summary = pd.DataFrame(audit_results)
print("\n" + "=" * 90)
print("FINAL MULTI-DOMAIN AUDIT SUMMARY")
print("=" * 90)
print(df_summary.to_string(index=False))
print("=" * 90)

# Save to CSV
df_summary.to_csv("models/diverse_multi_domain_benchmark_report.csv", index=False)
print("Saved report to: models/diverse_multi_domain_benchmark_report.csv")
