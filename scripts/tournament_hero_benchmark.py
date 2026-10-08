"""
TMRM v4.5.0 GRAND TOURNAMENT OF CHAMPIONS:
Compares:
1. TMRM v4.5.0 (Topological Manifold Resonant Machine)
2. Random Forest (Scikit-Learn)
3. XGBoost (Extreme Gradient Boosting)
4. LightGBM (Microsoft Light Gradient Boosting)
5. Logistic Regression / Ridge
Across 5-Fold Stratified Cross-Validation on premier real-world datasets.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from tmrm import TopologicalManifoldResonantMachine

DATA_DIR = "data/scholar_benchmarks"

datasets = {}

# 1. Sonar (60 dims, physical acoustics)
p_sonar = os.path.join(DATA_DIR, "sonar.csv")
if os.path.exists(p_sonar):
    df = pd.read_csv(p_sonar, header=None)
    y = (df.iloc[:, -1].astype(str).str.upper().str.startswith('M')).astype(int).values
    X = df.iloc[:, :-1].values.astype(float)
    datasets["Sonar (60D Acoustics)"] = (X, y)

# 2. Parkinson's (22 dims, neuro-acoustics)
p_park = os.path.join(DATA_DIR, "parkinsons.csv")
if os.path.exists(p_park):
    df = pd.read_csv(p_park)
    y = df['status'].astype(int).values
    X = df.drop(columns=['name', 'status']).values.astype(float)
    datasets["Parkinson's (22D Bio)"] = (X, y)

# 3. Glass (9 dims, 6 classes, forensic material)
p_glass = os.path.join(DATA_DIR, "glass.csv")
if os.path.exists(p_glass):
    df = pd.read_csv(p_glass)
    t = 'target' if 'target' in df.columns else df.columns[-1]
    y_raw = df[t].astype(str)
    g_map = {c: i for i, c in enumerate(np.unique(y_raw))}
    y = y_raw.map(g_map).astype(int).values
    X = df.drop(columns=[t]).values.astype(float)
    datasets["Glass (6-Class Physics)"] = (X, y)

# 4. Ionosphere (34 dims, aerospace radar pulses)
p_iono = os.path.join(DATA_DIR, "ionosphere.csv")
if os.path.exists(p_iono):
    df = pd.read_csv(p_iono, header=None)
    y = (df.iloc[:, -1].astype(str).str.lower().str.startswith('g')).astype(int).values
    X = df.iloc[:, :-1].values.astype(float)
    datasets["Ionosphere (34D Radar)"] = (X, y)

# 5. Breast Cancer (30 dims, clinical cytology)
p_bc = os.path.join(DATA_DIR, "breast_cancer.csv")
if os.path.exists(p_bc):
    df = pd.read_csv(p_bc)
    y = df['target'].astype(int).values
    X = df.drop(columns=['target']).values.astype(float)
    datasets["Breast Cancer (30D)"] = (X, y)


print("=" * 115)
print("TMRM v4.5.0 GRAND TOURNAMENT OF CHAMPIONS (5-FOLD STRATIFIED CROSS-VALIDATION)")
print("=" * 115)
print(f"{'Dataset':<26} | {'TMRM v4.5':<11} | {'RandomForest':<13} | {'XGBoost':<11} | {'LightGBM':<11} | {'LogisticReg':<11} | {'Hero Verdict'}")
print("-" * 115)

tournament_records = []

for name, (X, y) in datasets.items():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_tmrm, acc_rf, acc_xgb, acc_lgb, acc_lr = [], [], [], [], []

    for tr, te in skf.split(X, y):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # 1. TMRM v4.5.0
        m_tmrm = TopologicalManifoldResonantMachine(wavelet_phase="auto", random_state=42)
        m_tmrm.fit(X_tr, y_tr)
        acc_tmrm.append(accuracy_score(y_te, m_tmrm.predict(X_te)))

        # 2. Random Forest
        m_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        m_rf.fit(X_tr, y_tr)
        acc_rf.append(accuracy_score(y_te, m_rf.predict(X_te)))

        # 3. XGBoost
        m_xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)
        m_xgb.fit(X_tr, y_tr)
        acc_xgb.append(accuracy_score(y_te, m_xgb.predict(X_te)))

        # 4. LightGBM
        m_lgb = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        m_lgb.fit(X_tr, y_tr)
        acc_lgb.append(accuracy_score(y_te, m_lgb.predict(X_te)))

        # 5. Logistic Regression
        m_lr = LogisticRegression(max_iter=1000, random_state=42)
        try:
            m_lr.fit(X_tr, y_tr)
            acc_lr.append(accuracy_score(y_te, m_lr.predict(X_te)))
        except:
            acc_lr.append(0.5)

    sc_tmrm = np.mean(acc_tmrm) * 100
    sc_rf = np.mean(acc_rf) * 100
    sc_xgb = np.mean(acc_xgb) * 100
    sc_lgb = np.mean(acc_lgb) * 100
    sc_lr = np.mean(acc_lr) * 100

    max_other = max(sc_rf, sc_xgb, sc_lgb, sc_lr)
    delta = sc_tmrm - max_other
    if sc_tmrm >= max_other:
        verdict = f"[TMRM WINS +{delta:.2f}%]"
    else:
        verdict = f"[Diff {delta:.2f}%]"

    print(f"{name:<26} | {sc_tmrm:6.2f}%    | {sc_rf:6.2f}%       | {sc_xgb:6.2f}%    | {sc_lgb:6.2f}%    | {sc_lr:6.2f}%    | {verdict}")

    tournament_records.append({
        "dataset": name,
        "tmrm": sc_tmrm,
        "rf": sc_rf,
        "xgb": sc_xgb,
        "lgb": sc_lgb,
        "lr": sc_lr,
        "verdict": verdict
    })

print("=" * 115)
