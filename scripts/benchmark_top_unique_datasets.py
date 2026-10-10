"""
TMRM v4.5.0 vs Industry Giants on Top Unique Datasets:
1. Banknote Authentication (Wavelet Transform Continuous Features - 1,372 samples)
2. QSAR Molecular Biodegradation (41 Chemical Descriptors - 1,055 samples)
3. Statlog Heart Disease (Cardiovascular Clinical Biomarkers - 270 samples)
4. Vehicle Silhouettes (Geometric Shape Manifolds - 846 samples, 4 Classes)

Evaluated via 5-Fold Stratified Cross-Validation:
- TMRM v4.5.0 (Topological Manifold Resonant Machine)
- Random Forest (100 Trees)
- XGBoost (Extreme Gradient Boosting)
- LightGBM (Light Gradient Boosting)
- Logistic Regression (Linear Baseline)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from tmrm import TopologicalManifoldResonantMachine

CACHE_DIR = "data/unique_benchmarks"
os.makedirs(CACHE_DIR, exist_ok=True)

datasets = {}

# 1. Banknote Authentication (Wavelet Transform Manifold)
p_banknote = os.path.join(CACHE_DIR, "banknote_authentication.csv")
if not os.path.exists(p_banknote):
    print("Downloading Banknote Authentication dataset...")
    b_data = fetch_openml('banknote-authentication', version=1, as_frame=True)
    b_df = b_data.frame
    b_df.to_csv(p_banknote, index=False)
else:
    b_df = pd.read_csv(p_banknote)

y_b = b_df.iloc[:, -1].astype(str)
y_b = (y_b == y_b.unique()[1]).astype(int).values
X_b = b_df.iloc[:, :-1].values.astype(float)
datasets["Banknote Wavelet (4D)"] = (X_b, y_b)

# 2. QSAR Biodegradation (41 Molecular Topological Descriptors)
p_qsar = os.path.join(CACHE_DIR, "qsar_biodeg.csv")
if not os.path.exists(p_qsar):
    print("Downloading QSAR Biodegradation dataset...")
    q_data = fetch_openml(data_id=1494, as_frame=True)
    q_df = q_data.frame
    q_df.to_csv(p_qsar, index=False)
else:
    q_df = pd.read_csv(p_qsar)

y_q = q_df.iloc[:, -1].astype(str)
y_q = (y_q == y_q.unique()[0]).astype(int).values
X_q = q_df.iloc[:, :-1].values.astype(float)
datasets["QSAR Molecular (41D)"] = (X_q, y_q)

# 3. Statlog Heart Disease (Cardiovascular Clinical Manifold)
p_heart = os.path.join(CACHE_DIR, "heart_statlog.csv")
if not os.path.exists(p_heart):
    print("Downloading Statlog Heart Disease dataset...")
    h_data = fetch_openml(data_id=53, as_frame=True)
    h_df = h_data.frame
    h_df.to_csv(p_heart, index=False)
else:
    h_df = pd.read_csv(p_heart)

y_h = h_df.iloc[:, -1].astype(str)
y_h = (y_h == y_h.unique()[1]).astype(int).values
X_h = h_df.iloc[:, :-1].values.astype(float)
datasets["Statlog Heart (13D)"] = (X_h, y_h)

# 4. Vehicle Silhouettes (4-Class Geometric 2D/3D Projection Shapes)
p_veh = os.path.join(CACHE_DIR, "vehicle_silhouettes.csv")
if not os.path.exists(p_veh):
    print("Downloading Vehicle Silhouettes dataset...")
    v_data = fetch_openml('vehicle', version=1, as_frame=True)
    v_df = v_data.frame
    v_df.to_csv(p_veh, index=False)
else:
    v_df = pd.read_csv(p_veh)

y_v_raw = v_df.iloc[:, -1].astype(str)
labels = sorted(y_v_raw.unique())
label_map = {l: i for i, l in enumerate(labels)}
y_v = y_v_raw.map(label_map).values.astype(int)
X_v = v_df.iloc[:, :-1].values.astype(float)
datasets["Vehicle 4-Class (18D)"] = (X_v, y_v)

print("\n" + "=" * 125)
print("TOP UNIQUE DATASETS HEAD-TO-HEAD BATTLE: TMRM v4.5.0 vs INDUSTRY GIANTS")
print("=" * 125)
print(f"{'Dataset':<26} | {'Samples':<7} | {'TMRM v4.5':<11} | {'RandomForest':<13} | {'XGBoost':<11} | {'LightGBM':<11} | {'LogisticReg':<11} | {'Verdict'}")
print("-" * 125)

for name, (X, y) in datasets.items():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_tmrm, acc_rf, acc_xgb, acc_lgb, acc_lr = [], [], [], [], []
    t_tmrm_start = time.time()

    for tr, te in skf.split(X, y):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # 1. TMRM v4.5.0
        m_tmrm = TopologicalManifoldResonantMachine(wavelet_phase="auto", random_state=42)
        m_tmrm.fit(X_tr, y_tr)
        preds_tmrm = m_tmrm.predict(X_te)
        acc_tmrm.append(accuracy_score(y_te, preds_tmrm))

        # 2. Random Forest
        m_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        m_rf.fit(X_tr, y_tr)
        preds_rf = m_rf.predict(X_te)
        acc_rf.append(accuracy_score(y_te, preds_rf))

        # 3. XGBoost
        m_xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)
        m_xgb.fit(X_tr, y_tr)
        preds_xgb = m_xgb.predict(X_te)
        acc_xgb.append(accuracy_score(y_te, preds_xgb))

        # 4. LightGBM
        m_lgb = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        m_lgb.fit(X_tr, y_tr)
        preds_lgb = m_lgb.predict(X_te)
        acc_lgb.append(accuracy_score(y_te, preds_lgb))

        # 5. Logistic Regression
        m_lr = LogisticRegression(max_iter=1000, random_state=42)
        try:
            m_lr.fit(X_tr, y_tr)
            preds_lr = m_lr.predict(X_te)
            acc_lr.append(accuracy_score(y_te, preds_lr))
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

    print(f"{name:<26} | {len(X):<7} | {sc_tmrm:6.2f}%    | {sc_rf:6.2f}%       | {sc_xgb:6.2f}%    | {sc_lgb:6.2f}%    | {sc_lr:6.2f}%    | {verdict}")

print("=" * 125)
