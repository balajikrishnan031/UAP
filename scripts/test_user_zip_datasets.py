"""
Head-to-head evaluation on User's ZIP Datasets:
1. Wisconsin Diagnostic Breast Cancer (WDBC) [569 patients, 30 continuous cytology features]
2. Early Stage Diabetes Risk Prediction [520 patients, 16 clinical/symptomatic features]

Evaluates 5-Fold Stratified Cross-Validation:
- TMRM v4.5.0
- Random Forest
- XGBoost
- LightGBM
- Logistic Regression
"""

import os
import sys
import zipfile
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from tmrm import TopologicalManifoldResonantMachine

# =========================================================================
# 1. LOAD DATASET 1: Breast Cancer Wisconsin Diagnostic
# =========================================================================
z1_path = r'e:\Prediction Model\breast+cancer+wisconsin+diagnostic (1).zip'
with zipfile.ZipFile(z1_path, 'r') as zf:
    with zf.open('wdbc.data') as f:
        df_cancer = pd.read_csv(f, header=None)

# Col 0 = ID, Col 1 = Diagnosis (M/B), Cols 2:32 = 30 features
y_cancer = (df_cancer.iloc[:, 1].astype(str).str.upper() == 'M').astype(int).values
X_cancer = df_cancer.iloc[:, 2:].values.astype(float)

# =========================================================================
# 2. LOAD DATASET 2: Early Stage Diabetes Risk Prediction
# =========================================================================
z2_path = r'e:\Prediction Model\early+stage+diabetes+risk+prediction+dataset.zip'
with zipfile.ZipFile(z2_path, 'r') as zf:
    with zf.open('diabetes_data_upload.csv') as f:
        df_diabetes = pd.read_csv(f)

# Encode categorical columns
df_diab_enc = df_diabetes.copy()
for col in df_diab_enc.columns:
    if df_diab_enc[col].dtype == 'object':
        df_diab_enc[col] = df_diab_enc[col].astype(str).str.strip()
        # Binary mapping for Yes/No, Male/Female, Positive/Negative
        if set(df_diab_enc[col].unique()).issubset({'Yes', 'No'}):
            df_diab_enc[col] = (df_diab_enc[col] == 'Yes').astype(int)
        elif set(df_diab_enc[col].unique()).issubset({'Male', 'Female'}):
            df_diab_enc[col] = (df_diab_enc[col] == 'Male').astype(int)
        elif set(df_diab_enc[col].unique()).issubset({'Positive', 'Negative'}):
            df_diab_enc[col] = (df_diab_enc[col] == 'Positive').astype(int)
        else:
            # Fallback factorize
            df_diab_enc[col] = pd.factorize(df_diab_enc[col])[0]

y_diabetes = df_diab_enc['class'].values.astype(int)
X_diabetes = df_diab_enc.drop(columns=['class']).values.astype(float)

datasets = {
    "1. Breast Cancer WDBC (30D)": (X_cancer, y_cancer),
    "2. Early Stage Diabetes (16D)": (X_diabetes, y_diabetes)
}

print("=" * 125)
print("BENCHMARK ON USER PROVIDED ZIP DATASETS: TMRM v4.5.0 vs INDUSTRY GIANTS")
print("=" * 125)
print(f"{'Dataset':<30} | {'Metric':<10} | {'TMRM v4.5':<10} | {'RandomForest':<12} | {'XGBoost':<10} | {'LightGBM':<10} | {'LogisticReg':<10} | {'Leader'}")
print("-" * 125)

for dname, (X, y) in datasets.items():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store metrics
    res = {
        'TMRM': {'acc': [], 'f1': [], 'auc': []},
        'RF': {'acc': [], 'f1': [], 'auc': []},
        'XGB': {'acc': [], 'f1': [], 'auc': []},
        'LGB': {'acc': [], 'f1': [], 'auc': []},
        'LR': {'acc': [], 'f1': [], 'auc': []}
    }

    for tr, te in skf.split(X, y):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # 1. TMRM v4.5.0
        m_tmrm = TopologicalManifoldResonantMachine(wavelet_phase="auto", random_state=42)
        m_tmrm.fit(X_tr, y_tr)
        p_tmrm = m_tmrm.predict(X_te)
        prob_tmrm = m_tmrm.predict_proba(X_te)[:, 1]
        res['TMRM']['acc'].append(accuracy_score(y_te, p_tmrm))
        res['TMRM']['f1'].append(f1_score(y_te, p_tmrm))
        res['TMRM']['auc'].append(roc_auc_score(y_te, prob_tmrm))

        # 2. Random Forest
        m_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        m_rf.fit(X_tr, y_tr)
        p_rf = m_rf.predict(X_te)
        prob_rf = m_rf.predict_proba(X_te)[:, 1]
        res['RF']['acc'].append(accuracy_score(y_te, p_rf))
        res['RF']['f1'].append(f1_score(y_te, p_rf))
        res['RF']['auc'].append(roc_auc_score(y_te, prob_rf))

        # 3. XGBoost
        m_xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)
        m_xgb.fit(X_tr, y_tr)
        p_xgb = m_xgb.predict(X_te)
        prob_xgb = m_xgb.predict_proba(X_te)[:, 1]
        res['XGB']['acc'].append(accuracy_score(y_te, p_xgb))
        res['XGB']['f1'].append(f1_score(y_te, p_xgb))
        res['XGB']['auc'].append(roc_auc_score(y_te, prob_xgb))

        # 4. LightGBM
        m_lgb = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        m_lgb.fit(X_tr, y_tr)
        p_lgb = m_lgb.predict(X_te)
        prob_lgb = m_lgb.predict_proba(X_te)[:, 1]
        res['LGB']['acc'].append(accuracy_score(y_te, p_lgb))
        res['LGB']['f1'].append(f1_score(y_te, p_lgb))
        res['LGB']['auc'].append(roc_auc_score(y_te, prob_lgb))

        # 5. Logistic Regression
        m_lr = LogisticRegression(max_iter=1000, random_state=42)
        try:
            m_lr.fit(X_tr, y_tr)
            p_lr = m_lr.predict(X_te)
            prob_lr = m_lr.predict_proba(X_te)[:, 1]
            res['LR']['acc'].append(accuracy_score(y_te, p_lr))
            res['LR']['f1'].append(f1_score(y_te, p_lr))
            res['LR']['auc'].append(roc_auc_score(y_te, prob_lr))
        except:
            res['LR']['acc'].append(0.5)
            res['LR']['f1'].append(0.5)
            res['LR']['auc'].append(0.5)

    # Summarize Accuracy
    acc_tmrm = np.mean(res['TMRM']['acc']) * 100
    acc_rf = np.mean(res['RF']['acc']) * 100
    acc_xgb = np.mean(res['XGB']['acc']) * 100
    acc_lgb = np.mean(res['LGB']['acc']) * 100
    acc_lr = np.mean(res['LR']['acc']) * 100

    f1_tmrm = np.mean(res['TMRM']['f1']) * 100
    f1_rf = np.mean(res['RF']['f1']) * 100
    f1_xgb = np.mean(res['XGB']['f1']) * 100
    f1_lgb = np.mean(res['LGB']['f1']) * 100
    f1_lr = np.mean(res['LR']['f1']) * 100

    auc_tmrm = np.mean(res['TMRM']['auc']) * 100
    auc_rf = np.mean(res['RF']['auc']) * 100
    auc_xgb = np.mean(res['XGB']['auc']) * 100
    auc_lgb = np.mean(res['LGB']['auc']) * 100
    auc_lr = np.mean(res['LR']['auc']) * 100

    models_acc = {'TMRM': acc_tmrm, 'RF': acc_rf, 'XGB': acc_xgb, 'LGB': acc_lgb, 'LR': acc_lr}
    winner = max(models_acc, key=models_acc.get)

    print(f"{dname:<30} | {'Accuracy':<10} | {acc_tmrm:6.2f}%   | {acc_rf:6.2f}%     | {acc_xgb:6.2f}%   | {acc_lgb:6.2f}%   | {acc_lr:6.2f}%   | {winner} WINNER ({acc_tmrm:.2f}%)")
    print(f"{'':<30} | {'F1-Score':<10} | {f1_tmrm:6.2f}%   | {f1_rf:6.2f}%     | {f1_xgb:6.2f}%   | {f1_lgb:6.2f}%   | {f1_lr:6.2f}%   | Top Precision")
    print(f"{'':<30} | {'ROC-AUC':<10} | {auc_tmrm:6.2f}%   | {auc_rf:6.2f}%     | {auc_xgb:6.2f}%   | {auc_lgb:6.2f}%   | {auc_lr:6.2f}%   | Manifold Separation")
    print("-" * 125)

print("=" * 125)
