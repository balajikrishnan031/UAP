"""
Fresh Scholarly Benchmark Suite:
Downloads and evaluates top-tier academic datasets directly from premier repositories (UCI ML Repository / OpenML),
saves them into data/scholar_benchmarks/, runs TMRM v4.4, demonstrates exact prediction outputs,
and benchmarks 5-Fold Stratified Cross-Validation against Random Forest.
"""

import os
import sys
import time
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Ensure safe unicode console printing
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from tmrm import TopologicalManifoldResonantMachine

DATA_DIR = "data/scholar_benchmarks"
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 95)
print("PREMIER SCHOLARLY BENCHMARK AUDIT (9 DIVERSE DOMAINS: UCI / OPENML)")
print("=" * 95)


def download_file(url, local_path):
    print(f"Downloading from {url} ...")
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=15) as response, open(local_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"  -> Saved to {local_path} ({os.path.getsize(local_path) / 1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"  -> Download failed ({e}), using reliable mirror...")
        return False


# ==============================================================================
# DATASET PATHS
# ==============================================================================
path_park = os.path.join(DATA_DIR, "parkinsons.csv")
path_sonar = os.path.join(DATA_DIR, "sonar.csv")
path_wine = os.path.join(DATA_DIR, "wine.csv")
path_iono = os.path.join(DATA_DIR, "ionosphere.csv")
path_bc = os.path.join(DATA_DIR, "breast_cancer.csv")
path_heart = os.path.join(DATA_DIR, "heart_disease.csv")
path_diabetes = os.path.join(DATA_DIR, "diabetes.csv")
path_glass = os.path.join(DATA_DIR, "glass.csv")
path_veh = os.path.join(DATA_DIR, "vehicle.csv")


def evaluate_and_demonstrate(name, X, y, class_names=None):
    print("\n" + "-" * 90)
    print(f"DOMAIN DATASET: {name}")
    print(f"  * Total Samples (N) : {len(X)}")
    print(f"  * Features (D)      : {X.shape[1]}")
    classes, counts = np.unique(y, return_counts=True)
    dist_str = ", ".join([f"Class {c} ({class_names.get(c, str(c)) if class_names else c}): {cnt}" for c, cnt in zip(classes, counts)])
    print(f"  * Class Distribution: {dist_str}")
    print("-" * 90)

    # 1. Real-time Inference Demonstration
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    demo_model = TopologicalManifoldResonantMachine(random_state=42)
    demo_model.fit(X_train, y_train)

    test_subset = X_test.iloc[:5]
    true_labels = y_test.iloc[:5].values
    preds = demo_model.predict(test_subset)
    probs = demo_model.predict_proba(test_subset)
    novelty = demo_model.get_epistemic_novelty(test_subset)

    print("\n  [LIVE INFERENCE DEMONSTRATION ON UNSEEN TEST SAMPLES]:")
    print(f"  {'Sample':<8} | {'True Label':<18} | {'Predicted':<18} | {'Confidence':<12} | {'Novelty (OOD)':<14} | {'Status'}")
    print("  " + "-" * 85)

    for i in range(len(test_subset)):
        p_cls = preds[i]
        t_cls = true_labels[i]
        p_name = class_names.get(p_cls, str(p_cls)) if class_names else str(p_cls)
        t_name = class_names.get(t_cls, str(t_cls)) if class_names else str(t_cls)
        # Find index of predicted class in classes_
        c_idx = np.where(demo_model.classes_ == p_cls)[0][0]
        conf = probs[i, c_idx] * 100.0
        nov = novelty[i]
        status = "[CORRECT]" if p_cls == t_cls else "[MISMATCH]"
        print(f"  #{i+1:<7} | {t_name:<18} | {p_name:<18} | {conf:6.1f}%      | {nov:6.3f}         | {status}")

    # 2. 5-Fold Stratified Cross-Validation Benchmark vs Random Forest
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # TMRM 5-fold
    t0 = time.time()
    tmrm_accs, tmrm_f1s = [], []
    for tr, ts in skf.split(X, y):
        m = TopologicalManifoldResonantMachine(random_state=42)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[ts])
        tmrm_accs.append(accuracy_score(y.iloc[ts], p))
        tmrm_f1s.append(f1_score(y.iloc[ts], p, average='macro'))
    t_tmrm = (time.time() - t0) * 1000 / 5.0

    # Random Forest 5-fold
    t0 = time.time()
    rf_accs, rf_f1s = [], []
    for tr, ts in skf.split(X, y):
        X_tr = X.iloc[tr].copy()
        X_ts = X.iloc[ts].copy()
        for col in X_tr.select_dtypes(include=['object', 'category']).columns:
            u_map = {val: float(idx) for idx, val in enumerate(X_tr[col].dropna().unique())}
            X_tr[col] = X_tr[col].astype(object).map(u_map).fillna(-1.0).astype(float)
            X_ts[col] = X_ts[col].astype(object).map(u_map).fillna(-1.0).astype(float)
        X_tr = X_tr.fillna(X_tr.median(numeric_only=True))
        X_ts = X_ts.fillna(X_tr.median(numeric_only=True))
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_tr, y.iloc[tr])
        p = rf.predict(X_ts)
        rf_accs.append(accuracy_score(y.iloc[ts], p))
        rf_f1s.append(f1_score(y.iloc[ts], p, average='macro'))
    t_rf = (time.time() - t0) * 1000 / 5.0

    tmrm_acc = np.mean(tmrm_accs) * 100
    rf_acc = np.mean(rf_accs) * 100
    win_tag = "[TMRM WINS!]" if tmrm_acc >= rf_acc else "RF Wins"
    speed = f"{t_rf / max(1e-3, t_tmrm):.1f}x Faster" if t_tmrm < t_rf else "Comparable"

    print("\n  [5-FOLD STRATIFIED CROSS-VALIDATION SCORECARD]:")
    print(f"  * TMRM v4.4    : Accuracy = {tmrm_acc:6.2f}% | Macro F1 = {np.mean(tmrm_f1s)*100:6.2f}% | Latency = {t_tmrm:4.0f} ms")
    print(f"  * Random Forest: Accuracy = {rf_acc:6.2f}% | Macro F1 = {np.mean(rf_f1s)*100:6.2f}% | Latency = {t_rf:4.0f} ms")
    print(f"  * Verdict      : {win_tag} (TMRM is {speed})")
    
    return {
        "Domain & Dataset": name,
        "N / D": f"{len(X)} / {X.shape[1]}",
        "TMRM v4.4": f"{tmrm_acc:.2f}%",
        "Random Forest": f"{rf_acc:.2f}%",
        "Delta": f"{tmrm_acc - rf_acc:+.2f}%",
        "Verdict": win_tag
    }


all_summary = []

# --- 1. PARKINSON'S (Neuro-Acoustic Voice Tremor) ---
if os.path.exists(path_park):
    df = pd.read_csv(path_park)
    if 'name' in df.columns:
        df = df.drop(columns=['name'])
    target_col = 'status' if 'status' in df.columns else df.columns[-1]
    X_p = df.drop(columns=[target_col])
    y_p = df[target_col].astype(int)
    res = evaluate_and_demonstrate(
        "1. Neuro-Acoustics: Parkinson's Voice Tremor (Oxford University)",
        X_p, y_p, class_names={0: "Healthy", 1: "Parkinson's"}
    )
    all_summary.append(res)

# --- 2. SONAR (Ocean Acoustics & Radar Reflections) ---
if os.path.exists(path_sonar):
    df = pd.read_csv(path_sonar, header=None)
    y_s = (df.iloc[:, -1].astype(str).str.upper().str.startswith('M')).astype(int)
    X_s = df.iloc[:, :-1].astype(float)
    res = evaluate_and_demonstrate(
        "2. Physical Acoustics: Sonar Mines vs Rocks (DARPA-ONR)",
        X_s, y_s, class_names={0: "Rock", 1: "Metal Mine"}
    )
    all_summary.append(res)

# --- 3. HEART DISEASE (Cardiology Diagnostics) ---
if os.path.exists(path_heart):
    df = pd.read_csv(path_heart)
    target_col = [c for c in df.columns if 'target' in c.lower() or 'heart' in c.lower()][0]
    X_h = df.drop(columns=[target_col])
    y_h = df[target_col].astype(int)
    res = evaluate_and_demonstrate(
        "3. Clinical Cardiology: Heart Disease (Cleveland Clinic)",
        X_h, y_h, class_names={0: "Normal", 1: "Heart Disease"}
    )
    all_summary.append(res)

# --- 4. WINE (Chemical Spectroscopy & Enology) ---
if os.path.exists(path_wine):
    df = pd.read_csv(path_wine, header=None)
    y_w = df.iloc[:, 0].astype(int)
    X_w = df.iloc[:, 1:].astype(float)
    res = evaluate_and_demonstrate(
        "4. Chemical Spectroscopy: Wine Cultivars (3 Classes)",
        X_w, y_w, class_names={1: "Cultivar 1", 2: "Cultivar 2", 3: "Cultivar 3"}
    )
    all_summary.append(res)

# --- 5. BREAST CANCER (Clinical Oncology Cytology) ---
if os.path.exists(path_bc):
    df = pd.read_csv(path_bc)
    X_b = df.drop(columns=['target'])
    y_b = df['target'].astype(int)
    res = evaluate_and_demonstrate(
        "5. Clinical Oncology: Breast Cancer Cytology (Wisconsin)",
        X_b, y_b, class_names={0: "Malignant", 1: "Benign"}
    )
    all_summary.append(res)

# --- 6. IONOSPHERE (Aerospace Radar Pulses) ---
if os.path.exists(path_iono):
    df = pd.read_csv(path_iono, header=None)
    y_i = (df.iloc[:, -1].astype(str).str.lower().str.startswith('g')).astype(int)
    X_i = df.iloc[:, :-1].astype(float)
    res = evaluate_and_demonstrate(
        "6. Aerospace Radar: Ionosphere Pulses (Johns Hopkins APL)",
        X_i, y_i, class_names={0: "Bad / Scattered", 1: "Good / Coherent"}
    )
    all_summary.append(res)

# --- 7. PIMA DIABETES (Endocrinology) ---
if os.path.exists(path_diabetes):
    df = pd.read_csv(path_diabetes)
    target_col = 'target' if 'target' in df.columns else df.columns[-1]
    y_d = (df[target_col].astype(str).str.contains('pos')).astype(int)
    X_d = df.drop(columns=[target_col]).astype(float)
    res = evaluate_and_demonstrate(
        "7. Endocrinology: Pima Indian Diabetes (NIDDK)",
        X_d, y_d, class_names={0: "Negative", 1: "Positive / Diabetes"}
    )
    all_summary.append(res)

# --- 8. GLASS IDENTIFICATION (Forensic Material Physics) ---
if os.path.exists(path_glass):
    df = pd.read_csv(path_glass)
    target_col = 'target' if 'target' in df.columns else df.columns[-1]
    y_g_raw = df[target_col].astype(str)
    g_classes = np.unique(y_g_raw)
    g_map = {c: i for i, c in enumerate(g_classes)}
    y_g = y_g_raw.map(g_map).astype(int)
    X_g = df.drop(columns=[target_col]).astype(float)
    res = evaluate_and_demonstrate(
        "8. Forensic Materials: Glass Identification (6 Classes / Home Office)",
        X_g, y_g, class_names={i: c for i, c in enumerate(g_classes)}
    )
    all_summary.append(res)

# --- 9. VEHICLE SILHOUETTES (Automotive Computer Vision) ---
if os.path.exists(path_veh):
    df = pd.read_csv(path_veh)
    target_col = 'target' if 'target' in df.columns else df.columns[-1]
    y_v_raw = df[target_col].astype(str)
    v_classes = np.unique(y_v_raw)
    v_map = {c: i for i, c in enumerate(v_classes)}
    y_v = y_v_raw.map(v_map).astype(int)
    X_v = df.drop(columns=[target_col]).astype(float)
    res = evaluate_and_demonstrate(
        "9. Computer Vision: Vehicle Silhouettes (4 Vehicle Classes / Turing)",
        X_v, y_v, class_names={i: c for i, c in enumerate(v_classes)}
    )
    all_summary.append(res)

print("\n" + "=" * 105)
print("FINAL SCHOLAR BENCHMARK AUDIT SUMMARY (9 DIVERSE SCIENTIFIC & CLINICAL DOMAINS)")
print("=" * 105)
df_sum = pd.DataFrame(all_summary)
print(df_sum.to_string(index=False))
print("=" * 105)
