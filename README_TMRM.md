# TMRM: Topological Manifold Resonant Machine (v1.0.0)

A high-performance machine learning algorithm based on **Continuous Riemannian Energy Manifolds** and **Multi-Octave Wavelet Resonance**.

---

## ⚡ Installation

Install via pip in 1 line:

```bash
# From local directory:
pip install .

# Or in editable development mode:
pip install -e .
```

---

## 🚀 3-Line Quickstart

```python
from tmrm import TMRM
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. Load your data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train TMRM
model = TMRM()
model.fit(X_train, y_train)

# 3. Predict & Score
predictions = model.predict(X_test)
print(f"Accuracy: {model.score(X_test, y_test) * 100:.2f}%")
```

---

## 🩺 Instant Pre-Trained Heart Disease Prediction

No training required! Use the pre-trained clinical model directly:

```python
from tmrm import predict_heart_disease

# Input patient clinical measurements
patient = {
    "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
}

result = predict_heart_disease(patient)
print("Diagnosis:", result["label"])
print("Confidence:", result["confidence"])
```

---

## 💻 Terminal CLI Usage

```bash
# Check Model Info
python -m tmrm.cli info

# Run Instant Prediction via Command Line
python -m tmrm.cli predict --age 60 --chol 280 --trestbps 145 --cp 4
```

---

## 🌟 Key Advantages over Random Forest & XGBoost

1. **Continuous Topological Manifolds**: Captures non-linear curved geometries without jagged orthogonal decision tree splits.
2. **Inherent Self-Doubt (OOD Detection)**: Rejects corrupted or alien input without needing external outlier libraries.
3. **Prescriptive Recourse**: Mathematically calculates the minimal feature adjustments needed to move from an unsafe to a safe outcome.
4. **Adversarial Resilience**: Smooth topological manifolds resist adversarial noise injection far better than brittle tree leaf thresholds.
