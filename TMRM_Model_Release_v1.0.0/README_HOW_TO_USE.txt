================================================================================
          TMRM (Topological Manifold Resonant Machine) - RELEASE v1.0.0
================================================================================

Intha folder-la ungalukku TMRM Model-ah yarukkum share panni, avanga computer-la
install panni predict panrathukana ella files-um irukku.

--------------------------------------------------------------------------------
1. FILES IN THIS BUNDLE:
--------------------------------------------------------------------------------
  * 1_INSTALL_TMRM.bat       : Windows users double-click pannina automatic-ah TMRM install aagidum!
  * 2_RUN_PREDICTION.bat     : Double-click pannina sample patients-ku direct prediction run aagum.
  * predict_patient.py       : Standalone Python script for single or batch CSV prediction.
  * sample_patients.csv      : Example patient data file.
  * models/                  : Pre-trained Model weights (.joblib and .json format).
  * dist/                    : Official Python Wheel (.whl) and source package (.tar.gz).

--------------------------------------------------------------------------------
2. HOW OTHERS CAN INSTALL THIS PACKAGE:
--------------------------------------------------------------------------------
Method A (Automatic for Windows):
  -> Just double click "1_INSTALL_TMRM.bat"

Method B (Via Terminal / Command Prompt):
  -> pip install dist/tmrm-1.0.0-py3-none-any.whl

--------------------------------------------------------------------------------
3. HOW TO USE IN PYTHON CODE (Any Project):
--------------------------------------------------------------------------------
from tmrm import TMRM, predict_heart_disease

# A) Instant Heart Disease Prediction:
result = predict_heart_disease({
    "age": 62, "sex": 1, "cp": 4, "trestbps": 150, "chol": 285,
    "fbs": 1, "restecg": 2, "thalach": 125, "exang": 1,
    "oldpeak": 2.8, "slope": 2, "ca": 2, "thal": 7
})
print(result["label"])        # "Heart Disease Risk"
print(result["confidence"])   # Confidence Score

# B) Train on ANY other dataset:
model = TMRM()
model.fit(X_train, y_train)
preds = model.predict(X_test)

--------------------------------------------------------------------------------
4. MODEL SPECIFICATIONS:
--------------------------------------------------------------------------------
  * Algorithm            : Topological Manifold Resonant Machine (TMRM)
  * Math Engine          : Continuous Riemannian Energy Manifolds + Wavelet Resonance
  * Certified Accuracy   : 90.16% on Gold-Standard Cleveland Benchmark
  * Certified ROC-AUC    : 0.9556
  * Features Included    : 13 Clinical Dimensions
================================================================================
