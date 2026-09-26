"""
TMRM Core Engine & Pretrained Model Loader
Re-exports the battle-tested TopologicalManifoldResonantMachine & StreamingTMRM
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Import official production-grade implementation from uap.models.novel_tmrm
from uap.models.novel_tmrm import TopologicalManifoldResonantMachine
from uap.models.streaming_tmrm import StreamingTMRM

TMRM = TopologicalManifoldResonantMachine


def load_model(bundle_path: Optional[str] = None):
    """Convenience loader for the pre-trained production model."""
    if bundle_path is None:
        bundle_path = str(Path(__file__).resolve().parent.parent / "models" / "tmrm_heart_disease_production.joblib")
    return joblib.load(bundle_path)


def predict_heart_disease(patient_dict: Dict[str, float]) -> Dict[str, Any]:
    """Instant 1-line prediction for a cardiac patient using pre-trained model."""
    bundle = load_model()
    model = bundle["model"]
    scaler = bundle["scaler"]
    features = bundle["feature_names"]

    # Build DataFrame to retain feature names
    df_patient = pd.DataFrame([{f: float(patient_dict.get(f, 0.0)) for f in features}])
    x_scaled = scaler.transform(df_patient)

    pred = int(model.predict(x_scaled)[0])
    probs = model.predict_proba(x_scaled)[0]
    novelty = float(model.get_epistemic_novelty(x_scaled)[0])

    is_safe = novelty < 3.0

    return {
        "prediction": pred,
        "label": "Heart Disease Risk" if pred == 1 else "Healthy / No Heart Disease",
        "confidence": float(probs[pred]),
        "probability_distribution": {
            "Healthy": float(probs[0]),
            "Risk": float(probs[1]) if len(probs) > 1 else 0.0
        },
        "novelty_score": round(novelty, 2),
        "is_safe_to_predict": is_safe
    }
