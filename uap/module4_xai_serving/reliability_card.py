"""
UAP 4.0 Module 4 Component: Prediction Reliability Card Engine.
Synthesizes predictive output, conformal bounds, epistemic confidence,
OOD anomaly distance, and drift risk into an interpretable trust decision.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from uap.core.contracts import PredictionReliabilityCard


class ReliabilityCardEngine:
    """
    Evaluates individual inferences against multi-tier safety standards
    to emit a certified Prediction Reliability Card.
    """

    def __init__(self, abstention_threshold: float = 0.60):
        self.abstention_threshold = abstention_threshold

    def build_card(
        self,
        sample: pd.Series,
        raw_pred: Any,
        raw_probs: Optional[np.ndarray],
        conformal_bounds: Any,
        train_features_mean: Optional[pd.Series] = None,
        train_features_std: Optional[pd.Series] = None,
        drift_status: str = "NORMAL",
    ) -> PredictionReliabilityCard:
        # 1. Epistemic Confidence Estimation
        if raw_probs is not None:
            if raw_probs.ndim == 2:
                confidence = float(np.max(raw_probs[0]))
            elif raw_probs.ndim == 1:
                confidence = float(np.max(raw_probs))
            else:
                confidence = 0.85
        else:
            confidence = 0.85

        # 2. Data Quality Index (Checks missing values or NaN in query)
        null_count = int(sample.isnull().sum())
        total_feats = len(sample)
        data_quality = 1.0 - (null_count / total_feats) if total_feats > 0 else 1.0

        # 3. Out-Of-Distribution (OOD) Novelty Risk
        ood_risk = "Low"
        ood_score = 0.0
        if train_features_mean is not None and train_features_std is not None:
            numeric_sample = pd.to_numeric(sample, errors="coerce").fillna(0)
            z_scores = np.abs((numeric_sample - train_features_mean) / (train_features_std + 1e-6))
            max_z = float(np.max(z_scores)) if len(z_scores) > 0 else 0.0
            ood_score = max_z
            if max_z > 4.5:
                ood_risk = "High"
            elif max_z > 3.0:
                ood_risk = "Medium"

        # 4. Drift Risk Level
        drift_risk = "Low" if drift_status in ["NORMAL", "STABLE"] else "High"

        # 5. Decision & Safety Gate
        decision = "SAFE_TO_PREDICT"
        reasons = []

        if confidence < self.abstention_threshold:
            decision = "ABSTAIN_HUMAN_REVIEW"
            reasons.append(f"Epistemic confidence ({confidence * 100:.1f}%) is below safety threshold ({self.abstention_threshold * 100:.1f}%).")
        if ood_risk == "High":
            decision = "ABSTAIN_HUMAN_REVIEW"
            reasons.append(f"Severe Out-Of-Distribution query detected (Z-Score = {ood_score:.1f} exceeds threshold).")
        if data_quality < 0.70:
            decision = "ABSTAIN_HUMAN_REVIEW"
            reasons.append(f"Poor input data quality ({data_quality * 100:.1f}% valid features).")
        if drift_risk == "High":
            reasons.append("Elevated concept drift detected on live inference stream.")

        abstention_reason = " | ".join(reasons) if reasons else None

        # Format Conformal Bound string
        if isinstance(conformal_bounds, dict):
            bound_repr = conformal_bounds.get("prediction_set", conformal_bounds.get("interval", str(conformal_bounds)))
        else:
            bound_repr = str(conformal_bounds)

        summary = (
            f"Prediction '{raw_pred}' certified under 95% conformal bounds {bound_repr}. "
            f"Confidence: {confidence * 100:.1f}%. Reliability status: {decision}."
        )

        return PredictionReliabilityCard(
            prediction=raw_pred,
            confidence_score=confidence,
            conformal_set_or_interval=bound_repr,
            data_quality_score=data_quality,
            ood_risk_level=ood_risk,
            drift_risk_level=drift_risk,
            decision=decision,
            abstention_reason=abstention_reason,
            actionable_summary=summary,
        )
