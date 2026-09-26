"""
Module 4 - Part 3: The Abstention Gate (Selective Prediction Engine).
Evaluates tri-criterion safety:
  G(x) = I(S_conf(x) >= tau_conf  AND  S_ood(x) <= tau_ood  AND  Q_data(x) >= tau_qual)
Returns PREDICT, PREDICT_WITH_WARNING, or ABSTAIN (Unknown / Human-in-the-Loop review).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis


class AbstentionGate:
    """
    Intelligent safety gate that prevents overconfident hallucinations on
    out-of-distribution (OOD) or low-quality/uncertain records.
    """

    def __init__(
        self,
        tau_conf: float = 0.65,
        tau_ood: float = 3.5,
        tau_qual: float = 0.60,
    ):
        self.tau_conf = tau_conf
        self.tau_ood = tau_ood
        self.tau_qual = tau_qual

        self.mu_train: Optional[np.ndarray] = None
        self.inv_cov_train: Optional[np.ndarray] = None
        self.feature_weights: Dict[str, float] = {}
        self.feature_names: List[str] = []
        self.feature_iqr: Dict[str, Tuple[float, float]] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> "AbstentionGate":
        """
        Learns the empirical training distribution manifold for OOD Mahalanobis calculation.
        """
        self.feature_names = list(X_train.columns)
        self.feature_weights = feature_weights or {f: 1.0 for f in self.feature_names}

        X_numeric = X_train.select_dtypes(include=[np.number]).fillna(0)
        self.mu_train = X_numeric.mean().values

        # Compute empirical covariance with regularization for numerical stability
        cov = np.cov(X_numeric.values, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-4
        self.inv_cov_train = np.linalg.pinv(cov)

        # Precompute IQR bounds for outlier detection
        for col in X_numeric.columns:
            q1 = float(X_numeric[col].quantile(0.25))
            q3 = float(X_numeric[col].quantile(0.75))
            iqr = q3 - q1
            self.feature_iqr[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        return self

    def evaluate_sample(
        self,
        sample: pd.Series,
        raw_pred: Any,
        raw_prob: Optional[np.ndarray] = None,
        ensemble_preds: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates an individual input sample against the 3 safety criteria.
        """
        numeric_sample = pd.to_numeric(sample, errors="coerce").fillna(0).values

        # 1. Compute Out-of-Distribution (OOD) Distance S_ood
        s_ood = 0.0
        if self.mu_train is not None and self.inv_cov_train is not None:
            try:
                diff = numeric_sample - self.mu_train
                s_ood = float(np.sqrt(np.dot(np.dot(diff, self.inv_cov_train), diff.T)))
            except Exception:
                s_ood = 0.0

        # 2. Compute Confidence Score S_conf
        s_conf = 1.0
        if raw_prob is not None and len(raw_prob) >= 2:
            sorted_probs = np.sort(raw_prob)[::-1]
            margin = float(sorted_probs[0] - sorted_probs[1])
            # Shannon entropy normalized to [0, 1]
            probs_clean = np.where(raw_prob == 0, 1e-6, raw_prob)
            entropy = -np.sum(probs_clean * np.log(probs_clean)) / np.log(len(raw_prob))
            s_conf = float(np.clip(margin * (1.0 - entropy), 0.0, 1.0))
        elif ensemble_preds is not None and len(ensemble_preds) > 1:
            # Measure agreement across ensemble
            std = float(np.std(ensemble_preds))
            s_conf = float(np.exp(-std))

        # 3. Compute Instance Quality Score Q_data
        total_w = sum(self.feature_weights.values()) or 1.0
        defect_w = 0.0
        corrupted_features = []

        for col, val in sample.items():
            col_w = self.feature_weights.get(col, 0.5)
            # Check missing
            if pd.isna(val):
                defect_w += col_w
                corrupted_features.append(f"{col} (Missing)")
                continue

            # Check extreme outlier
            if col in self.feature_iqr:
                low, high = self.feature_iqr[col]
                try:
                    num_val = float(val)
                    if num_val < low or num_val > high:
                        defect_w += col_w * 0.5
                        corrupted_features.append(f"{col} (Extreme Outlier: {num_val:.2f})")
                except Exception:
                    pass

        q_data = max(0.0, float(1.0 - (defect_w / total_w)))

        # Tri-Criteria Verdict G(x)
        reasons = []
        is_ood = s_ood > self.tau_ood
        is_uncertain = s_conf < self.tau_conf
        is_bad_quality = q_data < self.tau_qual

        if is_ood:
            reasons.append(f"Out-of-Distribution Manifold Shift (Distance: {s_ood:.2f} > {self.tau_ood})")
        if is_uncertain:
            reasons.append(f"Predictive Epistemic Uncertainty High (Confidence: {s_conf:.2f} < {self.tau_conf})")
        if is_bad_quality:
            reasons.append(f"Critical Feature Corruption (Quality: {q_data:.2f} < {self.tau_qual})")

        # Multi-Criteria Gate G(x) Decision
        if is_ood or (is_uncertain and is_bad_quality) or s_ood > (self.tau_ood * 1.5):
            status = "ABSTAIN"
            decision_code = 0
            human_action = "TRIGGER_HUMAN_IN_THE_LOOP_REVIEW"
        elif is_uncertain or is_bad_quality or s_ood > (self.tau_ood * 0.8):
            status = "PREDICT_WITH_WARNING"
            decision_code = 1
            human_action = "PROCEED_WITH_DOMAIN_CAUTION"
        else:
            status = "PREDICT"
            decision_code = 1
            human_action = "AUTOMATED_EXECUTION_SAFE"

        return {
            "gate_decision": decision_code,
            "status": status,
            "human_action": human_action,
            "prediction": raw_pred if decision_code == 1 else "UNKNOWN",
            "confidence_score": round(s_conf, 3),
            "ood_distance": round(s_ood, 3),
            "quality_score": round(q_data, 3),
            "reasons": reasons,
            "corrupted_features": corrupted_features,
        }
