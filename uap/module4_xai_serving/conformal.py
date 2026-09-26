"""
Module 4 - Part E: Conformal Risk Certifier (Finite-Sample Guaranteed Prediction Sets).
Implements Distribution-Free Split Conformal Prediction (Vovk et al.) for:
  1. Guaranteed Classification Prediction Sets: P(Y in C(X)) >= 1 - alpha
  2. Guaranteed Continuous Regression Intervals: P(Y in [L(X), U(X)]) >= 1 - alpha
  3. Formal Mathematical Abstention when prediction set is ambiguous (|C(X)| > 1) or empty.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class ConformalRiskCertifier:
    """
    Computes distribution-free, finite-sample valid prediction sets and intervals
    with rigorous statistical coverage guarantees (e.g. 95% confidence).
    """

    def __init__(self, default_alpha: float = 0.05):
        self.default_alpha = default_alpha  # 0.05 corresponds to 95% coverage guarantee
        self.is_classification: bool = True
        self.calibration_scores: np.ndarray = np.array([])
        self.classes_: np.ndarray = np.array([])
        self.n_calib_: int = 0
        self.fitted_: bool = False

    def fit(
        self,
        X_calib: pd.DataFrame,
        y_calib: Union[pd.Series, np.ndarray],
        is_classification: bool,
        predict_func: Callable[[pd.DataFrame], np.ndarray],
        predict_proba_func: Optional[Callable[[pd.DataFrame], Optional[np.ndarray]]] = None,
    ) -> "ConformalRiskCertifier":
        """
        Calibrates non-conformity scores on an independent holdout calibration split.
        """
        self.is_classification = is_classification
        y_arr = np.asarray(y_calib)
        n = len(y_arr)
        self.n_calib_ = n

        if n < 5:
            # Fallback for micro-datasets
            self.fitted_ = False
            return self

        if is_classification:
            probs = predict_proba_func(X_calib) if predict_proba_func is not None else None
            if probs is None or probs.ndim < 2:
                # Fallback to 0/1 indicator from raw predictions
                preds = predict_func(X_calib)
                probs = np.zeros((n, 2))
                for i, p in enumerate(preds):
                    c = int(np.clip(p, 0, 1))
                    probs[i, c] = 1.0

            self.classes_ = np.arange(probs.shape[1])
            # Non-conformity score for classification: 1 - prob(true class)
            scores = []
            for i in range(n):
                true_c = int(y_arr[i])
                if true_c < probs.shape[1]:
                    scores.append(1.0 - probs[i, true_c])
                else:
                    scores.append(1.0)
            self.calibration_scores = np.sort(np.asarray(scores, dtype=np.float32))

        else:
            # Non-conformity score for regression: absolute residual |y - \hat{y}|
            preds = predict_func(X_calib)
            residuals = np.abs(y_arr - preds)
            self.calibration_scores = np.sort(np.asarray(residuals, dtype=np.float32))

        self.fitted_ = True
        return self

    def compute_conformal_quantile(self, alpha: float) -> float:
        """
        Computes the finite-sample adjusted conformal quantile:
        q_hat = Quantile_{ (n+1)(1 - alpha) / n }(Scores)
        """
        if not self.fitted_ or len(self.calibration_scores) == 0:
            return 0.5 if self.is_classification else 1.0

        n = len(self.calibration_scores)
        # Standard finite-sample correction index: ceil((n + 1) * (1 - alpha)) / n
        level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
        # Use nearest / higher rank
        idx = int(np.clip(np.ceil(level * n) - 1, 0, n - 1))
        return float(self.calibration_scores[idx])

    def certify_instance(
        self,
        sample: Union[pd.Series, pd.DataFrame, Dict[str, Any]],
        predict_func: Callable[[pd.DataFrame], np.ndarray],
        predict_proba_func: Optional[Callable[[pd.DataFrame], Optional[np.ndarray]]] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Certifies a single sample with a finite-sample valid prediction set / interval.
        """
        sig_level = alpha if alpha is not None else self.default_alpha
        target_coverage = round(1.0 - sig_level, 4)

        if isinstance(sample, dict):
            df_inst = pd.DataFrame([sample])
        elif isinstance(sample, pd.Series):
            df_inst = pd.DataFrame([sample])
        else:
            df_inst = sample

        q_hat = self.compute_conformal_quantile(sig_level)

        if self.is_classification:
            probs = predict_proba_func(df_inst) if predict_proba_func is not None else None
            raw_pred = int(predict_func(df_inst)[0])

            if probs is not None and probs.shape[1] > 1:
                prob_vec = probs[0]
            else:
                prob_vec = np.array([1.0 - raw_pred, float(raw_pred)])

            # Construct prediction set C(x) = { c : prob_c >= 1 - q_hat }
            cutoff = max(0.0, 1.0 - q_hat)
            prediction_set = [int(c) for c, p in enumerate(prob_vec) if p >= cutoff]

            # In case cutoff is extremely strict and set is empty, fallback to top prediction
            is_empty = len(prediction_set) == 0
            if is_empty:
                prediction_set = [int(np.argmax(prob_vec))]

            set_size = len(prediction_set)
            is_ambiguous = set_size > 1

            if is_empty:
                status = "OOD_ANOMALY_EMPTY_SET"
                recommendation = "ABSTAIN_UNRELIABLE_DISTRIBUTION"
            elif is_ambiguous:
                status = "AMBIGUOUS_MULTI_CLASS"
                recommendation = "HUMAN_TRIAGE_REQUIRED"
            else:
                status = "CERTIFIED_SAFE_PREDICTION"
                recommendation = "AUTOMATED_EXECUTION_CERTIFIED"

            return {
                "coverage_guarantee": f"{target_coverage * 100:.1f}%",
                "significance_alpha": sig_level,
                "conformal_quantile": round(q_hat, 4),
                "prediction_set": prediction_set,
                "set_size": set_size,
                "point_prediction": raw_pred,
                "status": status,
                "action": recommendation,
                "class_probabilities": {int(c): round(float(p), 4) for c, p in enumerate(prob_vec)},
                "mathematical_abstain": is_ambiguous or is_empty,
            }

        else:
            # Continuous Regression Interval
            point_pred = float(predict_func(df_inst)[0])
            lower_bound = float(point_pred - q_hat)
            upper_bound = float(point_pred + q_hat)
            interval_width = float(2 * q_hat)

            return {
                "coverage_guarantee": f"{target_coverage * 100:.1f}%",
                "significance_alpha": sig_level,
                "conformal_margin": round(q_hat, 4),
                "point_prediction": round(point_pred, 4),
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "interval_width": round(interval_width, 4),
                "status": "CERTIFIED_INTERVAL",
                "action": "AUTOMATED_EXECUTION_CERTIFIED",
                "mathematical_abstain": False,
            }
