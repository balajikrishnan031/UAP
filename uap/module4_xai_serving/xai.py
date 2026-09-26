"""
Module 4 - Part 1: Explainable AI (XAI) Engine.
Computes TreeSHAP feature attributions, global importance, and local waterfall explanations.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import shap


class ExplainabilityEngine:
    """
    Computes global and local Shapley attribution values using SHAP.
    """

    def __init__(self):
        self.explainer = None
        self.expected_value: float = 0.0
        self.feature_names: List[str] = []

    def fit(self, model: Any, X_background: pd.DataFrame):
        """
        Initializes SHAP explainer with a background dataset.
        """
        self.feature_names = list(X_background.columns)
        X_sample = shap.sample(X_background.fillna(0), min(100, len(X_background)), random_state=42)

        try:
            # Try TreeExplainer first for tree models
            self.explainer = shap.TreeExplainer(model, data=X_sample)
            exp_val = self.explainer.expected_value
            if isinstance(exp_val, (list, np.ndarray)):
                self.expected_value = float(exp_val[1]) if len(exp_val) > 1 else float(exp_val[0])
            else:
                self.expected_value = float(exp_val)
        except Exception:
            # Fallback to Permutation / Linear / Exact explainer
            try:
                self.explainer = shap.Explainer(model.predict, X_sample)
                self.expected_value = float(np.mean(model.predict(X_sample)))
            except Exception:
                self.explainer = None
                self.expected_value = 0.5

    def get_global_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Returns ranked global feature importances based on mean absolute SHAP values.
        """
        if self.explainer is None:
            # Fallback uniform
            return {f: 1.0 / len(self.feature_names) for f in self.feature_names}

        X_eval = X.head(100).fillna(0)
        try:
            shap_vals = self.explainer(X_eval)
            vals = shap_vals.values
            if vals.ndim == 3:  # Multiclass or binary with 2 outputs
                vals = vals[:, :, 1]
            mean_abs = np.mean(np.abs(vals), axis=0)
            importance_dict = {
                f: round(float(mean_abs[i]), 4) for i, f in enumerate(self.feature_names)
            }
            # Sort descending
            return dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        except Exception:
            return {f: 0.1 for f in self.feature_names}

    def explain_instance(self, instance: pd.Series) -> Dict[str, Any]:
        """
        Explains a single prediction (Local Waterfall decomposition).
        """
        df_inst = pd.DataFrame([instance]).fillna(0)
        if self.explainer is None:
            return {
                "base_value": self.expected_value,
                "contributions": {f: 0.0 for f in self.feature_names},
            }

        try:
            shap_obj = self.explainer(df_inst)
            vals = shap_obj.values[0]
            if vals.ndim == 2:
                vals = vals[:, 1]
            contribs = {
                f: round(float(vals[i]), 4) for i, f in enumerate(self.feature_names)
            }
            # Sort by absolute impact
            sorted_contribs = dict(
                sorted(contribs.items(), key=lambda item: abs(item[1]), reverse=True)
            )
            return {
                "base_value": round(self.expected_value, 4),
                "contributions": sorted_contribs,
                "top_positive_driver": max(sorted_contribs.items(), key=lambda x: x[1]),
                "top_negative_driver": min(sorted_contribs.items(), key=lambda x: x[1]),
            }
        except Exception:
            return {
                "base_value": self.expected_value,
                "contributions": {f: 0.0 for f in self.feature_names},
            }
