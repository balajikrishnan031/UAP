"""
Module 4 - Part 2: Counterfactual & What-If Simulation Engine.
Enables interactive probing and actionable recourse recommendations while respecting immutability constraints.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class WhatIfRecourseEngine:
    """
    Performs scenario probing and actionable counterfactual recommendations.
    """

    def __init__(
        self,
        predictor_func: Any,
        feature_names: List[str],
        immutable_features: Optional[List[str]] = None,
    ):
        self.predictor_func = predictor_func
        self.feature_names = feature_names
        self.immutable_features = set(immutable_features or [])

    def simulate_scenario(
        self,
        base_instance: pd.Series,
        modifications: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Calculates prediction outcome when specified features are modified.
        """
        # Check immutability constraints
        violating_features = [f for f in modifications if f in self.immutable_features]
        if violating_features:
            raise ValueError(
                f"Cannot modify immutable features: {violating_features}"
            )

        base_df = pd.DataFrame([base_instance])
        modified_df = base_df.copy()

        for col, val in modifications.items():
            if col in modified_df.columns:
                modified_df[col] = val

        base_pred = self.predictor_func(base_df)[0]
        new_pred = self.predictor_func(modified_df)[0]

        delta = float(new_pred - base_pred)

        return {
            "base_prediction": round(float(base_pred), 4),
            "new_prediction": round(float(new_pred), 4),
            "prediction_delta": round(delta, 4),
            "applied_modifications": modifications,
        }

    def find_actionable_recourse(
        self,
        base_instance: pd.Series,
        target_direction: str = "increase",
        step_pct: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Heuristic recourse finder: tests perturbations on mutable features to find
        which single feature change yields the highest positive or negative impact.
        """
        base_df = pd.DataFrame([base_instance])
        base_pred = float(self.predictor_func(base_df)[0])

        candidates = []
        for col in self.feature_names:
            if col in self.immutable_features:
                continue

            current_val = float(base_instance.get(col, 0.0))
            delta_val = abs(current_val * step_pct) if abs(current_val) > 1e-4 else 1.0

            # Test +step and -step
            for factor in [1.0, -1.0]:
                test_df = base_df.copy()
                test_df[col] = current_val + factor * delta_val
                new_pred = float(self.predictor_func(test_df)[0])
                diff = new_pred - base_pred

                candidates.append({
                    "feature": col,
                    "original_value": round(current_val, 3),
                    "suggested_value": round(float(test_df[col].iloc[0]), 3),
                    "new_prediction": round(new_pred, 4),
                    "impact": round(diff, 4),
                })

        if target_direction == "increase":
            candidates.sort(key=lambda x: x["impact"], reverse=True)
        else:
            candidates.sort(key=lambda x: x["impact"])

        top_recourse = candidates[0] if candidates else None
        return {
            "base_prediction": round(base_pred, 4),
            "top_actionable_recourse": top_recourse,
            "all_evaluated_options": candidates[:5],
        }
