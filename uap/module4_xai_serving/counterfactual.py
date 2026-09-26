"""
Module 4 - Part 5: Feasible Counterfactual Engine (FACE).
Generates actionable, bounded, realistic recourse plans to flip unfavorable outcomes:
  min_delta  ||delta||_MAD + lambda1 * L_target(f(x+delta), y*) + Immutable_Barrier
Respects immutable features (Age, Gender, Historical records) and directional constraints.
"""

from typing import Any, Callable, Dict, List, Optional, Set
import numpy as np
import pandas as pd


class FeasibleCounterfactualEngine:
    """
    Constrained Counterfactual Optimizer that produces mathematically sound
    and practically achievable intervention plans for humans.
    """

    def __init__(
        self,
        predictor_func: Callable[[pd.DataFrame], np.ndarray],
        feature_names: List[str],
        predict_proba_func: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
        immutable_features: Optional[List[str]] = None,
        direction_constraints: Optional[Dict[str, str]] = None,
    ):
        self.predictor_func = predictor_func
        self.predict_proba_func = predict_proba_func
        self.feature_names = feature_names
        self.immutable_features: Set[str] = set(immutable_features or [])
        self.direction_constraints: Dict[str, str] = direction_constraints or {}
        self.feature_mad: Dict[str, float] = {}
        self.feature_bounds: Dict[str, Dict[str, float]] = {}

    def fit_bounds(self, X_train: pd.DataFrame) -> "FeasibleCounterfactualEngine":
        """
        Calculates Median Absolute Deviation (MAD) and plausible [min, max] domain bounds.
        """
        for col in self.feature_names:
            if col in X_train.columns and pd.api.types.is_numeric_dtype(X_train[col]):
                vals = X_train[col].dropna()
                med = float(vals.median())
                mad = float(np.median(np.abs(vals - med)))
                self.feature_mad[col] = max(1e-4, mad if mad > 0 else float(vals.std()) or 1.0)
                self.feature_bounds[col] = {
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }
            else:
                self.feature_mad[col] = 1.0
                self.feature_bounds[col] = {"min": -np.inf, "max": np.inf}

        return self

    def _get_target_probability(self, df_in: pd.DataFrame, target_outcome: Any) -> float:
        """Helper to get probability of target outcome if predict_proba is supported."""
        if self.predict_proba_func is not None:
            try:
                probs = self.predict_proba_func(df_in)
                if probs is not None and len(probs) > 0:
                    prob_row = probs[0]
                    if isinstance(target_outcome, (int, np.integer)) and 0 <= target_outcome < len(prob_row):
                        return float(prob_row[target_outcome])
                    return float(np.max(prob_row))
            except Exception:
                pass
        # Fallback to binary check
        pred = self.predictor_func(df_in)[0]
        return 1.0 if pred == target_outcome else 0.0

    def generate_recourse(
        self,
        sample: pd.Series,
        target_outcome: Any = 1,
        max_features_to_change: int = 3,
        step_factor: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Solves constrained recourse search:
        Finds the sparsest, lowest-MAD combination of mutable feature modifications
        that flips prediction f(x + delta) to target_outcome.
        """
        sample_df = pd.DataFrame([sample])
        current_pred = self.predictor_func(sample_df)[0]

        if current_pred == target_outcome:
            return {
                "status": "ALREADY_FAVORABLE",
                "current_prediction": current_pred,
                "message": "Current outcome already matches target state. No recourse needed.",
                "actionable_plan": [],
            }

        mutable_features = [
            f for f in self.feature_names
            if f not in self.immutable_features and f in self.feature_bounds
        ]

        if not mutable_features:
            return {
                "status": "NO_MUTABLE_FEATURES",
                "current_prediction": current_pred,
                "message": "All features are immutable. Cannot generate actionable recourse.",
                "actionable_plan": [],
            }

        candidate_plans = []
        step_multipliers = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

        # Stage 1: Single-Feature Search with Fine-to-Coarse MAD Steps
        for col in mutable_features:
            mad = self.feature_mad.get(col, 1.0)
            orig_val = float(sample.get(col, 0.0))
            bounds = self.feature_bounds.get(col, {"min": -np.inf, "max": np.inf})
            constraint = self.direction_constraints.get(col, "any")

            test_directions = []
            if constraint in ["positive_only", "any"]:
                test_directions.append(1.0)
            if constraint in ["negative_only", "any"]:
                test_directions.append(-1.0)

            for direction in test_directions:
                for step_mult in step_multipliers:
                    delta = direction * step_mult * mad
                    new_val = float(np.clip(orig_val + delta, bounds["min"], bounds["max"]))

                    if abs(new_val - orig_val) < 1e-4:
                        continue

                    probe_df = sample_df.copy()
                    probe_df[col] = new_val
                    new_pred = self.predictor_func(probe_df)[0]

                    normalized_cost = abs(new_val - orig_val) / mad

                    if new_pred == target_outcome:
                        candidate_plans.append({
                            "features_changed": [col],
                            "actions": [{
                                "feature": col,
                                "original_value": round(orig_val, 3),
                                "suggested_value": round(new_val, 3),
                                "change_delta": round(new_val - orig_val, 3),
                                "normalized_cost_mad": round(normalized_cost, 3),
                            }],
                            "total_cost": round(normalized_cost, 3),
                            "flipped_outcome": new_pred,
                        })
                        # Found minimal single-step in this direction, break to next
                        break

        # Stage 2: Probability-Guided Greedy Coordinate Descent (for multi-feature recourse)
        if not candidate_plans and len(mutable_features) >= 2:
            current_solution = sample.copy()
            active_modified: Dict[str, float] = {}
            max_steps = 15

            for _ in range(max_steps):
                best_feature = None
                best_new_val = None
                best_prob_gain = -1.0
                curr_prob = self._get_target_probability(pd.DataFrame([current_solution]), target_outcome)

                for col in mutable_features:
                    if len(active_modified) >= max_features_to_change and col not in active_modified:
                        continue

                    mad = self.feature_mad.get(col, 1.0)
                    cur_val = float(current_solution[col])
                    bounds = self.feature_bounds.get(col, {"min": -np.inf, "max": np.inf})
                    constraint = self.direction_constraints.get(col, "any")

                    dirs = []
                    if constraint in ["positive_only", "any"]:
                        dirs.append(1.0)
                    if constraint in ["negative_only", "any"]:
                        dirs.append(-1.0)

                    for d in dirs:
                        for step_size in [0.5, 1.0, 1.5, 2.0, 2.5, 3.5]:
                            candidate_val = float(np.clip(cur_val + d * step_size * mad, bounds["min"], bounds["max"]))
                            if abs(candidate_val - cur_val) < 1e-4:
                                continue

                            test_sol = current_solution.copy()
                            test_sol[col] = candidate_val

                            # Check if this immediately flips the outcome
                            if self.predictor_func(pd.DataFrame([test_sol]))[0] == target_outcome:
                                active_modified[col] = candidate_val
                                actions = []
                                total_mad_cost = 0.0
                                for feat, n_val in active_modified.items():
                                    o_val = float(sample[feat])
                                    cost = abs(n_val - o_val) / self.feature_mad.get(feat, 1.0)
                                    total_mad_cost += cost
                                    actions.append({
                                        "feature": feat,
                                        "original_value": round(o_val, 3),
                                        "suggested_value": round(n_val, 3),
                                        "change_delta": round(n_val - o_val, 3),
                                        "normalized_cost_mad": round(cost, 3),
                                    })
                                candidate_plans.append({
                                    "features_changed": list(active_modified.keys()),
                                    "actions": actions,
                                    "total_cost": round(total_mad_cost, 3),
                                    "flipped_outcome": target_outcome,
                                })
                                break

                            test_prob = self._get_target_probability(pd.DataFrame([test_sol]), target_outcome)
                            gain = (test_prob - curr_prob) / max(0.5, step_size)

                            if gain > best_prob_gain and gain > 1e-4:
                                best_prob_gain = gain
                                best_feature = col
                                best_new_val = candidate_val

                        if candidate_plans:
                            break

                if candidate_plans:
                    break

                if best_feature is not None and best_new_val is not None:
                    current_solution[best_feature] = best_new_val
                    active_modified[best_feature] = best_new_val

                    # Check if flipped
                    probe_df = pd.DataFrame([current_solution])
                    if self.predictor_func(probe_df)[0] == target_outcome:
                        actions = []
                        total_mad_cost = 0.0
                        for feat, n_val in active_modified.items():
                            o_val = float(sample[feat])
                            cost = abs(n_val - o_val) / self.feature_mad.get(feat, 1.0)
                            total_mad_cost += cost
                            actions.append({
                                "feature": feat,
                                "original_value": round(o_val, 3),
                                "suggested_value": round(n_val, 3),
                                "change_delta": round(n_val - o_val, 3),
                                "normalized_cost_mad": round(cost, 3),
                            })

                        candidate_plans.append({
                            "features_changed": list(active_modified.keys()),
                            "actions": actions,
                            "total_cost": round(total_mad_cost, 3),
                            "flipped_outcome": target_outcome,
                        })
                        break
                else:
                    break

        if candidate_plans:
            # Pick lowest MAD normalized cost plan
            candidate_plans.sort(key=lambda p: p["total_cost"])
            best_plan = candidate_plans[0]
            return {
                "status": "RECOURSE_FOUND",
                "current_prediction": current_pred,
                "target_prediction": target_outcome,
                "minimal_action_plan": best_plan["actions"],
                "total_mad_cost": best_plan["total_cost"],
                "immutable_features_preserved": list(self.immutable_features),
            }

        return {
            "status": "RECOURSE_UNATTAINABLE_WITHIN_BOUNDS",
            "current_prediction": current_pred,
            "target_prediction": target_outcome,
            "message": "Target outcome could not be achieved without violating realistic bounds.",
            "immutable_features_preserved": list(self.immutable_features),
        }
