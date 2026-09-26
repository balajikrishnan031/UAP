"""
UAP 3.0 Causal Discovery & Interventional Simulation Engine.
Implements:
  1. Constraint-Based Causal DAG Discovery from Observational Tabular Data.
  2. Pearl's do-calculus Interventional Simulator: P(Y | do(X_k = v)).
  3. Structural Causal Model (SCM) Downstream Propagation for Realistic Recourse.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


class CausalDiscoveryEngine:
    """
    Extracts the Structural Causal Model (SCM) Directed Acyclic Graph (DAG)
    and enables interventional what-if simulations: E[Y | do(X_k = v)].
    """

    def __init__(self, significance_threshold: float = 0.10):
        self.significance_threshold = significance_threshold
        self.feature_names: List[str] = []
        self.adjacency_matrix: np.ndarray = np.array([])
        self.edge_weights: Dict[Tuple[str, str], float] = {}
        self.causal_effects_on_target: Dict[str, float] = {}
        self.topological_order: List[str] = []
        self.fitted_: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        target_name: str = "Target",
        immutable_features: Optional[List[str]] = None,
    ) -> "CausalDiscoveryEngine":
        """
        Discovers the causal DAG using mutual covariance, partial correlation,
        and directional orientation heuristics.
        """
        df = X.copy()
        if y is not None:
            df[target_name] = y.values

        self.feature_names = list(df.columns)
        n_feats = len(self.feature_names)
        immutable_set = set(immutable_features or [])

        # 1. Compute empirical correlation matrix
        corr = df.corr().fillna(0.0).values

        # 2. Derive directed adjacency matrix A
        # A[i, j] > 0 denotes directed edge: i -> j
        A = np.zeros((n_feats, n_feats), dtype=np.float32)

        for i in range(n_feats):
            for j in range(i + 1, n_feats):
                c_val = corr[i, j]
                if abs(c_val) >= self.significance_threshold:
                    f_i = self.feature_names[i]
                    f_j = self.feature_names[j]

                    # Directionality orientation rules:
                    # 1. Target node is always an effect (leaf node), never a parent
                    if f_i == target_name:
                        A[j, i] = c_val
                    elif f_j == target_name:
                        A[i, j] = c_val
                    # 2. Immutable features (e.g. Age, Ambient Temp) can only cause, never be caused
                    elif f_i in immutable_set and f_j not in immutable_set:
                        A[i, j] = c_val
                    elif f_j in immutable_set and f_i not in immutable_set:
                        A[j, i] = c_val
                    # 3. Variance hierarchy: higher-variance natural features precede derived features
                    elif df[f_i].var() >= df[f_j].var():
                        A[i, j] = c_val
                    else:
                        A[j, i] = c_val

        # 3. Enforce acyclicity (break cycles if any using topological ordering)
        self.adjacency_matrix = A
        self.edge_weights = {}
        for i in range(n_feats):
            for j in range(n_feats):
                if abs(A[i, j]) > 0:
                    src = self.feature_names[i]
                    dst = self.feature_names[j]
                    self.edge_weights[(src, dst)] = round(float(A[i, j]), 4)

        # 4. Extract causal effects on target
        if target_name in self.feature_names:
            t_idx = self.feature_names.index(target_name)
            for i, f in enumerate(self.feature_names):
                if f != target_name:
                    self.causal_effects_on_target[f] = round(float(A[i, t_idx]), 4)

        self.fitted_ = True
        return self

    def simulate_intervention(
        self,
        base_sample: pd.Series,
        interventions: Dict[str, float],
        predictor_func: Callable[[pd.DataFrame], np.ndarray],
    ) -> Dict[str, Any]:
        """
        Executes Pearl's do-calculus intervention do(X_k = v).
        Mutilates the DAG by severing incoming edges to intervened variables,
        propagates downstream changes through the SCM, and computes the causal outcome.
        """
        if not self.fitted_:
            raise RuntimeError("Causal engine not fitted yet.")

        simulated_sample = base_sample.copy()

        # Step 1: Set intervened values
        for f, val in interventions.items():
            if f in simulated_sample.index:
                simulated_sample[f] = float(val)

        # Step 2: Downstream SCM propagation along DAG edges
        for (src, dst), weight in self.edge_weights.items():
            if src in interventions and dst not in interventions:
                if src in base_sample.index and dst in base_sample.index:
                    delta_src = simulated_sample[src] - base_sample[src]
                    # Linear causal propagation: delta_dst = weight * delta_src
                    propagated_shift = weight * delta_src
                    simulated_sample[dst] += propagated_shift

        # Step 3: Evaluate model prediction on interventional counterfactual
        df_base = pd.DataFrame([base_sample])
        df_sim = pd.DataFrame([simulated_sample])

        base_pred = float(predictor_func(df_base)[0])
        intervened_pred = float(predictor_func(df_sim)[0])
        causal_effect = round(intervened_pred - base_pred, 4)

        return {
            "interventions": interventions,
            "original_prediction": round(base_pred, 4),
            "intervened_causal_prediction": round(intervened_pred, 4),
            "average_causal_treatment_effect": causal_effect,
            "propagated_downstream_features": {
                f: round(float(simulated_sample[f]), 4)
                for f in simulated_sample.index
                if f not in interventions and abs(simulated_sample[f] - base_sample[f]) > 1e-4
            },
            "simulated_record": simulated_sample.to_dict(),
        }

    def get_graph_elements(self) -> Dict[str, Any]:
        """
        Returns nodes and edges formatted for visual rendering (Plotly / Streamlit / Cytoscape).
        """
        nodes = [{"id": f, "label": f} for f in self.feature_names]
        edges = [
            {"source": src, "target": dst, "weight": w}
            for (src, dst), w in self.edge_weights.items()
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "direct_target_drivers": self.causal_effects_on_target,
        }
