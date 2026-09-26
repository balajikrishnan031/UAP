"""
Module 2 - Part A: Adaptive Feature Weighting Engine (AFWE).
Computes dynamic composite feature quality scores W_i and scales the input matrix:
W_i = Rel(X_i, Y) * (1 - Red(X_i)) * (1 - Noise(X_i))
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import spearmanr

from uap.core.contracts import ProblemType, FeatureWeightingConfig


class AdaptiveFeatureWeightingEngine:
    """
    Computes data-health-aware continuous feature weights and transforms X -> X_weighted.
    """

    def __init__(self, noise_threshold: float = 0.04):
        self.noise_threshold = noise_threshold
        self.feature_weights: Dict[str, float] = {}
        self.dropped_features: List[str] = []
        self.W_diag: np.ndarray = np.array([])
        self.feature_names: List[str] = []

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, problem_type: ProblemType
    ) -> Tuple[pd.DataFrame, FeatureWeightingConfig]:
        """
        Calculates AFWE weights and applies diagonal matrix transformation X_weighted = X * W.
        """
        self.feature_names = list(X.columns)
        n_samples, n_features = X.shape

        # Clean NaNs for weighting computation
        X_numeric = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True).fillna(0))

        if X_numeric.shape[1] == 0 or y is None:
            # If no numeric features, return uniform weights
            uniform_weights = {f: 1.0 for f in self.feature_names}
            config = FeatureWeightingConfig(
                feature_weights=uniform_weights,
                active_weights=[1.0] * len(self.feature_names),
                dropped_features=[],
            )
            return X, config

        # 1. Relevance Score: Mutual Information + Absolute Spearman Correlation
        try:
            if problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
                mi = mutual_info_classif(X_numeric, y, random_state=42)
            else:
                mi = mutual_info_regression(X_numeric, y, random_state=42)
            # Normalize MI to [0, 1]
            mi_max = np.max(mi) if np.max(mi) > 1e-6 else 1.0
            mi_norm = mi / mi_max
        except Exception:
            mi_norm = np.ones(X_numeric.shape[1]) * 0.5

        # Spearman correlation with target
        spearman_scores = []
        for col in X_numeric.columns:
            try:
                corr, _ = spearmanr(X_numeric[col], y)
                spearman_scores.append(abs(corr) if not np.isnan(corr) else 0.0)
            except Exception:
                spearman_scores.append(0.0)
        spearman_scores = np.array(spearman_scores)

        # Composite Relevance: Rel(X_i, Y) = (MI + RankCorr) / 2
        relevance = (mi_norm + spearman_scores) / 2.0
        # Clip relevance to [0.05, 1.0] so informative features aren't zeroed prematurely
        relevance = np.clip(relevance, 0.05, 1.0)

        # 2. Redundancy Penalty: Red(X_i) = max_{j != i, Rel_j > Rel_i} |rho(X_i, X_j)|
        corr_matrix = X_numeric.corr(method="spearman").abs().fillna(0).values
        redundancy = np.zeros(X_numeric.shape[1])

        for i in range(X_numeric.shape[1]):
            overlapping_corrs = []
            for j in range(X_numeric.shape[1]):
                if i != j and relevance[j] > relevance[i]:
                    overlapping_corrs.append(corr_matrix[i, j])
            if overlapping_corrs:
                redundancy[i] = min(0.9, max(overlapping_corrs))
            else:
                redundancy[i] = 0.0

        # 3. Noise & Outlier Discount:
        # Noise(X_i) = min(1.0, Outliers(X_i)/N + MissingRate(X_i)/2)
        noise_penalty = np.zeros(X_numeric.shape[1])
        for idx, col in enumerate(X_numeric.columns):
            vals = X[col] if col in X.columns else X_numeric[col]
            missing_rate = vals.isnull().mean()
            vals_clean = vals.dropna()
            if len(vals_clean) > 10:
                q1, q3 = vals_clean.quantile(0.25), vals_clean.quantile(0.75)
                iqr = q3 - q1
                if iqr > 1e-8:
                    outliers = ((vals_clean < q1 - 1.5 * iqr) | (vals_clean > q3 + 1.5 * iqr)).sum()
                    outlier_rate = outliers / len(vals_clean)
                else:
                    outlier_rate = 0.0
            else:
                outlier_rate = 0.0
            noise_penalty[idx] = min(0.85, float(outlier_rate + (missing_rate / 2.0)))

        # 4. Final Composite Feature Quality Score:
        # W_i = Rel(X_i, Y) * (1 - Red(X_i)) * (1 - Noise(X_i))
        final_numeric_weights = relevance * (1.0 - redundancy) * (1.0 - noise_penalty)

        # Map to all features
        weights_dict = {}
        dropped = []
        for f in self.feature_names:
            if f in X_numeric.columns:
                idx = list(X_numeric.columns).index(f)
                w = float(final_numeric_weights[idx])
                if w < self.noise_threshold:
                    dropped.append(f)
                    w = 0.0
            else:
                w = 0.8  # Default weight for non-numeric/categorical
            weights_dict[f] = round(w, 4)

        self.feature_weights = weights_dict
        self.dropped_features = dropped
        self.W_diag = np.array([weights_dict[f] for f in self.feature_names])

        # Apply diagonal transformation matrix: X_weighted = X * W
        X_weighted = X.copy()
        for f in self.feature_names:
            if pd.api.types.is_numeric_dtype(X_weighted[f]):
                X_weighted[f] = X_weighted[f] * weights_dict[f]

        config = FeatureWeightingConfig(
            feature_weights=weights_dict,
            active_weights=self.W_diag.tolist(),
            dropped_features=dropped,
            transformation_applied="Diagonal_Matrix_Multiplication",
        )

        return X_weighted, config

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies learned weights to a new dataset (e.g. test set).
        """
        X_weighted = X.copy()
        for f in self.feature_names:
            if f in X_weighted.columns and pd.api.types.is_numeric_dtype(X_weighted[f]):
                w = self.feature_weights.get(f, 1.0)
                X_weighted[f] = X_weighted[f] * w
        return X_weighted
