"""
Module 1: Dataset Profiler & Automatic Problem Detection System.
Inspects raw tabular datasets, detects task type, and extracts the Dataset Characteristic Vector (DCV).
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

from uap.core.contracts import (
    ProblemType,
    DatasetSummary,
    DataHealthScores,
    DatasetCharacteristicVector,
)


class DatasetProfiler:
    """
    Automated Profiler implementing Phase 1, Phase 2, and Phase 3 of Module 1.
    """

    def __init__(self, target_column: Optional[str] = None):
        self.target_column = target_column

    def profile(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[DatasetCharacteristicVector, pd.DataFrame, pd.Series]:
        """
        Profiles the dataset and separates X and y.
        Returns:
            dcv: DatasetCharacteristicVector
            X: Cleaned Feature DataFrame
            y: Target Series (or None if unsupervised)
        """
        target_col = target_column or self.target_column
        df_clean = df.copy()

        # Phase 1: Column Type Inference
        col_types = self._infer_column_types(df_clean, target_col)

        # Phase 2: Target & Task Classifier
        if target_col is None or target_col not in df_clean.columns:
            target_col = self._auto_detect_target(df_clean, col_types)

        if target_col and target_col in df_clean.columns:
            y = df_clean[target_col]
            X = df_clean.drop(columns=[target_col])
            problem_type = self._detect_problem_type(y, df_clean)
        else:
            y = None
            X = df_clean
            problem_type = ProblemType.CLUSTERING

        # Update feature lists after removing target
        continuous_features = [c for c in col_types["continuous"] if c != target_col]
        categorical_features = [c for c in col_types["categorical"] if c != target_col]
        temporal_features = [c for c in col_types["temporal"] if c != target_col]

        missing_overall = float(df_clean.isnull().sum().sum() / (df_clean.shape[0] * df_clean.shape[1]))

        summary = DatasetSummary(
            total_samples=len(df_clean),
            total_features=X.shape[1],
            problem_type=problem_type,
            target_name=target_col,
            feature_names=list(X.columns),
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            temporal_features=temporal_features,
            missing_rate_overall=missing_overall,
        )

        # Phase 3: Deep Data Health Profiler
        health_scores, precautions = self._compute_health_scores(X, y, summary)

        dcv = DatasetCharacteristicVector(
            dataset_summary=summary,
            data_health_scores=health_scores,
            recommended_precautions=precautions,
        )

        return dcv, X, y

    def _infer_column_types(self, df: pd.DataFrame, target_col: Optional[str]) -> dict:
        continuous = []
        categorical = []
        temporal = []
        text = []

        n = len(df)
        if n == 0:
            return {"continuous": [], "categorical": [], "temporal": [], "text": []}

        for col in df.columns:
            # Check for datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                temporal.append(col)
                continue

            # Try parsing date strings if string
            if df[col].dtype == object:
                sample = df[col].dropna().head(20)
                try:
                    if len(sample) > 0 and pd.to_datetime(sample, errors='coerce').notnull().mean() > 0.8:
                        temporal.append(col)
                        continue
                except Exception:
                    pass

            # Numeric check
            if pd.api.types.is_numeric_dtype(df[col]):
                unique_ratio = df[col].nunique() / n
                if unique_ratio > 0.05 or df[col].nunique() > 50:
                    continuous.append(col)
                else:
                    categorical.append(col)
            elif df[col].dtype == object or pd.api.types.is_categorical_dtype(df[col]):
                if df[col].nunique() / n > 0.8 and df[col].nunique() > 100:
                    text.append(col)
                else:
                    categorical.append(col)
            else:
                categorical.append(col)

        return {
            "continuous": continuous,
            "categorical": categorical,
            "temporal": temporal,
            "text": text,
        }

    def _auto_detect_target(self, df: pd.DataFrame, col_types: dict) -> Optional[str]:
        common_target_names = ["target", "label", "y", "price", "churn", "class", "outcome", "status"]
        for col in df.columns:
            if col.lower() in common_target_names:
                return col
        # If last column is a candidate
        if len(df.columns) > 1:
            return df.columns[-1]
        return None

    def _detect_problem_type(self, y: pd.Series, df: pd.DataFrame) -> ProblemType:
        n = len(y.dropna())
        if n == 0:
            return ProblemType.CLUSTERING

        u_count = y.nunique()
        u_ratio = u_count / n

        # Check for sequence / temporal ordering in df
        has_temporal = any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)

        if pd.api.types.is_numeric_dtype(y):
            # Regression Index RI = U(Y)/N
            if u_ratio > 0.05 and u_count > 20:
                if has_temporal:
                    return ProblemType.TIME_SERIES
                return ProblemType.REGRESSION

        if u_count == 2:
            return ProblemType.BINARY_CLASSIFICATION
        elif 2 < u_count <= 50:
            return ProblemType.MULTICLASS_CLASSIFICATION
        else:
            return ProblemType.REGRESSION

    def _compute_health_scores(
        self, X: pd.DataFrame, y: Optional[pd.Series], summary: DatasetSummary
    ) -> Tuple[DataHealthScores, List[str]]:
        scores = DataHealthScores()
        precautions = []
        n_samples, n_features = X.shape

        # 1. Dimensionality Ratio DR = D / N
        if n_samples > 0:
            scores.dimensionality_ratio = float(n_features / n_samples)
            if scores.dimensionality_ratio > 0.10:
                precautions.append("High_Dimensionality_Risk: Use L1 regularization or Tree Ensembles")

        # 2. Imbalance Ratio IR (For classification)
        if summary.problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION] and y is not None:
            counts = y.value_counts()
            if len(counts) > 1 and counts.min() > 0:
                scores.imbalance_ratio = float(counts.max() / counts.min())
                if scores.imbalance_ratio > 4.0:
                    precautions.append("Severe_Class_Imbalance: Apply dynamic loss weighting or SMOTE")

        # 3. Outlier Severity Index (OSI via IQR)
        numeric_cols = list(summary.continuous_features)
        if numeric_cols and n_samples > 10:
            total_numeric_cells = n_samples * len(numeric_cols)
            outlier_cells = 0
            for col in numeric_cols:
                vals = X[col].dropna()
                q1 = vals.quantile(0.25)
                q3 = vals.quantile(0.75)
                iqr = q3 - q1
                if iqr > 1e-8:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_cells += ((vals < lower) | (vals > upper)).sum()

            # For regression, also check target outlier rate
            if summary.problem_type == ProblemType.REGRESSION and y is not None:
                y_clean = y.dropna()
                if len(y_clean) > 10:
                    y_q1, y_q3 = y_clean.quantile(0.25), y_clean.quantile(0.75)
                    y_iqr = y_q3 - y_q1
                    if y_iqr > 1e-8:
                        y_outliers = ((y_clean < y_q1 - 1.5 * y_iqr) | (y_clean > y_q3 + 1.5 * y_iqr)).sum()
                        y_outlier_rate = y_outliers / len(y_clean)
                        # Factor in target outliers if high
                        if y_outlier_rate > 0.05:
                            outlier_cells += int(y_outlier_rate * total_numeric_cells)

            scores.outlier_severity_index = float(outlier_cells / max(1, total_numeric_cells))
            if scores.outlier_severity_index > 0.08:
                precautions.append("High_Outlier_Severity: Use RobustScaler and Huber/Tree models")

        # 4. Multicollinearity Score
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = X[numeric_cols].corr(method='spearman').abs()
                np.fill_diagonal(corr_matrix.values, 0)
                high_corr_count = int((corr_matrix > 0.85).sum().sum() // 2)
                scores.high_correlation_pairs = high_corr_count
                scores.multicollinearity_flag = high_corr_count > 0
                if scores.multicollinearity_flag:
                    precautions.append("Multicollinearity_Detected: Apply AFWE redundancy penalty")
            except Exception:
                pass

        # 5. Non-Linearity Index (NLI)
        if y is not None and len(numeric_cols) > 0 and n_samples >= 50:
            try:
                sample_size = min(1000, n_samples)
                sample_idx = np.random.choice(n_samples, sample_size, replace=False)
                X_sub = X[numeric_cols].iloc[sample_idx].fillna(0)
                y_sub = y.iloc[sample_idx]

                from sklearn.preprocessing import StandardScaler
                X_sub_scaled = StandardScaler().fit_transform(X_sub)

                if summary.problem_type == ProblemType.REGRESSION:
                    lin_model = Ridge(alpha=1.0).fit(X_sub_scaled, y_sub)
                    tree_model = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_sub_scaled, y_sub)
                    mse_lin = mean_squared_error(y_sub, lin_model.predict(X_sub_scaled))
                    mse_tree = mean_squared_error(y_sub, tree_model.predict(X_sub_scaled))
                    if mse_lin > 1e-8:
                        nli = 1.0 - (mse_tree / mse_lin)
                        scores.non_linearity_index = float(np.clip(nli, 0.0, 1.0))
                else:
                    # Classification NLI via Accuracy comparison
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y_sub)
                    lin_model = RidgeClassifier().fit(X_sub_scaled, y_encoded)
                    tree_model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_sub_scaled, y_encoded)
                    acc_lin = accuracy_score(y_encoded, lin_model.predict(X_sub_scaled))
                    acc_tree = accuracy_score(y_encoded, tree_model.predict(X_sub_scaled))
                    if acc_tree > acc_lin:
                        scores.non_linearity_index = float(np.clip((acc_tree - acc_lin) * 2, 0.0, 1.0))
                    else:
                        scores.non_linearity_index = 0.1
            except Exception:
                scores.non_linearity_index = 0.5

            if scores.non_linearity_index > 0.65:
                precautions.append("High_Non_Linearity: Prioritize Gradient Boosting (LightGBM/XGBoost)")

        return scores, precautions
