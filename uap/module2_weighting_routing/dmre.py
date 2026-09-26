"""
Module 2 - Part B: Dynamic Model Routing Engine (DMRE).
Reads the Dataset Characteristic Vector (DCV) and produces the Model Strategy Configuration (MSC).
"""

from typing import Any, Dict, List, Tuple
from uap.core.contracts import (
    DatasetCharacteristicVector,
    ProblemType,
    RoutingDecision,
    ModelStrategyConfiguration,
    FeatureWeightingConfig,
)


class DynamicModelRoutingEngine:
    """
    Evaluates DCV characteristics against the mathematical routing matrix to pick
    the optimal candidate models, scaling, and hyperparameter search space.
    """

    def generate_strategy(
        self, dcv: DatasetCharacteristicVector, feature_weight_config: FeatureWeightingConfig
    ) -> ModelStrategyConfiguration:
        problem_type = dcv.dataset_summary.problem_type
        health = dcv.data_health_scores

        selected_models: List[str] = []
        excluded_models: List[str] = []
        pipeline_modifications: Dict[str, Any] = {}
        search_space: Dict[str, Any] = {}

        # 1. Scaling Strategy
        if health.outlier_severity_index > 0.08:
            scaling_strategy = "RobustScaler"
            pipeline_modifications["scaling"] = "RobustScaler (Median & IQR)"
        else:
            scaling_strategy = "StandardScaler"
            pipeline_modifications["scaling"] = "StandardScaler"

        # 2. Resampling & Class Weighting
        if problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            if health.imbalance_ratio > 4.0:
                resampling_strategy = "Dynamic_Class_Weighting"
                pipeline_modifications["imbalance_handling"] = f"scale_pos_weight={round(health.imbalance_ratio, 2)}"
            else:
                resampling_strategy = "None"
                pipeline_modifications["imbalance_handling"] = "Balanced"
        else:
            resampling_strategy = "None"
            pipeline_modifications["imbalance_handling"] = "Not_Applicable"

        # 3. Model Decision & Routing Matrix
        n_samples = dcv.dataset_summary.total_samples
        if problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            selected_models.append("TMRMClassifier")
            if n_samples < 2500:
                selected_models.extend(["RandomForestClassifier", "ExtraTreesClassifier", "LightGBMClassifier", "XGBoostClassifier"])
                if health.non_linearity_index <= 0.70:
                    selected_models.append("LogisticRegression")
                else:
                    selected_models.append("CatBoostClassifier")
            elif health.non_linearity_index > 0.75:
                excluded_models.append("LogisticRegression")
                selected_models.extend(["LightGBMClassifier", "XGBoostClassifier", "CatBoostClassifier", "RandomForestClassifier", "ExtraTreesClassifier"])
            else:
                selected_models.extend(["LightGBMClassifier", "XGBoostClassifier", "LogisticRegression", "CatBoostClassifier", "RandomForestClassifier"])

            if health.high_correlation_pairs == 0 and n_samples < 2500:
                selected_models.append("GaussianNB")

            if health.outlier_severity_index > 0.10:
                if "LogisticRegression" not in excluded_models:
                    excluded_models.append("LogisticRegression")

            if health.dimensionality_ratio > 0.10:
                excluded_models.append("KNeighborsClassifier")
                if "ExtraTreesClassifier" not in selected_models:
                    selected_models.append("ExtraTreesClassifier")
            elif n_samples < 1500 and "KNeighborsClassifier" not in excluded_models:
                selected_models.append("KNeighborsClassifier")

            # Hyperparameter search space
            search_space = {
                "LightGBMClassifier": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "num_leaves": [15, 31, 63],
                    "subsample": [0.7, 0.9, 1.0],
                },
                "CatBoostClassifier": {
                    "iterations": [50, 100],
                    "learning_rate": [0.03, 0.1],
                    "depth": [4, 6],
                },
                "GradientBoostingClassifier": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.03, 0.1],
                    "max_depth": [3, 5],
                },
                "XGBoostClassifier": {
                    "n_estimators": [50, 100, 150],
                    "learning_rate": [0.03, 0.1],
                    "max_depth": [3, 6, 8],
                },
                "ExtraTreesClassifier": {
                    "n_estimators": [50, 100, 150],
                    "max_depth": [6, 12, None],
                },
                "RandomForestClassifier": {
                    "n_estimators": [50, 100],
                    "max_depth": [6, 10, None],
                },
                "LogisticRegression": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                },
                "SVC": {
                    "C": [0.1, 1.0, 10.0],
                },
                "GaussianNB": {},
                "MLPClassifier": {
                    "alpha": [1e-4, 1e-2],
                },
                "KNeighborsClassifier": {
                    "n_neighbors": [3, 5, 9],
                },
                "TMRMClassifier": {
                    "harmonic_octaves": [2, 3],
                    "metric_regularization": [1e-3, 1e-2],
                },
                "DynamicFeatureGraphModel": {
                    "embed_dim": [16, 32],
                    "epochs": [30, 50],
                }
            }

        elif problem_type == ProblemType.REGRESSION:
            selected_models.append("TMRMRegressor")
            if n_samples < 2500:
                selected_models.extend(["LightGBMRegressor", "XGBoostRegressor", "RandomForestRegressor"])
                if health.non_linearity_index <= 0.70:
                    selected_models.append("RidgeRegressor")
                else:
                    selected_models.append("CatBoostRegressor")
            elif health.non_linearity_index > 0.75:
                excluded_models.append("LinearRegression")
                selected_models.extend(["LightGBMRegressor", "XGBoostRegressor", "CatBoostRegressor", "ExtraTreesRegressor"])
            else:
                selected_models.extend(["LightGBMRegressor", "XGBoostRegressor", "RidgeRegressor", "CatBoostRegressor", "RandomForestRegressor"])

            if health.multicollinearity_flag or health.high_correlation_pairs > 3:
                selected_models.append("ElasticNet")

            if health.outlier_severity_index > 0.10:
                excluded_models.append("LinearRegression")
                if "HuberRegressor" not in selected_models:
                    selected_models.append("HuberRegressor")

            if health.dimensionality_ratio > 0.10:
                excluded_models.append("KNeighborsRegressor")
            elif n_samples < 2000 and "KNeighborsRegressor" not in excluded_models:
                selected_models.append("KNeighborsRegressor")

            # Hyperparameter search space
            search_space = {
                "LightGBMRegressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "num_leaves": [15, 31, 63],
                },
                "CatBoostRegressor": {
                    "iterations": [50, 100],
                    "learning_rate": [0.03, 0.1],
                    "depth": [4, 6],
                },
                "GradientBoostingRegressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.03, 0.1],
                    "max_depth": [3, 5],
                },
                "XGBoostRegressor": {
                    "n_estimators": [50, 100, 150],
                    "learning_rate": [0.03, 0.1],
                    "max_depth": [3, 6, 8],
                },
                "ExtraTreesRegressor": {
                    "n_estimators": [50, 100, 150],
                    "max_depth": [6, 12, None],
                },
                "RandomForestRegressor": {
                    "n_estimators": [50, 100],
                    "max_depth": [6, 10, None],
                },
                "HuberRegressor": {
                    "epsilon": [1.1, 1.35, 1.75],
                    "alpha": [0.0001, 0.01, 1.0],
                },
                "RidgeRegressor": {
                    "alpha": [0.1, 1.0, 10.0],
                },
                "ElasticNet": {
                    "alpha": [0.01, 0.1, 1.0],
                    "l1_ratio": [0.2, 0.5, 0.8],
                },
                "SVR": {
                    "C": [0.1, 1.0, 10.0],
                },
                "MLPRegressor": {
                    "alpha": [1e-4, 1e-2],
                },
                "KNeighborsRegressor": {
                    "n_neighbors": [3, 5, 9],
                },
                "TMRMRegressor": {
                    "harmonic_octaves": [2, 3],
                    "metric_regularization": [1e-3, 1e-2],
                }
            }

        else:
            # Clustering or Time-Series
            selected_models = ["LightGBMRegressor", "RandomForestRegressor"]
            search_space = {"LightGBMRegressor": {"n_estimators": [100]}}

        # Deduplicate & pick top 4 candidate architectures
        selected_models = list(dict.fromkeys(selected_models))[:4]
        primary_strategy = "Stacking_Ensemble" if len(selected_models) > 1 else "Single_Best_Model"

        routing_decision = RoutingDecision(
            primary_strategy=primary_strategy,
            selected_models=selected_models,
            excluded_models=list(set(excluded_models)),
            resampling_strategy=resampling_strategy,
            scaling_strategy=scaling_strategy,
        )

        return ModelStrategyConfiguration(
            feature_weighting=feature_weight_config,
            routing_decision=routing_decision,
            pipeline_modifications=pipeline_modifications,
            hyperparameter_search_space={m: search_space.get(m, {}) for m in selected_models},
        )
