"""
UAP 4.0 Module 1 Extension: Goal & Risk Strategy Analyzer.
Translates Dataset Characteristic Vectors (DCV) and User Utility Constraints into
formal strategy decisions, answering: "Why was this specific configuration chosen?"
"""

from typing import List, Optional
from uap.core.contracts import (
    ProblemType,
    DatasetCharacteristicVector,
    UserUtilitySpec,
    GoalRiskProfile,
)


class GoalRiskAnalyzer:
    """
    Synthesizes dataset geometry, task constraints, and risk preferences into
    an explainable operational strategy for downstream modules.
    """

    def __init__(self, default_utility_spec: Optional[UserUtilitySpec] = None):
        self.utility_spec = default_utility_spec or UserUtilitySpec()

    def analyze(
        self,
        dcv: DatasetCharacteristicVector,
        utility_spec: Optional[UserUtilitySpec] = None,
    ) -> GoalRiskProfile:
        spec = utility_spec or self.utility_spec
        summary = dcv.dataset_summary
        health = dcv.data_health_scores
        problem_type = summary.problem_type

        tradeoffs: List[str] = []
        rationales: List[str] = []

        # 1. Optimal Evaluation Metric Selection
        if problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            if health.imbalance_ratio > 3.0:
                selected_metric = "Macro_F1"
                rationales.append(
                    f"Selected Macro-F1 metric due to severe class imbalance ({health.imbalance_ratio:.1f}:1). "
                    "Standard accuracy is rejected to prevent trivial majority-class classification bias."
                )
                tradeoffs.append("Prioritizes minority class recall over raw overall accuracy.")
            else:
                selected_metric = "Macro_F1"
                rationales.append("Selected Macro-F1 to guarantee balanced per-class predictive fidelity.")
        else:
            if health.outlier_severity_index > 0.15:
                selected_metric = "MAE"
                rationales.append(
                    f"Selected Mean Absolute Error (MAE) due to heavy outlier contamination ({health.outlier_severity_index * 100:.1f}%). "
                    "RMSE is deprioritized to avoid quadratic penalization of extreme edge-case values."
                )
                tradeoffs.append("Linear penalization prevents single outlier spikes from distorting optimization.")
            else:
                selected_metric = "R2_Score"
                rationales.append("Selected R² Score / RMSE to maximize variance explained in continuous space.")

        # 2. Cross-Validation & Partitioning Strategy
        if problem_type == ProblemType.TIME_SERIES:
            cv_strategy = "TimeSeriesSplit(n_splits=5)"
            rationales.append("Enforced TimeSeriesSplit to prevent temporal lookahead data leakage.")
            tradeoffs.append("Chronological validation reduces effective training set size in early folds.")
        elif problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            cv_strategy = "StratifiedKFold(n_splits=3, shuffle=True)"
            rationales.append("Enforced Stratified K-Fold to maintain identical target distributions across all validation folds.")
        else:
            cv_strategy = "KFold(n_splits=3, shuffle=True)"
            rationales.append("Enforced Shuffle K-Fold for independent continuous target validation.")

        # 3. Risk Posture & Abstention Threshold
        if spec.risk_tolerance == "Low":
            risk_posture = "Strict_Safety_Critical"
            abstention_threshold = 0.70
            conformal_coverage = max(0.99, spec.conformal_coverage_target)
            rationales.append("Configured strict safety posture: High abstention threshold (70%) and 99% conformal coverage.")
            tradeoffs.append("Higher abstention rate requires human-in-the-loop review for ambiguous edge cases.")
        elif spec.risk_tolerance == "High":
            risk_posture = "Permissive_High_Coverage"
            abstention_threshold = 0.45
            conformal_coverage = min(0.90, spec.conformal_coverage_target)
            rationales.append("Configured permissive posture: Maximizes autonomous prediction coverage.")
        else:
            risk_posture = "Balanced_Industrial_Standard"
            abstention_threshold = 0.60
            conformal_coverage = spec.conformal_coverage_target
            rationales.append("Configured balanced industrial standard: 60% abstention threshold with 95% conformal bounds.")

        # 4. Latency vs. Accuracy Strategy
        if spec.latency_budget_ms == "Ultra_Low_<10ms":
            tradeoffs.append("Prioritizes shallow tree depths and fast linear baselines to meet sub-10ms SLA.")
        elif spec.latency_budget_ms == "Batch":
            tradeoffs.append("Authorizes deep tree iterations and exhaustive Bayesian exploration.")

        full_rationale = " | ".join(rationales)

        return GoalRiskProfile(
            selected_metric=selected_metric,
            cv_strategy=cv_strategy,
            risk_posture=risk_posture,
            abstention_threshold=abstention_threshold,
            conformal_coverage=conformal_coverage,
            strategy_rationale=full_rationale,
            key_tradeoffs=tradeoffs,
        )
