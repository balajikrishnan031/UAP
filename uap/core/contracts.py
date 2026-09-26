"""
Core Data Contracts and Schema Definitions for the UAP Framework.
Defines DCV (Module 1), MSC (Module 2), and EMA (Module 3) inter-module artifacts.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class ProblemType(str, Enum):
    BINARY_CLASSIFICATION = "Binary_Classification"
    MULTICLASS_CLASSIFICATION = "Multiclass_Classification"
    REGRESSION = "Regression"
    TIME_SERIES = "Time_Series_Forecasting"
    CLUSTERING = "Clustering"


@dataclass
class DatasetSummary:
    total_samples: int
    total_features: int
    problem_type: ProblemType
    target_name: Optional[str] = None
    feature_names: List[str] = field(default_factory=list)
    continuous_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    temporal_features: List[str] = field(default_factory=list)
    missing_rate_overall: float = 0.0


@dataclass
class DataHealthScores:
    imbalance_ratio: float = 1.0
    dimensionality_ratio: float = 0.0
    outlier_severity_index: float = 0.0
    multicollinearity_flag: bool = False
    non_linearity_index: float = 0.0
    high_correlation_pairs: int = 0
    missing_value_severity: float = 0.0


@dataclass
class DatasetCharacteristicVector:
    """
    Contract 1: Emitted by Module 1 (Profiler) -> Ingested by Module 2 (Routing & AFWE).
    """
    dataset_summary: DatasetSummary
    data_health_scores: DataHealthScores
    recommended_precautions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_summary": {
                "total_samples": self.dataset_summary.total_samples,
                "total_features": self.dataset_summary.total_features,
                "problem_type": self.dataset_summary.problem_type.value,
                "target_name": self.dataset_summary.target_name,
                "continuous_features_count": len(self.dataset_summary.continuous_features),
                "categorical_features_count": len(self.dataset_summary.categorical_features),
                "missing_rate_overall": round(self.dataset_summary.missing_rate_overall, 4),
            },
            "data_health_scores": {
                "imbalance_ratio": round(self.data_health_scores.imbalance_ratio, 3),
                "dimensionality_ratio": round(self.data_health_scores.dimensionality_ratio, 5),
                "outlier_severity_index": round(self.data_health_scores.outlier_severity_index, 3),
                "multicollinearity_flag": self.data_health_scores.multicollinearity_flag,
                "non_linearity_index": round(self.data_health_scores.non_linearity_index, 3),
                "high_correlation_pairs": self.data_health_scores.high_correlation_pairs,
            },
            "recommended_precautions": self.recommended_precautions,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class FeatureWeightingConfig:
    feature_weights: Dict[str, float] = field(default_factory=dict)
    active_weights: List[float] = field(default_factory=list)
    dropped_features: List[str] = field(default_factory=list)
    transformation_applied: str = "Diagonal_Matrix_Multiplication"


@dataclass
class RoutingDecision:
    primary_strategy: str
    selected_models: List[str]
    excluded_models: List[str]
    resampling_strategy: str
    scaling_strategy: str


@dataclass
class ModelStrategyConfiguration:
    """
    Contract 2: Emitted by Module 2 (Routing) -> Ingested by Module 3 (Training/HPO).
    """
    feature_weighting: FeatureWeightingConfig
    routing_decision: RoutingDecision
    pipeline_modifications: Dict[str, Any] = field(default_factory=dict)
    hyperparameter_search_space: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_weighting": {
                "active_features_count": len(self.feature_weighting.feature_weights) - len(self.feature_weighting.dropped_features),
                "dropped_features_count": len(self.feature_weighting.dropped_features),
                "dropped_features": self.feature_weighting.dropped_features,
                "weights_summary": {k: round(v, 3) for k, v in list(self.feature_weighting.feature_weights.items())[:10]}
            },
            "routing_decision": {
                "primary_strategy": self.routing_decision.primary_strategy,
                "selected_models": self.routing_decision.selected_models,
                "excluded_models": self.routing_decision.excluded_models,
                "resampling_strategy": self.routing_decision.resampling_strategy,
                "scaling_strategy": self.routing_decision.scaling_strategy,
            },
            "pipeline_modifications": self.pipeline_modifications,
            "hyperparameter_search_space": self.hyperparameter_search_space,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class PerformanceMetrics:
    primary_metric_name: str
    validation_score: float
    cross_fold_std: float
    test_score: Optional[float] = None
    ece_calibration_error: Optional[float] = None
    robustness_decay_score: float = 0.0


@dataclass
class EvaluatedModelArtifact:
    """
    Contract 3: Emitted by Module 3 (Training) -> Ingested by Module 4 (XAI & Serving).
    """
    model_execution_summary: Dict[str, Any]
    performance_metrics: PerformanceMetrics
    final_hyperparameters: Dict[str, Any]
    deployment_readiness: Dict[str, Any]
    serialized_model_path: Optional[str] = None
    winning_model_name: str = ""
    feature_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_execution_summary": self.model_execution_summary,
            "performance_metrics": {
                "primary_metric_name": self.performance_metrics.primary_metric_name,
                "validation_score": round(self.performance_metrics.validation_score, 4),
                "cross_fold_std": round(self.performance_metrics.cross_fold_std, 4),
                "test_score": round(self.performance_metrics.test_score, 4) if self.performance_metrics.test_score is not None else None,
                "ece_calibration_error": round(self.performance_metrics.ece_calibration_error, 4) if self.performance_metrics.ece_calibration_error is not None else None,
                "robustness_decay_score": round(self.performance_metrics.robustness_decay_score, 4),
            },
            "final_hyperparameters": self.final_hyperparameters,
            "deployment_readiness": self.deployment_readiness,
            "winning_model_name": self.winning_model_name,
            "serialized_model_path": self.serialized_model_path,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class UserUtilitySpec:
    """
    User/Application-level operational constraints and risk preferences.
    """
    risk_tolerance: str = "Medium"  # "Low", "Medium", "High"
    latency_budget_ms: str = "Standard"  # "Ultra_Low_<10ms", "Standard", "Batch"
    interpretability_need: str = "Standard"  # "High", "Standard"
    conformal_coverage_target: float = 0.95


@dataclass
class GoalRiskProfile:
    """
    Strategy synthesis: Explains why this specific configuration was selected.
    """
    selected_metric: str
    cv_strategy: str
    risk_posture: str
    abstention_threshold: float
    conformal_coverage: float
    strategy_rationale: str
    key_tradeoffs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_metric": self.selected_metric,
            "cv_strategy": self.cv_strategy,
            "risk_posture": self.risk_posture,
            "abstention_threshold": self.abstention_threshold,
            "conformal_coverage": self.conformal_coverage,
            "strategy_rationale": self.strategy_rationale,
            "key_tradeoffs": self.key_tradeoffs,
        }


@dataclass
class PredictionReliabilityCard:
    """
    UAP 4.0 Core User-Facing Artifact: Provides comprehensive trustworthiness audit for an inference.
    """
    prediction: Any
    confidence_score: float
    conformal_set_or_interval: Any
    data_quality_score: float
    ood_risk_level: str
    drift_risk_level: str
    decision: str  # "SAFE_TO_PREDICT" or "ABSTAIN_HUMAN_REVIEW"
    abstention_reason: Optional[str] = None
    actionable_summary: str = ""

    @property
    def conformal_interval(self) -> Any:
        return self.conformal_set_or_interval

    @property
    def ood_risk(self) -> str:
        return self.ood_risk_level

    @property
    def drift_risk(self) -> str:
        return self.drift_risk_level

    @property
    def reason(self) -> Optional[str]:
        return self.abstention_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction,
            "confidence_score": round(self.confidence_score, 4),
            "conformal_set_or_interval": self.conformal_set_or_interval,
            "conformal_interval": self.conformal_set_or_interval,
            "data_quality_score": round(self.data_quality_score, 4),
            "ood_risk_level": self.ood_risk_level,
            "ood_risk": self.ood_risk_level,
            "drift_risk_level": self.drift_risk_level,
            "drift_risk": self.drift_risk_level,
            "decision": self.decision,
            "abstention_reason": self.abstention_reason,
            "reason": self.abstention_reason,
            "actionable_summary": self.actionable_summary,
        }

    def format_text_card(self) -> str:
        border = "═" * 45
        divider = "─" * 45
        lines = [
            border,
            "       UAP 4.0 PREDICTION RELIABILITY CARD",
            border,
            "  PREDICTION",
            divider,
            f"  Result             : {self.prediction}",
            f"  Conformal (95% Set): {self.conformal_set_or_interval}",
            divider,
            "  RELIABILITY AUDIT",
            divider,
            f"  Confidence Score   : {self.confidence_score * 100:.1f}%",
            f"  Data Quality Index : {self.data_quality_score * 100:.1f}%",
            f"  OOD (Novelty) Risk : {self.ood_risk_level}",
            f"  Concept Drift Risk : {self.drift_risk_level}",
            divider,
            f"  DECISION           : {self.decision}",
        ]
        if self.abstention_reason:
            lines.append(f"  Flag / Action      : {self.abstention_reason}")
        lines.append(border)
        return "\n".join(lines)


@dataclass
class LivingDecisionDossier:
    """
    UAP 5.0 Core Output Artifact: The Living Decision Dossier.
    Radically transcends simple point-prediction numbers by synthesizing:
    1. Predictive Output & 95% Conformal Safety Bounds.
    2. Multi-tier Trust Certification (OOD, Drift, Data Quality, Abstention).
    3. Primary Causal / Attribution Drivers (Why did this happen?).
    4. Actionable Counterfactual Recourse (What exact minimal action flips the outcome?).
    5. Actionable Business Impact Guidance.
    """
    prediction: Any
    confidence_score: float
    conformal_interval_95: Any
    trust_badge: str  # "SAFE_TO_PREDICT" or "ABSTAIN_HUMAN_REVIEW"
    abstention_reason: Optional[str] = None
    ood_risk_level: str = "LOW"
    drift_risk_level: str = "NORMAL"
    data_quality_score: float = 1.0
    top_causal_drivers: List[Dict[str, Any]] = field(default_factory=list)
    actionable_recourse: Optional[Dict[str, Any]] = None
    business_impact_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction,
            "confidence_score": round(self.confidence_score, 4),
            "conformal_interval_95": self.conformal_interval_95,
            "trust_badge": self.trust_badge,
            "abstention_reason": self.abstention_reason,
            "ood_risk_level": self.ood_risk_level,
            "drift_risk_level": self.drift_risk_level,
            "data_quality_score": round(self.data_quality_score, 4),
            "top_causal_drivers": self.top_causal_drivers,
            "actionable_recourse": self.actionable_recourse,
            "business_impact_summary": self.business_impact_summary,
        }

    def format_executive_dossier(self) -> str:
        border = "=" * 70
        divider = "-" * 70
        badge_symbol = "[SAFE]" if self.trust_badge == "SAFE_TO_PREDICT" else "[ALERT]"

        lines = [
            border,
            "              [UAP 5.0 LIVING DECISION DOSSIER]",
            border,
            f"  [1] PREDICTIVE OUTCOME",
            divider,
            f"      Result                  : {self.prediction}",
            f"      Calibrated Confidence   : {self.confidence_score * 100:.1f}%",
            f"      95% Conformal Set/Range : {self.conformal_interval_95}",
            divider,
            f"  [2] SAFETY & TRUST AUDIT",
            divider,
            f"      Decision Trust Badge    : {badge_symbol} {self.trust_badge}",
            f"      OOD (Novelty) Risk      : {self.ood_risk_level}",
            f"      Concept Drift Status    : {self.drift_risk_level}",
            f"      Data Quality Index      : {self.data_quality_score * 100:.1f}%",
        ]

        if self.abstention_reason:
            lines.append(f"      Safety Flag / Action    : {self.abstention_reason}")

        lines.append(divider)
        lines.append("  [3] WHY DID THIS HAPPEN? (Key Contributing Drivers)")
        lines.append(divider)
        if self.top_causal_drivers:
            for idx, driver in enumerate(self.top_causal_drivers[:4], 1):
                feat = driver.get("feature", "Unknown")
                imp = driver.get("impact", driver.get("importance", 0.0))
                val = driver.get("value", "")
                val_str = f" [Current Value: {val}]" if val != "" else ""
                lines.append(f"      {idx}. {feat}{val_str} -> Impact: {imp:+.3f}")
        else:
            lines.append("      Top global features weighted by UAP AFWE & TreeSHAP.")

        lines.append(divider)
        lines.append("  [4] ACTIONABLE RECOURSE (What Action Flips The Outcome?)")
        lines.append(divider)
        if self.actionable_recourse and self.actionable_recourse.get("status") == "RECOURSE_FOUND":
            rec = self.actionable_recourse
            cost = rec.get("total_mad_cost", rec.get("cost", 0.0))
            lines.append(f"      Recourse Status         : RECOURSE_FOUND (Difficulty/Cost: {cost:.2f})")
            lines.append(f"      Prescribed Action Plan  :")
            for act in rec.get("minimal_action_plan", [])[:3]:
                lines.append(f"        -> Change '{act['feature']}': {act['original_value']} -> {act['suggested_value']} (Shift: {act['change_delta']:+})")
        elif self.actionable_recourse and self.actionable_recourse.get("status") == "ALREADY_FAVORABLE":
            lines.append("      Status: Outcome is already favorable. No intervention required.")
        else:
            lines.append("      Recourse Status: Automated recourse not triggered or outcome is optimal.")

        lines.append(divider)
        lines.append(f"  [5] STRATEGIC GUIDANCE: {self.business_impact_summary}")
        lines.append(border)
        return "\n".join(lines)
