"""
Unit tests for UAP 4.0 innovations:
1. Goal & Risk Strategy Analyzer (Metric & CV Selection based on DCV)
2. Explainable Strategy Audit ("Why was this configuration chosen?")
3. Prediction Reliability Card (Multi-tier Trustworthiness & Selective Abstention)
"""

import numpy as np
import pandas as pd
import pytest

from uap.core.contracts import (
    ProblemType,
    UserUtilitySpec,
    GoalRiskProfile,
    PredictionReliabilityCard,
)
from uap.module1_profiler.profiler import DatasetProfiler
from uap.module1_profiler.goal_risk_analyzer import GoalRiskAnalyzer
from uap.engine import UAPEngine


def test_goal_risk_analyzer_imbalanced():
    profiler = DatasetProfiler()
    analyzer = GoalRiskAnalyzer()

    # Create synthetic imbalanced dataset (90 zeros, 10 ones)
    df = pd.DataFrame({
        "f1": np.random.randn(100),
        "f2": np.random.randn(100),
        "target": [0] * 90 + [1] * 10
    })

    dcv, X, y = profiler.profile(df, target_column="target")
    assert dcv.data_health_scores.imbalance_ratio >= 8.0

    profile = analyzer.analyze(dcv)
    assert profile.selected_metric == "Macro_F1"
    assert "Macro-F1" in profile.strategy_rationale
    assert "imbalance" in profile.strategy_rationale.lower()
    assert "StratifiedKFold" in profile.cv_strategy


def test_goal_risk_analyzer_risk_tolerance():
    profiler = DatasetProfiler()
    analyzer = GoalRiskAnalyzer()

    df = pd.DataFrame({
        "f1": np.random.randn(80),
        "f2": np.random.randn(80),
        "target": np.random.randint(0, 2, 80)
    })
    dcv, X, y = profiler.profile(df, target_column="target")

    # Strict low risk tolerance
    spec_strict = UserUtilitySpec(risk_tolerance="Low", conformal_coverage_target=0.99)
    profile_strict = analyzer.analyze(dcv, utility_spec=spec_strict)
    assert profile_strict.risk_posture == "Strict_Safety_Critical"
    assert profile_strict.abstention_threshold == 0.70
    assert profile_strict.conformal_coverage == 0.99

    # High risk tolerance (permissive)
    spec_perm = UserUtilitySpec(risk_tolerance="High", conformal_coverage_target=0.90)
    profile_perm = analyzer.analyze(dcv, utility_spec=spec_perm)
    assert profile_perm.risk_posture == "Permissive_High_Coverage"
    assert profile_perm.abstention_threshold == 0.45


def test_uap4_end_to_end_strategy_and_reliability_card():
    np.random.seed(42)
    X = np.random.randn(120, 4)
    # Simple linear decision with noise
    y = (X[:, 0] * 1.5 + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    df["label"] = y

    engine = UAPEngine(n_trials=2, timeout_sec=10)
    engine.fit(df, target_column="label", domain_tag="TestMedical", utility_spec=UserUtilitySpec(risk_tolerance="Low"))

    # 1. Verify Strategy Rationale
    strategy = engine.explain_strategy()
    assert "winning_model" in strategy
    assert "why_this_configuration" in strategy
    assert strategy["risk_posture"] == "Strict_Safety_Critical"
    assert len(strategy["tradeoffs"]) > 0

    # 2. Verify In-Distribution Normal Query -> SAFE TO PREDICT
    normal_query = {"f1": 1.2, "f2": 0.8, "f3": -0.1, "f4": 0.2}
    card_normal = engine.predict_with_reliability_card(normal_query)
    assert isinstance(card_normal, PredictionReliabilityCard)
    assert card_normal.data_quality_score == 1.0
    assert card_normal.ood_risk_level in ["Low", "Medium"]
    formatted_card = card_normal.format_text_card()
    assert "PREDICTION RELIABILITY CARD" in formatted_card
    assert "Confidence Score" in formatted_card

    # 3. Verify Extreme Outlier Query -> ABSTAIN_HUMAN_REVIEW
    crazy_ood_query = {"f1": 95.0, "f2": -85.0, "f3": 120.0, "f4": -110.0}
    card_ood = engine.predict_with_reliability_card(crazy_ood_query)
    assert card_ood.decision == "ABSTAIN_HUMAN_REVIEW"
    assert card_ood.ood_risk_level == "High"
    assert "Out-Of-Distribution" in str(card_ood.abstention_reason)
