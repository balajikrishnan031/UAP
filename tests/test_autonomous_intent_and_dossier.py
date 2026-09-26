"""
Unit and Integration Tests for UAP 5.0 Autonomous Intent Discovery and Living Decision Dossier.
"""

from pathlib import Path
import sys
import pytest
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from uap import UAP, LivingDecisionDossier, IntentDiscoveryCard
from uap.core.contracts import ProblemType, UserUtilitySpec
from uap.module1_profiler.autonomous_intent import AutonomousIntentEngine

def test_autonomous_intent_discovery_classification():
    engine = AutonomousIntentEngine()
    df = pd.read_csv("data/real_world/breast_cancer.csv")

    card = engine.discover(df)
    assert isinstance(card, IntentDiscoveryCard)
    assert card.recommended_target == "target"
    assert card.problem_type == ProblemType.BINARY_CLASSIFICATION
    assert "Healthcare" in card.detected_domain or "Medicine" in card.detected_domain
    assert card.confidence_score > 0.60
    assert len(card.candidate_ranking) > 0
    text_report = card.format_text_report()
    assert "UAP 5.0 AUTONOMOUS INTENT" in text_report


def test_autonomous_intent_discovery_regression():
    engine = AutonomousIntentEngine()
    df = pd.read_csv("data/real_world/california_housing.csv").head(100)

    card = engine.discover(df)
    assert card.recommended_target == "MedHouseVal"
    assert card.problem_type == ProblemType.REGRESSION
    assert "Real Estate" in card.detected_domain or "Macroeconomics" in card.detected_domain


def test_natural_language_goal_alignment():
    engine = AutonomousIntentEngine()
    # Mock dataframe with multiple candidates
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "monthly_spend": [100.5, 250.0, 75.2, 500.0],
        "fraud_score": [0.01, 0.95, 0.05, 0.88],
        "is_fraud": [0, 1, 0, 1],
        "churn": [0, 0, 1, 1],
    })

    # User explicitly asks to focus on fraud
    card_fraud = engine.discover(df, user_goal="detect fraudulent transactions")
    assert card_fraud.recommended_target == "is_fraud"

    # User explicitly asks to focus on churn
    card_churn = engine.discover(df, user_goal="minimize customer churn")
    assert card_churn.recommended_target == "churn"


def test_zero_config_training_without_target_specified():
    """
    Verifies that a user can simply call engine.fit(df) without knowing
    or passing target_column, and UAP automatically discovers and trains it.
    """
    df = pd.read_csv("data/real_world/breast_cancer.csv").sample(n=120, random_state=42)
    engine = UAP(n_trials=3, timeout_sec=10)

    # Calling fit with ZERO target arguments!
    engine.fit(df)

    assert engine.target_name == "target"
    assert engine.intent_card is not None
    assert engine.intent_card.recommended_target == "target"
    assert engine.ema is not None
    assert engine.ema.winning_model_name != ""


def test_living_decision_dossier_generation():
    """
    Verifies the Living Decision Dossier generation on an individual sample.
    """
    df = pd.read_csv("data/real_world/breast_cancer.csv").sample(n=120, random_state=42)
    engine = UAP(n_trials=3, timeout_sec=10)
    engine.fit(df, target_column="target")

    test_sample = df.drop(columns=["target"]).iloc[0].to_dict()
    dossier = engine.predict_dossier(test_sample)

    assert isinstance(dossier, LivingDecisionDossier)
    assert dossier.prediction in [0, 1]
    assert 0.0 <= dossier.confidence_score <= 1.0
    assert dossier.trust_badge in ["SAFE_TO_PREDICT", "ABSTAIN_HUMAN_REVIEW"]
    assert isinstance(dossier.top_causal_drivers, list)
    assert len(dossier.top_causal_drivers) > 0
    assert dossier.business_impact_summary != ""

    formatted_text = dossier.format_executive_dossier()
    assert "UAP 5.0 LIVING DECISION DOSSIER" in formatted_text
    assert "WHY DID THIS HAPPEN?" in formatted_text
    assert "ACTIONABLE RECOURSE" in formatted_text


def test_api_auto_discover_and_dossier():
    from fastapi.testclient import TestClient
    import api

    df = pd.read_csv("data/real_world/breast_cancer.csv").sample(n=30, random_state=42)
    records = df.to_dict(orient="records")

    client = TestClient(api.app)

    # 1. Test POST /auto-discover
    res_disc = client.post("/auto-discover", json={"records": records})
    assert res_disc.status_code == 200
    disc_data = res_disc.json()
    assert disc_data["recommended_target"] == "target"
    assert "Healthcare" in disc_data["detected_domain"] or "Medicine" in disc_data["detected_domain"]

    # 2. Test POST /dossier
    engine = UAP(n_trials=2, timeout_sec=5)
    engine.fit(df, target_column="target")
    api.GLOBAL_ENGINE = engine

    sample_feats = df.drop(columns=["target"]).iloc[0].to_dict()
    res_dossier = client.post("/dossier", json={"features": sample_feats})
    assert res_dossier.status_code == 200
    doss_data = res_dossier.json()
    assert doss_data["trust_badge"] in ["SAFE_TO_PREDICT", "ABSTAIN_HUMAN_REVIEW"]
    assert "formatted_executive_dossier" in doss_data
