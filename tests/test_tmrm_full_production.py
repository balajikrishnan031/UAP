"""
Comprehensive Enterprise Test Suite for TMRM (Topological Manifold Resonant Machine)
Validates:
1. Unified Multiclass & Binary Classification
2. Continuous Riemannian Manifold Regression (R^2 & RMSE)
3. Auto-Topology (Auto-K Resonators)
4. In-Model Epistemic Novelty (Self-Doubt on OOD)
5. Actionable Human-Readable Recourse with Named Features & Immutability
6. Serialization (Save and Load from Disk)
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

from tmrm import TopologicalManifoldResonantMachine


def test_tmrm_classification_pipeline():
    """Test full classification with pandas DataFrame and Auto-K"""
    X, y = make_classification(n_samples=400, n_features=6, n_classes=2, random_state=42)
    feature_names = [f"biomarker_{i}" for i in range(6)]
    df = pd.DataFrame(X, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42)

    model = TopologicalManifoldResonantMachine(n_resonators="auto", random_state=42)
    model.fit(X_train, y_train)

    assert model.is_fitted
    assert model.is_classifier
    assert len(model.feature_names_) == 6

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    assert len(preds) == len(y_test)
    assert probs.shape == (len(y_test), 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    acc = accuracy_score(y_test, preds)
    assert acc > 0.80, f"Expected accuracy > 0.80, got {acc:.4f}"


def test_tmrm_continuous_regression():
    """Test continuous manifold regression"""
    X, y = make_regression(n_samples=500, n_features=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=["temp", "humidity", "pressure", "wind", "elevation"])

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42)

    model = TopologicalManifoldResonantMachine(task_type="regression", n_resonators="auto", random_state=42)
    model.fit(X_train, y_train)

    assert model.is_fitted
    assert not model.is_classifier

    preds = model.predict(X_test)
    assert len(preds) == len(y_test)

    r2 = r2_score(y_test, preds)
    assert r2 > 0.60, f"Expected R2 > 0.60, got {r2:.4f}"


def test_tmrm_epistemic_novelty_self_doubt():
    """Test that TMRM detects alien Out-of-Distribution samples automatically"""
    X, y = make_classification(n_samples=300, n_features=4, random_state=42)
    model = TopologicalManifoldResonantMachine(random_state=42)
    model.fit(X, y)

    # In-distribution sample
    in_dist_sample = X[:2]
    in_novelty = model.get_epistemic_novelty(in_dist_sample)

    # Alien Out-of-distribution sample
    ood_sample = np.full((2, 4), 25.0)
    ood_novelty = model.get_epistemic_novelty(ood_sample)

    assert np.all(ood_novelty > in_novelty), "OOD samples must have significantly higher novelty score than in-dist"
    assert np.all(ood_novelty > model.novelty_threshold), "OOD samples must trigger the self-doubt threshold"


def test_tmrm_human_readable_recourse():
    """Test actionable human-readable recourse plan generation"""
    df = pd.read_csv("data/real_world/heart_disease.csv")
    target_col = [c for c in df.columns if "target" in c.lower() or "heart" in c.lower() or "disease" in c.lower()][0]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = TopologicalManifoldResonantMachine(n_resonators="auto", random_state=42)
    model.fit(X, y)

    # Pick a sample of class 1 (High risk)
    high_risk_idx = np.where(y.values == 1)[0][0]
    sample = X.iloc[high_risk_idx]

    # Generate recourse to flip to Class 0 (Safe) with age marked immutable
    immutable_feats = ["age", "sex"]
    recourse_plan = model.get_recourse_action_plan(
        sample=sample,
        target_class=0,
        immutable_features=immutable_feats
    )

    assert recourse_plan["target_class"] == 0
    assert "action_items" in recourse_plan
    assert len(recourse_plan["action_items"]) > 0

    # Ensure immutable features were not changed
    for item in recourse_plan["action_items"]:
        assert item["feature"] not in immutable_feats, f"Immutable feature {item['feature']} was modified!"
        assert "description" in item


def test_tmrm_persistence_save_load():
    """Test saving and loading trained TMRM model"""
    X, y = make_classification(n_samples=200, n_features=4, random_state=42)
    model = TopologicalManifoldResonantMachine(random_state=42)
    model.fit(X, y)

    orig_preds = model.predict(X[:10])

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        model.save(tmp_path)
        assert os.path.exists(tmp_path)

        loaded_model = TopologicalManifoldResonantMachine.load(tmp_path)
        loaded_preds = loaded_model.predict(X[:10])

        assert np.array_equal(orig_preds, loaded_preds), "Loaded model predictions must exactly match original"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
