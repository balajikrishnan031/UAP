"""
UAP 2.0 Production MLOps Serving Microservice.
Built with FastAPI for low-latency (<10ms) inference, selective prediction (Abstention Gate),
feasible actionable counterfactual recourse (FACE), and real-time data drift monitoring.
"""

from typing import Any, Dict, List, Optional
import time
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from uap.engine import UAPEngine
from uap.core.system_telemetry import SystemTelemetryManager
from synthetic_data.generator import UAPSyntheticDataGenerator

app = FastAPI(
    title="UAP 2.0 Prediction Intelligence API",
    description="Production REST API for Universal Adaptive Prediction with Abstention Gate & Feasible Recourse.",
    version="2.0.0",
)

# Global in-memory engine instance
GLOBAL_ENGINE: Optional[UAPEngine] = None
START_TIME = time.time()


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        example={"feature_1": 0.5, "feature_2": -1.2, "feature_3": 0.8},
        description="Feature key-value pairs for the single inference record."
    )


class RecourseRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Current unfavorable record features.")
    target_outcome: int = Field(default=1, description="Desired flipped target class.")
    immutable_features: Optional[List[str]] = Field(
        default_factory=list,
        example=["feature_1"],
        description="Features that cannot be modified (e.g. Age, History)."
    )
    direction_constraints: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        example={"feature_3": "positive_only"},
        description="Allowable directional shifts per feature."
    )


class DriftCheckRequest(BaseModel):
    batch: List[Dict[str, float]] = Field(..., description="List of production streaming feature records.")


@app.on_event("startup")
def startup_event():
    """
    Initializes and warms up the UAP engine with the real-world production model on startup.
    """
    global GLOBAL_ENGINE
    from pathlib import Path
    model_path = Path("models/heart_disease_uap.joblib")
    if model_path.exists():
        print(f"[API Startup] Loading real-world Cardiology Model from {model_path}...")
        GLOBAL_ENGINE = UAPEngine.load(str(model_path))
    else:
        print("[API Startup] Real-world model not found on disk. Training from real heart disease dataset...")
        from data.dataset_loader import fetch_heart_disease
        df = fetch_heart_disease()
        engine = UAPEngine(n_trials=6, timeout_sec=20)
        engine.fit(df, target_column="target")
        engine.save(str(model_path))
        GLOBAL_ENGINE = engine
    print(f"[API Startup] UAP Production Engine ready! Features: {GLOBAL_ENGINE.feature_names}")


@app.get("/health", tags=["Monitoring"])
def get_health() -> Dict[str, Any]:
    """
    Returns API health, uptime, and host hardware telemetry for Green AI resource management.
    """
    telemetry = SystemTelemetryManager.get_hardware_profile()
    return {
        "status": "HEALTHY",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "engine_loaded": GLOBAL_ENGINE is not None,
        "hardware_telemetry": {
            "total_ram_gb": telemetry.total_ram_gb,
            "available_ram_gb": telemetry.available_ram_gb,
            "ram_usage_percent": telemetry.ram_usage_percent,
            "cpu_usage_percent": telemetry.cpu_usage_percent,
            "recommended_strategy_tier": telemetry.recommended_strategy_tier,
        }
    }


def to_serializable(val: Any) -> Any:
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    if isinstance(val, list):
        return [to_serializable(v) for v in val]
    return val


@app.post("/predict", tags=["Inference"])
def predict(request: PredictionRequest) -> Dict[str, Any]:
    """
    Executes safe selective prediction through the Abstention Gate G(x).
    Returns PREDICT, PREDICT_WITH_WARNING, or ABSTAIN.
    """
    if GLOBAL_ENGINE is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")

    sample = pd.Series(request.features)
    t0 = time.perf_counter()
    eval_result = GLOBAL_ENGINE.predict_safe(sample)
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    return to_serializable({
        "latency_ms": latency_ms,
        "gate_status": eval_result["status"],
        "gate_decision": eval_result["gate_decision"],
        "prediction": eval_result["prediction"],
        "confidence_score": eval_result["confidence_score"],
        "ood_distance": eval_result["ood_distance"],
        "human_action": eval_result["human_action"],
        "reasons": eval_result["reasons"],
    })


@app.post("/recourse", tags=["Prescriptive Recourse"])
def generate_recourse(request: RecourseRequest) -> Dict[str, Any]:
    """
    Generates actionable, bounded recourse plans using the Feasible Counterfactual Engine (FACE).
    """
    if GLOBAL_ENGINE is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")

    sample = pd.Series(request.features)
    t0 = time.perf_counter()
    recourse = GLOBAL_ENGINE.generate_feasible_recourse(
        sample=sample,
        target_outcome=request.target_outcome,
        immutable_features=request.immutable_features,
        direction_constraints=request.direction_constraints,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    return to_serializable({
        "latency_ms": latency_ms,
        "recourse_status": recourse["status"],
        "current_prediction": recourse.get("current_prediction"),
        "target_prediction": recourse.get("target_prediction"),
        "total_mad_cost": recourse.get("total_mad_cost"),
        "actionable_plan": recourse.get("minimal_action_plan", []),
        "immutable_features_preserved": recourse.get("immutable_features_preserved", []),
        "message": recourse.get("message", ""),
    })


@app.post("/drift", tags=["Monitoring"])
def check_drift(request: DriftCheckRequest) -> Dict[str, Any]:
    """
    Monitors live feature drift via Population Stability Index (PSI).
    """
    if GLOBAL_ENGINE is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")

    batch_df = pd.DataFrame(request.batch)
    if len(batch_df) < 10:
        raise HTTPException(status_code=400, detail="Batch size must contain at least 10 records.")

    drift_report = GLOBAL_ENGINE.monitor_drift(batch_df)
    return drift_report


@app.post("/reliability-card", tags=["UAP 4.0 Reliability"])
def get_prediction_reliability_card(request: PredictionRequest) -> Dict[str, Any]:
    """
    UAP 4.0 Core Trustworthiness Endpoint:
    Returns the certified Prediction Reliability Card evaluating confidence,
    95% conformal bounds, data quality, OOD novelty risk, drift risk, and decision.
    """
    if GLOBAL_ENGINE is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")

    sample = pd.Series(request.features)
    t0 = time.perf_counter()
    card = GLOBAL_ENGINE.predict_with_reliability_card(sample)
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    res = card.to_dict()
    res["latency_ms"] = latency_ms
    res["formatted_text_card"] = card.format_text_card()
    return to_serializable(res)


@app.get("/explain-strategy", tags=["UAP 4.0 Strategy Audit"])
def explain_strategy() -> Dict[str, Any]:
    """
    Answers: 'Why was this specific model and configuration chosen?'
    Returns goal-aware metric selection, risk posture, and trade-offs.
    """
    if GLOBAL_ENGINE is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")

    return to_serializable(GLOBAL_ENGINE.explain_strategy())


class AutoDiscoverRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="Sample dataset rows to infer intent and targets from.")
    user_goal: Optional[str] = Field(default=None, description="Optional natural language goal prompt.")


@app.post("/auto-discover", tags=["UAP 5.0 Autonomous Intent"])
def auto_discover_intent(request: AutoDiscoverRequest) -> Dict[str, Any]:
    """
    Zero-config intent discovery: Automatically discovers domain, target column,
    and business objective from raw records without user guessing.
    """
    from uap.module1_profiler.autonomous_intent import AutonomousIntentEngine
    df = pd.DataFrame(request.records)
    engine = AutonomousIntentEngine()
    card = engine.discover(df, user_goal=request.user_goal)
    res = card.to_dict()
    res["formatted_report"] = card.format_text_report()
    return to_serializable(res)


class DossierRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature key-value pairs for the single inference record.")
    target_goal: Optional[int] = Field(default=None, description="Desired target outcome for recourse optimization.")
    immutable_features: Optional[List[str]] = Field(default_factory=list, description="Features that cannot be changed.")


@app.post("/dossier", tags=["UAP 5.0 Living Decision Dossier"])
def get_living_decision_dossier(request: DossierRequest) -> Dict[str, Any]:
    """
    UAP 5.0 Core Output Artifact: The Living Decision Dossier.
    Synthesizes prediction, 95% conformal bounds, trust certification,
    causal driver attribution, and actionable recourse.
    """
    if GLOBAL_ENGINE is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")

    t0 = time.perf_counter()
    dossier = GLOBAL_ENGINE.predict_dossier(
        sample=request.features,
        target_goal=request.target_goal,
        immutable_features=request.immutable_features,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    res = dossier.to_dict()
    res["latency_ms"] = latency_ms
    res["formatted_executive_dossier"] = dossier.format_executive_dossier()
    return to_serializable(res)

