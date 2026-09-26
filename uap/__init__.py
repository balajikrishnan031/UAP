"""
Universal Adaptive Prediction (UAP) Framework
A closed-loop, data-driven adaptive machine learning framework.
"""

from uap.core.contracts import (
    DatasetCharacteristicVector,
    ModelStrategyConfiguration,
    EvaluatedModelArtifact,
    ProblemType,
    LivingDecisionDossier,
    PredictionReliabilityCard,
)
from uap.module1_profiler.autonomous_intent import IntentDiscoveryCard
from uap.models.novel_tmrm import TopologicalManifoldResonantMachine
from uap.models.streaming_tmrm import StreamingTMRM
from uap.engine import UAPEngine

# Ultra-ergonomic developer aliases
UAP = UAPEngine
TMRM = TopologicalManifoldResonantMachine

__all__ = [
    "DatasetCharacteristicVector",
    "ModelStrategyConfiguration",
    "EvaluatedModelArtifact",
    "ProblemType",
    "PredictionReliabilityCard",
    "LivingDecisionDossier",
    "IntentDiscoveryCard",
    "TopologicalManifoldResonantMachine",
    "StreamingTMRM",
    "TMRM",
    "UAPEngine",
    "UAP",
]

