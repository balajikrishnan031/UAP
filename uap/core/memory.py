"""
Module: Meta-Learning Strategy Memory (MLS-Memory).
Stores dataset topological signatures (DCVs) and associated winning algorithmic pipelines.
Provides zero-shot/few-shot strategy retrieval using high-dimensional cosine similarity matching.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import numpy as np

from uap.core.contracts import (
    DatasetCharacteristicVector,
    ModelStrategyConfiguration,
    ProblemType,
)


class MetaLearningMemory:
    """
    Experience-driven Meta-Learning Memory Bank.
    Caches historical training journeys and retrieves prior optimal strategies
    when an incoming dataset exhibits topological similarity to known distributions.
    """

    def __init__(self, memory_file: Optional[str] = None):
        self.memory_file = Path(memory_file) if memory_file else None
        self.records: List[Dict[str, Any]] = []
        if self.memory_file and self.memory_file.exists():
            self.load(str(self.memory_file))

    @staticmethod
    def extract_signature_vector(dcv: DatasetCharacteristicVector) -> np.ndarray:
        """
        Converts a DatasetCharacteristicVector into a continuous, scale-normalized signature vector.
        Features:
          0: log10(samples)
          1: total_features / 100.0
          2: min(imbalance_ratio, 20.0) / 20.0
          3: min(dimensionality_ratio, 1.0)
          4: min(outlier_severity_index, 1.0)
          5: min(non_linearity_index, 1.0)
          6: min(high_correlation_pairs, 20) / 20.0
          7: 1.0 if classification else 0.0
        """
        summary = dcv.dataset_summary
        health = dcv.data_health_scores
        is_classif = 1.0 if summary.problem_type in [
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ] else 0.0

        vec = np.array([
            math.log10(max(10, summary.total_samples)),
            summary.total_features / 100.0,
            min(20.0, max(1.0, health.imbalance_ratio)) / 20.0,
            min(1.0, max(0.0, health.dimensionality_ratio)),
            min(1.0, max(0.0, health.outlier_severity_index)),
            min(1.0, max(0.0, health.non_linearity_index)),
            min(20, max(0, health.high_correlation_pairs)) / 20.0,
            is_classif,
        ], dtype=np.float32)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def log_experience(
        self,
        dcv: DatasetCharacteristicVector,
        winning_model_name: str,
        best_hyperparameters: Dict[str, Any],
        validation_score: float,
        holdout_score: Optional[float] = None,
        domain: str = "General",
        dataset_name: str = "Unknown",
        selected_models: Optional[List[str]] = None,
        scaling_strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Registers a completed training journey into long-term meta-memory.
        """
        vec = self.extract_signature_vector(dcv).tolist()
        record = {
            "domain": domain,
            "dataset_name": dataset_name,
            "problem_type": dcv.dataset_summary.problem_type.value,
            "samples": dcv.dataset_summary.total_samples,
            "features": dcv.dataset_summary.total_features,
            "signature_vector": vec,
            "winning_model_name": winning_model_name,
            "selected_models": selected_models or [winning_model_name],
            "scaling_strategy": scaling_strategy or "StandardScaler",
            "best_hyperparameters": best_hyperparameters,
            "validation_score": float(validation_score),
            "holdout_score": float(holdout_score) if holdout_score is not None else None,
        }

        # Deduplicate by dataset_name if existing
        self.records = [r for r in self.records if r.get("dataset_name") != dataset_name]
        self.records.append(record)

        if self.memory_file:
            self.save(str(self.memory_file))

        return record

    def find_closest_strategy(
        self, dcv: DatasetCharacteristicVector, similarity_threshold: float = 0.90
    ) -> Optional[Dict[str, Any]]:
        """
        Calculates cosine similarity across memory bank and returns the closest strategy
        if similarity exceeds the threshold.
        """
        if not self.records:
            return None

        query_vec = self.extract_signature_vector(dcv)
        query_is_classif = 1.0 if dcv.dataset_summary.problem_type in [
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ] else 0.0

        best_sim = -1.0
        best_record = None

        for rec in self.records:
            # Enforce same problem category (classification vs regression)
            rec_is_classif = 1.0 if "Classification" in rec["problem_type"] else 0.0
            if query_is_classif != rec_is_classif:
                continue

            rec_vec = np.array(rec["signature_vector"], dtype=np.float32)
            # Dot product since vectors are pre-normalized
            sim = float(np.dot(query_vec, rec_vec))

            if sim > best_sim:
                best_sim = sim
                best_record = rec

        if best_record and best_sim >= similarity_threshold:
            return {
                "similarity": round(best_sim, 4),
                "matched_domain": best_record["domain"],
                "matched_dataset": best_record["dataset_name"],
                "recommended_model": best_record["winning_model_name"],
                "recommended_candidate_pool": best_record["selected_models"],
                "recommended_scaling": best_record["scaling_strategy"],
                "warm_start_hyperparameters": best_record["best_hyperparameters"],
                "historical_score": best_record["validation_score"],
            }

        return None

    def save(self, filepath: str) -> str:
        """Persists meta-memory to a JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)
        return str(p)

    def load(self, filepath: str) -> "MetaLearningMemory":
        """Loads meta-memory from a JSON file."""
        p = Path(filepath)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                self.records = json.load(f)
        return self

    def __len__(self) -> int:
        return len(self.records)
