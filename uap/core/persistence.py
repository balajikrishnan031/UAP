"""
UAP 2.0 Model Persistence Engine.
Serializes and restores complete fitted production UAP engines:
  - Pipeline weights and transformers
  - Winning model architecture / Stacking ensemble
  - Abstention Gate manifold parameters (empirical mean & precision matrix)
  - Feasible Counterfactual bounds and MAD normalization dictionaries
"""

from pathlib import Path
from typing import Any
import joblib


class ModelPersistenceManager:
    """
    Handles saving and restoring trained UAP 2.0 artifacts to disk.
    """

    @staticmethod
    def save_engine(engine: Any, filepath: str) -> str:
        """
        Serializes a fitted UAPEngine instance to the target path.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(engine, path, compress=3)
        print(f"[Persistence] Successfully saved trained UAP model -> {path}")
        return str(path)

    @staticmethod
    def load_engine(filepath: str) -> Any:
        """
        Loads and restores a serialized UAPEngine from disk.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")
        engine = joblib.load(path)
        print(f"[Persistence] Successfully loaded UAP model from -> {path}")
        return engine
