"""
TMRM (Topological Manifold Resonant Machine)
============================================
A novel ground-up machine learning algorithm based on continuous
Riemannian energy manifolds and multi-octave wavelet resonance.

Quickstart:
    >>> from tmrm import TMRM, load_model
    >>> model = TMRM()
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
"""

__version__ = "1.0.0"
__author__ = "Antigravity AI Team"

from .core import (
    TMRM,
    TopologicalManifoldResonantMachine,
    StreamingTMRM,
    load_model,
    predict_heart_disease
)

__all__ = [
    "TMRM",
    "TopologicalManifoldResonantMachine",
    "StreamingTMRM",
    "load_model",
    "predict_heart_disease",
    "__version__"
]
