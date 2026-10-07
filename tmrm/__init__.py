"""
TMRM (Topological Manifold Resonant Machine)
============================================
A novel ground-up machine learning algorithm based on continuous
Riemannian energy manifolds and multi-octave wavelet resonance.

Quickstart:
    >>> from tmrm import TMRM
    >>> model = TMRM()
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
"""

__version__ = "4.4.0"
__author__ = "Balaji P, Navaneetham V, Dhavan RG"

from .core import (
    TMRM,
    TopologicalManifoldResonantMachine,
    StreamingTMRM
)

__all__ = [
    "TMRM",
    "TopologicalManifoldResonantMachine",
    "StreamingTMRM",
    "__version__"
]
