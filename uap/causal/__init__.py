"""
UAP 3.0 Causal Intelligence Subsystem.
Discovers Structural Causal Models (SCMs), extracts causal Directed Acyclic Graphs (DAGs),
and executes Pearl's do-calculus interventional simulation.
"""

from uap.causal.engine import CausalDiscoveryEngine

__all__ = ["CausalDiscoveryEngine"]
