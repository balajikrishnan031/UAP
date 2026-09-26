"""
Module 4 - Part 4: Real-Time Drift & Continuous Monitoring Engine.
Computes feature-level Population Stability Index (PSI) to detect data drift in production streams.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd


class DriftMonitor:
    """
    Monitors inference data distributions against baseline training distributions using PSI.
    """

    def __init__(self, baseline_df: pd.DataFrame, n_bins: int = 10):
        self.baseline_df = baseline_df.copy()
        self.n_bins = n_bins
        self.feature_bins: Dict[str, np.ndarray] = {}
        self.feature_baseline_dist: Dict[str, np.ndarray] = {}
        self._initialize_baseline()

    def _initialize_baseline(self):
        for col in self.baseline_df.select_dtypes(include=[np.number]).columns:
            vals = self.baseline_df[col].dropna()
            if len(vals) < 10 or vals.nunique() < 3:
                continue

            quantiles = np.linspace(0, 1, self.n_bins + 1)
            bins = np.percentile(vals, quantiles * 100)
            bins[0] = -np.inf
            bins[-1] = np.inf
            bins = np.unique(bins)

            if len(bins) > 2:
                self.feature_bins[col] = bins
                counts, _ = np.histogram(vals, bins=bins)
                probs = counts / len(vals)
                # Apply small epsilon smoothing to avoid log(0)
                probs = np.where(probs == 0, 1e-4, probs)
                self.feature_baseline_dist[col] = probs / np.sum(probs)

    def calculate_psi(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates PSI for each monitored numerical feature.
        """
        feature_psis = {}
        alert_features = []

        for col, bins in self.feature_bins.items():
            if col not in current_df.columns:
                continue

            vals = current_df[col].dropna()
            if len(vals) == 0:
                continue

            counts, _ = np.histogram(vals, bins=bins)
            probs = counts / len(vals)
            probs = np.where(probs == 0, 1e-4, probs)
            probs = probs / np.sum(probs)

            base_probs = self.feature_baseline_dist[col]

            # Length alignment if needed
            min_len = min(len(probs), len(base_probs))
            p = base_probs[:min_len]
            q = probs[:min_len]

            psi_val = np.sum((q - p) * np.log(q / p))
            psi_val = max(0.0, float(psi_val))
            feature_psis[col] = round(psi_val, 4)

            if psi_val > 0.25:
                alert_features.append(col)

        overall_psi = float(np.mean(list(feature_psis.values()))) if feature_psis else 0.0

        if overall_psi < 0.10:
            status = "STABLE"
        elif overall_psi <= 0.25:
            status = "MODERATE_DRIFT_WARNING"
        else:
            status = "CRITICAL_DRIFT_RETRAIN_ALERT"

        return {
            "overall_psi": round(overall_psi, 4),
            "status": status,
            "alert_features_count": len(alert_features),
            "alert_features": alert_features,
            "feature_psis": feature_psis,
            "action_required": overall_psi > 0.25,
        }
