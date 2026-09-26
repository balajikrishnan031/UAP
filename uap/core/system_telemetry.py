"""
UAP 2.0 Green AI: Dynamic Resource Awareness & Hardware Telemetry.
Probes host hardware (RAM, CPU cores, Memory Pressure) to dynamically calibrate
model selection, ensemble depth, and batch sizes for edge and resource-constrained devices.
"""

from dataclasses import dataclass
from typing import Dict, Any
import os
import psutil


@dataclass
class HardwareProfile:
    total_ram_gb: float
    available_ram_gb: float
    ram_usage_percent: float
    cpu_cores_logical: int
    cpu_cores_physical: int
    cpu_usage_percent: float
    is_resource_constrained: bool
    recommended_strategy_tier: str  # "LITE" (Edge/Low RAM), "STANDARD", "HIGH_PERFORMANCE"


class SystemTelemetryManager:
    """
    Monitors host hardware constraints to dynamically steer AutoML resource consumption.
    """

    @staticmethod
    def get_hardware_profile() -> HardwareProfile:
        vmem = psutil.virtual_memory()
        total_ram_gb = round(vmem.total / (1024 ** 3), 2)
        avail_ram_gb = round(vmem.available / (1024 ** 3), 2)
        ram_percent = vmem.percent

        logical_cpus = os.cpu_count() or 2
        try:
            physical_cpus = psutil.cpu_count(logical=False) or logical_cpus
        except Exception:
            physical_cpus = logical_cpus

        cpu_percent = psutil.cpu_percent(interval=None)

        # Classify resource tier
        # If available RAM < 2.0 GB or RAM usage > 85%, flag as constrained
        if avail_ram_gb < 2.5 or ram_percent > 85.0 or logical_cpus <= 2:
            tier = "LITE"
            is_constrained = True
        elif avail_ram_gb < 6.0 or ram_percent > 70.0:
            tier = "STANDARD"
            is_constrained = False
        else:
            tier = "HIGH_PERFORMANCE"
            is_constrained = False

        return HardwareProfile(
            total_ram_gb=total_ram_gb,
            available_ram_gb=avail_ram_gb,
            ram_usage_percent=ram_percent,
            cpu_cores_logical=logical_cpus,
            cpu_cores_physical=physical_cpus,
            cpu_usage_percent=cpu_percent,
            is_resource_constrained=is_constrained,
            recommended_strategy_tier=tier,
        )

    @staticmethod
    def calibrate_hyperparameter_bounds(profile: HardwareProfile) -> Dict[str, Any]:
        """
        Dynamically throttles n_estimators, tree depth, and parallel jobs based on memory pressure.
        """
        if profile.recommended_strategy_tier == "LITE":
            return {
                "max_estimators": 60,
                "max_depth": 5,
                "n_trials": 5,
                "enable_heavy_stacking": False,
                "n_jobs": 1,
                "batch_size": 64,
            }
        elif profile.recommended_strategy_tier == "STANDARD":
            return {
                "max_estimators": 120,
                "max_depth": 8,
                "n_trials": 10,
                "enable_heavy_stacking": True,
                "n_jobs": max(1, profile.cpu_cores_logical // 2),
                "batch_size": 128,
            }
        else:
            return {
                "max_estimators": 250,
                "max_depth": 12,
                "n_trials": 20,
                "enable_heavy_stacking": True,
                "n_jobs": max(1, profile.cpu_cores_logical - 1),
                "batch_size": 256,
            }
