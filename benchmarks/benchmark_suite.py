"""
UAP Benchmark Suite: Evaluates UAP Framework against XGBoost Baselines and Ensembles.
Measures Predictive Performance, Robustness Decay Score (RDS), and Convergence Time.
"""

from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, root_mean_squared_error, r2_score
import xgboost as xgb

from uap.engine import UAPEngine
from synthetic_data.generator import UAPSyntheticDataGenerator


class BenchmarkSuite:
    """
    Standardized benchmark harness comparing UAP vs XGBoost Baselines.
    """

    @staticmethod
    def run_classification_benchmark(n_samples: int = 2500) -> pd.DataFrame:
        print("\n=======================================================")
        print("RUNNING BENCHMARK 1: Imbalanced Non-Linear Classification")
        print("=======================================================")
        gen = UAPSyntheticDataGenerator()
        df = gen.generate_imbalanced_non_linear_classification(
            n_samples=n_samples, minority_ratio=0.04, label_noise=0.03
        )

        train_df, test_df = train_test_split(
            df, test_size=0.20, random_state=42, stratify=df["target"]
        )
        X_train = train_df.drop(columns=["target"])
        y_train = train_df["target"]
        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]

        # Corrupted test set for RDS evaluation (15% Gaussian noise)
        X_test_corr = X_test.copy()
        X_test_corr += np.random.normal(0, X_test.std() * 0.15, size=X_test.shape)

        results = []

        # System 1: Standard XGBoost Baseline (Untuned)
        t0 = time.time()
        xgb_base = xgb.XGBClassifier(random_state=42, eval_metric="logloss")
        xgb_base.fit(X_train, y_train)
        t_base = time.time() - t0

        base_clean = f1_score(y_test, xgb_base.predict(X_test), average="macro")
        base_corr = f1_score(y_test, xgb_base.predict(X_test_corr), average="macro")
        base_rds = max(0.0, (base_clean - base_corr) / max(1e-4, base_clean))

        results.append({
            "System": "XGBoost (Standard Baseline)",
            "Clean Macro F1": round(base_clean, 4),
            "Corrupted Macro F1": round(base_corr, 4),
            "Robustness Decay (RDS)": round(base_rds, 4),
            "Fit Time (s)": round(t_base, 2),
        })

        # System 2: UAP Framework
        t0 = time.time()
        uap = UAPEngine(n_trials=8, timeout_sec=25)
        uap.fit(train_df, target_column="target")
        t_uap = time.time() - t0

        uap_clean = f1_score(y_test, uap.predict(X_test), average="macro")
        uap_corr = f1_score(y_test, uap.predict(X_test_corr), average="macro")
        uap_rds = max(0.0, (uap_clean - uap_corr) / max(1e-4, uap_clean))

        results.append({
            "System": "UAP Engine (Adaptive Layer)",
            "Clean Macro F1": round(uap_clean, 4),
            "Corrupted Macro F1": round(uap_corr, 4),
            "Robustness Decay (RDS)": round(uap_rds, 4),
            "Fit Time (s)": round(t_uap, 2),
        })

        res_df = pd.DataFrame(results)
        print("\nBenchmark 1 Results:")
        print(res_df.to_string(index=False))
        return res_df

    @staticmethod
    def run_regression_benchmark(n_samples: int = 2500) -> pd.DataFrame:
        print("\n=======================================================")
        print("RUNNING BENCHMARK 2: Outlier-Corrupted Non-Linear Regression")
        print("=======================================================")
        gen = UAPSyntheticDataGenerator()
        df = gen.generate_noisy_outlier_regression(
            n_samples=n_samples, outlier_ratio=0.10
        )

        train_df, test_df = train_test_split(
            df, test_size=0.20, random_state=42
        )
        X_train = train_df.drop(columns=["target"])
        y_train = train_df["target"]
        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]

        X_test_corr = X_test.copy()
        X_test_corr += np.random.normal(0, X_test.std() * 0.15, size=X_test.shape)

        results = []

        # System 1: Standard XGBoost Baseline
        t0 = time.time()
        xgb_reg = xgb.XGBRegressor(random_state=42)
        xgb_reg.fit(X_train, y_train)
        t_base = time.time() - t0

        base_clean_r2 = r2_score(y_test, xgb_reg.predict(X_test))
        base_clean_rmse = root_mean_squared_error(y_test, xgb_reg.predict(X_test))
        base_corr_rmse = root_mean_squared_error(y_test, xgb_reg.predict(X_test_corr))
        base_rds = max(0.0, (base_corr_rmse - base_clean_rmse) / max(1e-4, base_corr_rmse))

        results.append({
            "System": "XGBoost (Standard Regressor)",
            "Clean R2": round(base_clean_r2, 4),
            "Clean RMSE": round(base_clean_rmse, 4),
            "Corrupted RMSE": round(base_corr_rmse, 4),
            "Robustness Decay (RDS)": round(base_rds, 4),
            "Fit Time (s)": round(t_base, 2),
        })

        # System 2: UAP Framework
        t0 = time.time()
        uap = UAPEngine(n_trials=8, timeout_sec=25)
        uap.fit(train_df, target_column="target")
        t_uap = time.time() - t0

        uap_clean_r2 = r2_score(y_test, uap.predict(X_test))
        uap_clean_rmse = root_mean_squared_error(y_test, uap.predict(X_test))
        uap_corr_rmse = root_mean_squared_error(y_test, uap.predict(X_test_corr))
        uap_rds = max(0.0, (uap_corr_rmse - uap_clean_rmse) / max(1e-4, uap_corr_rmse))

        results.append({
            "System": "UAP Engine (Adaptive Layer)",
            "Clean R2": round(uap_clean_r2, 4),
            "Clean RMSE": round(uap_clean_rmse, 4),
            "Corrupted RMSE": round(uap_corr_rmse, 4),
            "Robustness Decay (RDS)": round(uap_rds, 4),
            "Fit Time (s)": round(t_uap, 2),
        })

        res_df = pd.DataFrame(results)
        print("\nBenchmark 2 Results:")
        print(res_df.to_string(index=False))
        return res_df


if __name__ == "__main__":
    suite = BenchmarkSuite()
    suite.run_classification_benchmark(n_samples=2000)
    suite.run_regression_benchmark(n_samples=2000)
