"""
Automated Unit Tests for UAP Framework Components.
Tests Module 1 (Profiler & DCV), Module 2 (AFWE & DMRE), Module 3 (Stress-Tester), and Module 4 (Drift & XAI).
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from uap.core.contracts import ProblemType
from uap.module1_profiler.profiler import DatasetProfiler
from uap.module2_weighting_routing.afwe import AdaptiveFeatureWeightingEngine
from uap.module2_weighting_routing.dmre import DynamicModelRoutingEngine
from uap.module4_xai_serving.drift import DriftMonitor
from synthetic_data.generator import UAPSyntheticDataGenerator


class TestUAPFramework(unittest.TestCase):

    def setUp(self):
        self.gen = UAPSyntheticDataGenerator()

    def test_module1_profiling(self):
        # Generate classification dataset with 4% minority
        df = self.gen.generate_imbalanced_non_linear_classification(
            n_samples=500, n_features=6, minority_ratio=0.04
        )
        profiler = DatasetProfiler()
        dcv, X, y = profiler.profile(df, target_column="target")

        self.assertEqual(dcv.dataset_summary.problem_type, ProblemType.BINARY_CLASSIFICATION)
        self.assertGreater(dcv.data_health_scores.imbalance_ratio, 4.0)
        self.assertEqual(X.shape[1], 6)
        self.assertEqual(len(y), 500)

    def test_module2_afwe_feature_weighting(self):
        # Generate high-dimensional noisy data (5 info + 15 noise)
        df = self.gen.generate_high_dimensional_noisy(
            n_samples=400, n_informative=5, n_noise_features=15
        )
        X = df.drop(columns=["target"])
        y = df["target"]

        afwe = AdaptiveFeatureWeightingEngine()
        X_weighted, config = afwe.fit_transform(X, y, ProblemType.BINARY_CLASSIFICATION)

        # Informative features should have on average higher weights than pure noise features
        info_weights = [config.feature_weights[f"informative_{i+1}"] for i in range(5)]
        noise_weights = [config.feature_weights[f"noise_{i+1}"] for i in range(15)]

        self.assertGreater(np.mean(info_weights), np.mean(noise_weights))

    def test_module2_dmre_routing(self):
        # Generate outlier regression data
        df = self.gen.generate_noisy_outlier_regression(
            n_samples=500, outlier_ratio=0.12
        )
        profiler = DatasetProfiler()
        dcv, X, y = profiler.profile(df, target_column="target")

        afwe = AdaptiveFeatureWeightingEngine()
        _, f_config = afwe.fit_transform(X, y, dcv.dataset_summary.problem_type)

        dmre = DynamicModelRoutingEngine()
        msc = dmre.generate_strategy(dcv, f_config)

        # Due to high outlier severity, HuberRegressor or RobustScaler must be selected
        self.assertEqual(msc.routing_decision.scaling_strategy, "RobustScaler")
        self.assertIn("LinearRegression", msc.routing_decision.excluded_models)

    def test_module4_drift_monitoring(self):
        df = self.gen.generate_imbalanced_non_linear_classification(n_samples=500)
        X = df.drop(columns=["target"])

        monitor = DriftMonitor(baseline_df=X)

        # In-distribution test
        res_in = monitor.calculate_psi(X.sample(200, random_state=42))
        self.assertLess(res_in["overall_psi"], 0.25)
        self.assertEqual(res_in["status"], "STABLE")

        # Heavily shifted distribution test
        X_shifted = X.sample(200, random_state=42).copy()
        X_shifted += 5.0
        res_shifted = monitor.calculate_psi(X_shifted)
        self.assertGreater(res_shifted["overall_psi"], 0.25)
        self.assertEqual(res_shifted["status"], "CRITICAL_DRIFT_RETRAIN_ALERT")

    def test_abstention_gate(self):
        from uap.module4_xai_serving.abstention import AbstentionGate

        df = self.gen.generate_imbalanced_non_linear_classification(n_samples=500, n_features=6)
        X = df.drop(columns=["target"])

        gate = AbstentionGate(tau_ood=3.5, tau_conf=0.60)
        gate.fit(X)

        # 1. In-distribution normal sample
        normal_sample = X.iloc[0]
        res_normal = gate.evaluate_sample(
            sample=normal_sample,
            raw_pred=1,
            raw_prob=np.array([0.15, 0.85]),
        )
        self.assertIn(res_normal["status"], ["PREDICT", "PREDICT_WITH_WARNING"])
        self.assertEqual(res_normal["gate_decision"], 1)

        # 2. Extreme Out-of-Distribution sample
        ood_sample = normal_sample.copy() + 25.0
        res_ood = gate.evaluate_sample(
            sample=ood_sample,
            raw_pred=1,
            raw_prob=np.array([0.15, 0.85]),
        )
        self.assertEqual(res_ood["status"], "ABSTAIN")
        self.assertEqual(res_ood["gate_decision"], 0)
        self.assertEqual(res_ood["prediction"], "UNKNOWN")

    def test_feasible_counterfactual_engine(self):
        from uap.module4_xai_serving.counterfactual import FeasibleCounterfactualEngine

        # Synthetic threshold predictor: f(x) = 1 if x1 + x2 > 3.0 else 0
        def dummy_predictor(df_in: pd.DataFrame) -> np.ndarray:
            return (df_in["feature_1"] + df_in["feature_2"] > 3.0).astype(int).values

        X_train = pd.DataFrame({
            "feature_1": np.linspace(0, 5, 100),
            "feature_2": np.linspace(0, 5, 100),
        })

        face = FeasibleCounterfactualEngine(
            predictor_func=dummy_predictor,
            feature_names=["feature_1", "feature_2"],
            immutable_features=["feature_1"],  # Frozen!
        ).fit_bounds(X_train)

        # Start with unfavorable state f(x) = 0 (sum = 1.0)
        unfavorable_sample = pd.Series({"feature_1": 0.5, "feature_2": 0.5})
        recourse = face.generate_recourse(unfavorable_sample, target_outcome=1)

        self.assertEqual(recourse["status"], "RECOURSE_FOUND")
        self.assertEqual(recourse["target_prediction"], 1)
        # feature_1 must be preserved and only feature_2 modified
        changed_features = [act["feature"] for act in recourse["minimal_action_plan"]]
        self.assertNotIn("feature_1", changed_features)
        self.assertIn("feature_2", changed_features)

    def test_dfga_model(self):
        from uap.models.dfga import DynamicFeatureGraphModel

        X = np.random.randn(150, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        # Introduce missing values to test zero-imputation
        X[0:10, 0] = np.nan

        model = DynamicFeatureGraphModel(embed_dim=8, n_hops=1, epochs=8, batch_size=32)
        model.fit(X, y)
        preds = model.predict(X[:10])
        probs = model.predict_proba(X[:10])
        adj = model.get_learned_adjacency(X[:5])

        self.assertEqual(len(preds), 10)
        self.assertEqual(probs.shape, (10, 2))
        self.assertEqual(adj.shape, (5, 4, 4))

    def test_system_telemetry(self):
        from uap.core.system_telemetry import SystemTelemetryManager

        profile = SystemTelemetryManager.get_hardware_profile()
        self.assertGreater(profile.total_ram_gb, 0.0)
        self.assertIn(profile.recommended_strategy_tier, ["LITE", "STANDARD", "HIGH_PERFORMANCE"])

        bounds = SystemTelemetryManager.calibrate_hyperparameter_bounds(profile)
        self.assertIn("max_estimators", bounds)
        self.assertIn("max_depth", bounds)

    def test_meta_learning_memory(self):
        from uap.core.memory import MetaLearningMemory
        from uap.core.contracts import DatasetSummary, DataHealthScores, DatasetCharacteristicVector, ProblemType

        mem = MetaLearningMemory()
        dcv = DatasetCharacteristicVector(
            dataset_summary=DatasetSummary(total_samples=500, total_features=10, problem_type=ProblemType.BINARY_CLASSIFICATION),
            data_health_scores=DataHealthScores(imbalance_ratio=1.2, non_linearity_index=0.85, high_correlation_pairs=1)
        )
        mem.log_experience(
            dcv=dcv,
            winning_model_name="CatBoostClassifier",
            best_hyperparameters={"iterations": 50},
            validation_score=0.92,
            holdout_score=0.91,
            domain="Healthcare",
            dataset_name="Heart_Demo"
        )
        self.assertEqual(len(mem), 1)

        # Query with exact same dcv -> similarity should be close to 1.0
        match = mem.find_closest_strategy(dcv, similarity_threshold=0.90)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(match["similarity"], 0.95)
        self.assertEqual(match["recommended_model"], "CatBoostClassifier")
        self.assertEqual(match["matched_domain"], "Healthcare")

    def test_expanded_model_families(self):
        from uap.module3_training_hpo.optimizer import DynamicModelOptimizer

        opt = DynamicModelOptimizer(n_trials=1, timeout_sec=5)
        gb = opt._create_model_instance("GradientBoostingClassifier", {"n_estimators": 10}, True, 1.0)
        self.assertIsNotNone(gb)
        nb = opt._create_model_instance("GaussianNB", {}, True, 1.0)
        self.assertIsNotNone(nb)
        en = opt._create_model_instance("ElasticNet", {"alpha": 0.1, "l1_ratio": 0.5}, False, 1.0)
        self.assertIsNotNone(en)
        dfga = opt._create_model_instance("DynamicFeatureGraphModel", {"embed_dim": 8, "epochs": 5}, True, 1.0)
        self.assertIsNotNone(dfga)

    def test_conformal_risk_certifier(self):
        from uap.module4_xai_serving.conformal import ConformalRiskCertifier

        np.random.seed(42)
        X = pd.DataFrame({"x1": np.random.randn(100), "x2": np.random.randn(100)})
        y = (X["x1"] + X["x2"] > 0).astype(int)

        def dummy_pred(df_in):
            return (df_in["x1"] + df_in["x2"] > 0).astype(int).values

        def dummy_proba(df_in):
            p1 = 1.0 / (1.0 + np.exp(-(df_in["x1"] + df_in["x2"]).values))
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T

        certifier = ConformalRiskCertifier()
        certifier.fit(X, y, is_classification=True, predict_func=dummy_pred, predict_proba_func=dummy_proba)

        self.assertTrue(certifier.fitted_)
        self.assertGreater(len(certifier.calibration_scores), 0)

        # Certify an in-distribution sample
        res = certifier.certify_instance(X.iloc[0], predict_func=dummy_pred, predict_proba_func=dummy_proba, alpha=0.05)
        self.assertIn("coverage_guarantee", res)
        self.assertEqual(res["coverage_guarantee"], "95.0%")
        self.assertIn(res["point_prediction"], [0, 1])
        self.assertGreaterEqual(res["set_size"], 1)

    def test_causal_discovery_engine(self):
        from uap.causal.engine import CausalDiscoveryEngine

        np.random.seed(42)
        n = 200
        # Synthetic causal chain: Age -> BloodPressure -> Risk
        age = np.random.uniform(20, 80, size=n)
        bp = 100 + 0.5 * age + np.random.normal(0, 5, size=n)
        risk = (0.02 * bp + np.random.normal(0, 0.5, size=n) > 2.8).astype(int)

        df = pd.DataFrame({"age": age, "blood_pressure": bp, "target": risk})
        X = df.drop(columns=["target"])
        y = df["target"]

        causal = CausalDiscoveryEngine()
        causal.fit(X, y, target_name="target", immutable_features=["age"])

        self.assertTrue(causal.fitted_)
        graph = causal.get_graph_elements()
        self.assertGreaterEqual(len(graph["nodes"]), 3)
        self.assertGreaterEqual(len(graph["edges"]), 1)

        # Test intervention: do(blood_pressure = 110)
        def dummy_risk(df_in):
            return (df_in["blood_pressure"] > 130).astype(int).values

        interv_res = causal.simulate_intervention(
            base_sample=X.iloc[0],
            interventions={"blood_pressure": 110.0},
            predictor_func=dummy_risk,
        )
        self.assertIn("intervened_causal_prediction", interv_res)
        self.assertIn("average_causal_treatment_effect", interv_res)

    def test_self_healing_adaptor(self):
        from uap.module4_xai_serving.conformal import ConformalRiskCertifier
        from uap.module4_xai_serving.self_healing import SelfHealingAdaptor

        np.random.seed(42)
        base_df = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        y_base = (base_df["x"] > 0).astype(int)

        def dummy_p(df_in):
            return (df_in["x"] > 0).astype(int).values

        certifier = ConformalRiskCertifier()
        certifier.fit(base_df, y_base, is_classification=True, predict_func=dummy_p)

        adaptor = SelfHealingAdaptor(baseline_df=base_df, certifier=certifier, drift_psi_threshold=0.20)

        # Non-drifting batch
        normal_batch = pd.DataFrame({"x": np.random.normal(0, 1, 50)})
        res_normal = adaptor.evaluate_and_heal(normal_batch)
        self.assertFalse(res_normal["drift_detected"])

        # Drifting batch (shift mean by +4.0)
        drifting_batch = pd.DataFrame({"x": np.random.normal(4.0, 1, 50)})
        res_drift = adaptor.evaluate_and_heal(drifting_batch)
        self.assertTrue(res_drift["drift_detected"])
        self.assertTrue(res_drift["self_healing_applied"])


if __name__ == "__main__":
    unittest.main()



