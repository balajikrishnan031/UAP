"""
UAP Master Engine: Orchestrates Module 1 through Module 4 into a unified, closed-loop pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple
import time
import pandas as pd
import numpy as np

from uap.core.contracts import (
    DatasetCharacteristicVector,
    ModelStrategyConfiguration,
    EvaluatedModelArtifact,
    ProblemType,
    UserUtilitySpec,
    GoalRiskProfile,
    PredictionReliabilityCard,
    LivingDecisionDossier,
)
from uap.module1_profiler.profiler import DatasetProfiler
from uap.module1_profiler.goal_risk_analyzer import GoalRiskAnalyzer
from uap.module1_profiler.autonomous_intent import AutonomousIntentEngine, IntentDiscoveryCard
from uap.module2_weighting_routing.afwe import AdaptiveFeatureWeightingEngine
from uap.module2_weighting_routing.dmre import DynamicModelRoutingEngine
from uap.module3_training_hpo.optimizer import DynamicModelOptimizer
from uap.module3_training_hpo.stress_tester import RobustnessStressTester
from uap.module4_xai_serving.xai import ExplainabilityEngine
from uap.module4_xai_serving.what_if import WhatIfRecourseEngine
from uap.module4_xai_serving.drift import DriftMonitor
from uap.module4_xai_serving.abstention import AbstentionGate
from uap.module4_xai_serving.counterfactual import FeasibleCounterfactualEngine
from uap.module4_xai_serving.conformal import ConformalRiskCertifier
from uap.module4_xai_serving.reliability_card import ReliabilityCardEngine
from uap.module4_xai_serving.self_healing import SelfHealingAdaptor
from uap.causal.engine import CausalDiscoveryEngine
from uap.core.memory import MetaLearningMemory


class UAPEngine:
    """
    Universal Adaptive Prediction (UAP) Engine.
    Closed-loop automated ML with dynamic feature weighting, meta-routing,
    stress testing, explainability, and drift guard.
    """

    def __init__(
        self,
        n_trials: int = 12,
        timeout_sec: int = 45,
        memory: Optional[MetaLearningMemory] = None,
        memory_path: str = "models/meta_learning_memory.json",
        utility_spec: Optional[UserUtilitySpec] = None,
    ):
        self.n_trials = n_trials
        self.timeout_sec = timeout_sec
        self.memory = memory or MetaLearningMemory(memory_file=memory_path)
        self.default_utility_spec = utility_spec

        # Artifacts
        self.dcv: Optional[DatasetCharacteristicVector] = None
        self.msc: Optional[ModelStrategyConfiguration] = None
        self.ema: Optional[EvaluatedModelArtifact] = None
        self.goal_risk_profile: Optional[GoalRiskProfile] = None
        self.intent_card: Optional[IntentDiscoveryCard] = None

        # Components
        self.intent_engine = AutonomousIntentEngine()
        self.profiler = DatasetProfiler()
        self.goal_risk_analyzer = GoalRiskAnalyzer()
        self.afwe = AdaptiveFeatureWeightingEngine()
        self.dmre = DynamicModelRoutingEngine()
        self.optimizer = DynamicModelOptimizer(n_trials=n_trials, timeout_sec=timeout_sec)
        self.stress_tester = RobustnessStressTester()
        self.xai = ExplainabilityEngine()
        self.what_if_engine: Optional[WhatIfRecourseEngine] = None
        self.drift_monitor: Optional[DriftMonitor] = None
        self.abstention_gate = AbstentionGate()
        self.reliability_card_engine = ReliabilityCardEngine()
        self.feasible_counterfactual_engine: Optional[FeasibleCounterfactualEngine] = None
        self.conformal_certifier = ConformalRiskCertifier()
        self.causal_engine = CausalDiscoveryEngine()
        self.self_healing_adaptor: Optional[SelfHealingAdaptor] = None

        self.feature_names: List[str] = []
        self.target_name: Optional[str] = None
        self.train_features_mean: Optional[pd.Series] = None
        self.train_features_std: Optional[pd.Series] = None

    def auto_discover(self, df: pd.DataFrame, goal_prompt: Optional[str] = None) -> IntentDiscoveryCard:
        """
        UAP 5.0 Cognitive Intent Discovery:
        Automatically infers domain, target column, and business objective without user guessing.
        """
        self.intent_card = self.intent_engine.discover(df, user_goal=goal_prompt)
        return self.intent_card

    def fit(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        target_col: Optional[str] = None,
        test_size: float = 0.20,
        random_state: int = 42,
        domain_tag: str = "General",
        dataset_name: Optional[str] = None,
        utility_spec: Optional[UserUtilitySpec] = None,
        goal_prompt: Optional[str] = None,
    ) -> "UAPEngine":
        """
        Executes end-to-end training:
        Raw Data -> Module 1 (DCV + Goal/Risk) -> Module 2 (MSC) -> Module 3 (EMA) -> Module 4 (XAI, Safety & Drift).
        """
        target_column = target_column or target_col
        utility_spec = utility_spec or self.default_utility_spec

        # Autonomous Intent Discovery if target is unspecified
        if target_column is None:
            self.intent_card = self.auto_discover(df, goal_prompt=goal_prompt)
            target_column = self.intent_card.recommended_target
            if domain_tag == "General":
                domain_tag = self.intent_card.detected_domain
            print(f"[UAP Engine] Autonomous Intent Discovery: Domain='{self.intent_card.detected_domain}' | Target='{target_column}' | Objective='{self.intent_card.primary_business_goal}'")

        t0 = time.time()

        # ----------------------------------------------------
        # Module 1: Dataset Profiler & Goal/Risk Strategy Analyzer
        # ----------------------------------------------------
        print("[UAP Engine] Executing Module 1: Profiling & Health Assessment...")
        self.dcv, X, y = self.profiler.profile(df, target_column=target_column)
        self.target_name = self.dcv.dataset_summary.target_name
        self.feature_names = list(X.columns)

        # UAP 4.0: Goal & Risk Strategy Formulation
        self.goal_risk_profile = self.goal_risk_analyzer.analyze(self.dcv, utility_spec)
        print(f"[UAP Engine] Strategy Formulated: Metric={self.goal_risk_profile.selected_metric} | Posture={self.goal_risk_profile.risk_posture}")

        # Split into Train and Holdout Test before any weighting to prevent data leakage
        from sklearn.model_selection import train_test_split
        stratify = y if self.dcv.dataset_summary.problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION] else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        self.train_features_mean = X_train.mean(numeric_only=True)
        self.train_features_std = X_train.std(numeric_only=True)

        # ----------------------------------------------------
        # Module 2: Feature Weighting & Dynamic Model Routing
        # ----------------------------------------------------
        print("[UAP Engine] Executing Module 2: Adaptive Feature Weighting & Routing...")
        X_train_weighted, feature_config = self.afwe.fit_transform(
            X_train, y_train, self.dcv.dataset_summary.problem_type
        )
        X_test_weighted = self.afwe.transform(X_test)

        self.msc = self.dmre.generate_strategy(self.dcv, feature_config)

        # Check Meta-Learning Memory for topological match & zero-shot warm-starting
        if self.memory:
            prior = self.memory.find_closest_strategy(self.dcv, similarity_threshold=0.88)
            if prior:
                print(f"[UAP Engine] [Meta-Memory Match] Found prior from domain '{prior['matched_domain']}' (Dataset: {prior['matched_dataset']}) with similarity {prior['similarity']:.3f}.")
                print(f"[UAP Engine] Prioritizing historical winning model: {prior['recommended_model']}")
                if prior["recommended_model"] not in self.msc.routing_decision.selected_models:
                    self.msc.routing_decision.selected_models.insert(0, prior["recommended_model"])
                # Prioritize top 4 diverse candidates
                self.msc.routing_decision.selected_models = self.msc.routing_decision.selected_models[:4]
        else:
            self.msc.routing_decision.selected_models = self.msc.routing_decision.selected_models[:4]

        # ----------------------------------------------------
        # Module 3: Automated HPO & Robustness Stress-Testing
        # ----------------------------------------------------
        print(f"[UAP Engine] Executing Module 3: HPO on {self.msc.routing_decision.selected_models}...")
        self.optimizer.fit(X_train_weighted, y_train, self.dcv, self.msc)

        elapsed = time.time() - t0
        print("[UAP Engine] Running Perturbation Stress-Testing & Robustness Analysis...")
        self.ema = self.stress_tester.evaluate_and_stress_test(
            self.optimizer,
            X_test_weighted,
            y_test,
            self.dcv,
            self.msc,
            training_time_sec=elapsed,
        )

        # Register completed journey into Meta-Learning Memory
        if self.memory:
            resolved_ds_name = dataset_name or (self.target_name or "Dataset")
            self.memory.log_experience(
                dcv=self.dcv,
                winning_model_name=self.ema.winning_model_name,
                best_hyperparameters=self.ema.final_hyperparameters,
                validation_score=self.ema.performance_metrics.validation_score,
                holdout_score=self.ema.performance_metrics.test_score,
                domain=domain_tag,
                dataset_name=resolved_ds_name,
                selected_models=self.msc.routing_decision.selected_models,
                scaling_strategy=self.msc.routing_decision.scaling_strategy,
            )

        # ----------------------------------------------------
        # Module 4: Explainability, Safety Gate, Recourse & Drift
        # ----------------------------------------------------
        print("[UAP Engine] Initializing Module 4: TreeSHAP, Abstention Gate, FACE & Drift Guard...")
        # Prioritize tree-based model for fast exact TreeSHAP
        tree_candidates = [m for m in self.optimizer.trained_models.values() if "LGBM" in str(type(m)) or "RandomForest" in str(type(m)) or "ExtraTrees" in str(type(m))]
        primary_model = tree_candidates[0] if tree_candidates else list(self.optimizer.trained_models.values())[0]
        self.xai.fit(primary_model, X_train_weighted)

        self.what_if_engine = WhatIfRecourseEngine(
            predictor_func=self.predict,
            feature_names=self.feature_names,
        )

        self.drift_monitor = DriftMonitor(baseline_df=X_train)

        # Fit Abstention Gate manifold and Feasible Counterfactual bounds
        abst_thresh = self.goal_risk_profile.abstention_threshold if self.goal_risk_profile else 0.60
        self.abstention_gate = AbstentionGate(tau_conf=abst_thresh)
        self.abstention_gate.fit(X_train_weighted, feature_weights=self.afwe.feature_weights)
        self.reliability_card_engine = ReliabilityCardEngine(abstention_threshold=abst_thresh)
        self.feasible_counterfactual_engine = FeasibleCounterfactualEngine(
            predictor_func=self.predict,
            feature_names=self.feature_names,
            predict_proba_func=self.predict_proba,
        ).fit_bounds(X_train)

        # ----------------------------------------------------
        # Module 4 (Advanced): Conformal Risk Certification & Causal SCM
        # ----------------------------------------------------
        is_classif = self.dcv.dataset_summary.problem_type in [
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ]
        self.conformal_certifier.fit(
            X_calib=X_test,
            y_calib=y_test,
            is_classification=is_classif,
            predict_func=self.predict,
            predict_proba_func=self.predict_proba,
        )

        self.causal_engine.fit(
            X=X_train,
            y=y_train,
            target_name=self.target_name or "Target",
        )

        self.self_healing_adaptor = SelfHealingAdaptor(
            baseline_df=X_train,
            certifier=self.conformal_certifier,
        )

        print(f"[UAP Engine] Pipeline Complete. Winning Architecture: {self.ema.winning_model_name} | Score: {self.ema.performance_metrics.test_score} | RDS: {self.ema.performance_metrics.robustness_decay_score}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms input with AFWE weights and makes predictions with winning model/ensemble.
        """
        X_weighted = self.afwe.transform(X)
        return self.optimizer.predict(X_weighted)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        X_weighted = self.afwe.transform(X)
        return self.optimizer.predict_proba(X_weighted)

    def explain_global(self, X: pd.DataFrame) -> Dict[str, float]:
        X_weighted = self.afwe.transform(X)
        return self.xai.get_global_importance(X_weighted)

    def explain_instance(self, instance: pd.Series) -> Dict[str, Any]:
        inst_df = pd.DataFrame([instance])
        inst_weighted = self.afwe.transform(inst_df).iloc[0]
        return self.xai.explain_instance(inst_weighted)

    def simulate_what_if(self, base_instance: pd.Series, modifications: Dict[str, float]) -> Dict[str, Any]:
        if self.what_if_engine is None:
            raise RuntimeError("Engine not fitted yet.")
        return self.what_if_engine.simulate_scenario(base_instance, modifications)

    def find_actionable_recourse(self, base_instance: pd.Series, target_direction: str = "increase") -> Dict[str, Any]:
        if self.what_if_engine is None:
            raise RuntimeError("Engine not fitted yet.")
        return self.what_if_engine.find_actionable_recourse(base_instance, target_direction=target_direction)

    def monitor_drift(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        if self.drift_monitor is None:
            raise RuntimeError("Engine not fitted yet.")
        return self.drift_monitor.calculate_psi(current_df)

    def predict_safe(self, sample: pd.Series) -> Dict[str, Any]:
        """
        Executes safe selective prediction through the Abstention Gate G(x).
        Returns PREDICT, PREDICT_WITH_WARNING, or ABSTAIN.
        """
        sample_df = pd.DataFrame([sample])
        raw_pred = self.predict(sample_df)[0]
        raw_prob = self.predict_proba(sample_df)
        prob_vec = raw_prob[0] if raw_prob is not None else None

        sample_weighted = self.afwe.transform(sample_df).iloc[0]
        return self.abstention_gate.evaluate_sample(
            sample=sample_weighted,
            raw_pred=raw_pred,
            raw_prob=prob_vec,
        )

    def generate_feasible_recourse(
        self,
        sample: pd.Series,
        target_outcome: Any = 1,
        immutable_features: Optional[List[str]] = None,
        direction_constraints: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes constrained counterfactual optimization respecting realistic bounds.
        """
        if self.feasible_counterfactual_engine is None:
            raise RuntimeError("Engine not fitted yet.")

        if immutable_features:
            self.feasible_counterfactual_engine.immutable_features = set(immutable_features)
        if direction_constraints:
            self.feasible_counterfactual_engine.direction_constraints = direction_constraints

        return self.feasible_counterfactual_engine.generate_recourse(
            sample=sample,
            target_outcome=target_outcome,
        )

    def predict_conformal(
        self,
        sample: Any,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Computes finite-sample valid prediction set / interval with (1 - alpha) guarantee.
        For alpha=0.05, coverage is guaranteed to be >= 95%.
        """
        if not self.conformal_certifier.fitted_:
            raise RuntimeError("Conformal certifier not calibrated yet.")
        return self.conformal_certifier.certify_instance(
            sample=sample,
            predict_func=self.predict,
            predict_proba_func=self.predict_proba,
            alpha=alpha,
        )

    def simulate_causal_intervention(
        self,
        base_sample: pd.Series,
        interventions: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Executes Pearl's do-calculus intervention do(X_k = v) and propagates SCM changes.
        """
        if not self.causal_engine.fitted_:
            raise RuntimeError("Causal engine not fitted yet.")
        return self.causal_engine.simulate_intervention(
            base_sample=base_sample,
            interventions=interventions,
            predictor_func=self.predict,
        )

    def get_causal_dag(self) -> Dict[str, Any]:
        """
        Returns nodes and edges of the learned causal DAG for visualization.
        """
        if not self.causal_engine.fitted_:
            raise RuntimeError("Causal engine not fitted yet.")
        return self.causal_engine.get_graph_elements()

    def heal_under_drift(
        self,
        incoming_batch: pd.DataFrame,
        y_incoming: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Performs instant online re-calibration if streaming data drift is detected.
        """
        if self.self_healing_adaptor is None:
            raise RuntimeError("Self-healing adaptor not initialized.")
        return self.self_healing_adaptor.evaluate_and_heal(
            incoming_batch=incoming_batch,
            y_incoming=y_incoming,
            predict_func=self.predict,
            predict_proba_func=self.predict_proba,
        )

    def explain_strategy(self) -> Dict[str, Any]:
        """
        UAP 4.0 Core Strategy Explainer:
        Answers 'Why was this model and configuration selected?' based on data geometry and user goals.
        """
        if not self.goal_risk_profile:
            return {"status": "Model not fitted yet"}
        return {
            "winning_model": self.optimizer.winning_model_name,
            "strategy_name": f"{self.optimizer.winning_model_name} ({self.goal_risk_profile.risk_posture})",
            "is_ensemble": self.optimizer.is_ensemble,
            "ensemble_weights": self.optimizer.ensemble_weights,
            "selected_metric": self.goal_risk_profile.selected_metric,
            "risk_posture": self.goal_risk_profile.risk_posture,
            "cv_strategy": self.goal_risk_profile.cv_strategy,
            "cv_scheme": self.goal_risk_profile.cv_strategy,
            "why_this_configuration": self.goal_risk_profile.strategy_rationale,
            "primary_rationale": self.goal_risk_profile.strategy_rationale,
            "key_tradeoffs": self.goal_risk_profile.key_tradeoffs,
            "tradeoffs": self.goal_risk_profile.key_tradeoffs,
        }

    def predict_with_reliability_card(self, sample: Any) -> PredictionReliabilityCard:
        """
        UAP 4.0 Core User-Facing Artifact:
        Synthesizes predictive output, calibrated confidence, conformal bounds,
        data quality, OOD novelty risk, and drift into an interpretable trust decision.
        """
        if isinstance(sample, dict):
            sample_s = pd.Series(sample)
        elif isinstance(sample, pd.DataFrame):
            sample_s = sample.iloc[0]
        else:
            sample_s = sample

        if self.target_name and hasattr(sample_s, "index") and self.target_name in sample_s.index:
            sample_s = sample_s.drop(self.target_name)

        df_sample = pd.DataFrame([sample_s])
        preds = self.predict(df_sample)
        raw_pred = preds[0] if len(preds) > 0 else None
        raw_probs = self.predict_proba(df_sample)

        try:
            conf_bounds = self.predict_conformal(df_sample)
        except Exception:
            conf_bounds = "N/A"

        drift_stat = "NORMAL"
        if self.drift_monitor and hasattr(self.drift_monitor, "last_drift_detected"):
            drift_stat = "DRIFT_DETECTED" if self.drift_monitor.last_drift_detected else "NORMAL"

        return self.reliability_card_engine.build_card(
            sample=sample_s,
            raw_pred=raw_pred,
            raw_probs=raw_probs,
            conformal_bounds=conf_bounds,
            train_features_mean=self.train_features_mean,
            train_features_std=self.train_features_std,
            drift_status=drift_stat,
        )

    def predict_dossier(
        self,
        sample: Any,
        target_goal: Optional[int] = None,
        immutable_features: Optional[List[str]] = None,
        direction_constraints: Optional[Dict[str, str]] = None,
    ) -> LivingDecisionDossier:
        """
        UAP 5.0 Core Output Artifact: The Living Decision Dossier.
        Radically transcends simple point-prediction numbers by synthesizing:
        1. Predictive Output + 95% Conformal Set/Range.
        2. Multi-tier Trust Certification (OOD, Drift, Data Quality, Abstention).
        3. Primary Causal / Attribution Drivers (Why did this happen?).
        4. Actionable Counterfactual Recourse (What exact minimal action flips the outcome?).
        5. Actionable Business Impact Guidance.
        """
        if isinstance(sample, dict):
            sample_s = pd.Series(sample)
        elif isinstance(sample, pd.DataFrame):
            sample_s = sample.iloc[0]
        else:
            sample_s = sample

        if self.target_name and hasattr(sample_s, "index") and self.target_name in sample_s.index:
            sample_s = sample_s.drop(self.target_name)

        # 1. Base Reliability Card
        card = self.predict_with_reliability_card(sample_s)

        # 2. Causal / Attribution Drivers
        causal_drivers: List[Dict[str, Any]] = []
        try:
            if self.xai is not None:
                df_s = pd.DataFrame([sample_s])
                shap_contribs = self.xai.explain_instance(df_s)
                sorted_feats = sorted(shap_contribs.items(), key=lambda x: abs(x[1]), reverse=True)
                for f, imp in sorted_feats[:4]:
                    causal_drivers.append({
                        "feature": f,
                        "impact": round(float(imp), 4),
                        "value": sample_s.get(f, ""),
                    })
        except Exception:
            pass

        # Fallback to feature weights if SHAP unavailable
        if not causal_drivers and self.msc and self.msc.feature_weighting:
            sorted_weights = sorted(self.msc.feature_weighting.feature_weights.items(), key=lambda x: x[1], reverse=True)
            for f, w in sorted_weights[:4]:
                causal_drivers.append({
                    "feature": f,
                    "impact": round(float(w), 4),
                    "value": sample_s.get(f, ""),
                })

        # 3. Actionable Recourse (if outcome is unfavorable or specified)
        recourse_result: Optional[Dict[str, Any]] = None
        is_classification = self.dcv and self.dcv.dataset_summary.problem_type in [
            ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION
        ]

        if is_classification and card.decision != "ABSTAIN_HUMAN_REVIEW":
            pred_int = int(card.prediction) if card.prediction is not None else 0
            desired_target = target_goal if target_goal is not None else (0 if pred_int == 1 else 1)

            if pred_int != desired_target:
                try:
                    recourse_result = self.generate_feasible_recourse(
                        sample=sample_s,
                        target_outcome=desired_target,
                        immutable_features=immutable_features or [],
                        direction_constraints=direction_constraints or {},
                    )
                except Exception:
                    recourse_result = {"status": "RECOURSE_UNAVAILABLE"}
            else:
                recourse_result = {"status": "ALREADY_FAVORABLE"}

        # 4. Strategic Business Impact Guidance
        domain = self.intent_card.detected_domain if self.intent_card else "Industrial Production"
        if card.decision == "ABSTAIN_HUMAN_REVIEW":
            guidance = f"Autonomous prediction halted for {domain}. Human specialist escalation required due to high novelty or uncertainty."
        elif is_classification:
            guidance = f"Autonomous execution certified for {domain} with {card.confidence_score*100:.1f}% calibrated confidence."
        else:
            guidance = f"Forecast verified for {domain} within conformal bounds {card.conformal_set_or_interval}."

        return LivingDecisionDossier(
            prediction=card.prediction,
            confidence_score=card.confidence_score,
            conformal_interval_95=card.conformal_set_or_interval,
            trust_badge=card.decision,
            abstention_reason=card.abstention_reason,
            ood_risk_level=card.ood_risk_level,
            drift_risk_level=card.drift_risk_level,
            data_quality_score=card.data_quality_score,
            top_causal_drivers=causal_drivers,
            actionable_recourse=recourse_result,
            business_impact_summary=guidance,
        )

    def save(self, filepath: str) -> str:
        """Serializes this fitted engine to disk."""
        from uap.core.persistence import ModelPersistenceManager
        return ModelPersistenceManager.save_engine(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "UAPEngine":
        """Restores a serialized engine from disk."""
        from uap.core.persistence import ModelPersistenceManager
        return ModelPersistenceManager.load_engine(filepath)


