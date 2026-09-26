"""
Module 3 - Part 2 & 4: Resource-Aware Multi-Fidelity HPO & Dynamic Model Trainer.
Optimizes candidate models using Optuna (TPE) and constructs winning models / Stacking Ensembles.
"""

from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, root_mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, HuberRegressor, ElasticNet, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier, CatBoostRegressor = None, None
try:
    from uap.models.dfga import DynamicFeatureGraphModel
except ImportError:
    DynamicFeatureGraphModel = None
from uap.models.novel_tmrm import TopologicalManifoldResonantMachine

from uap.core.contracts import (
    ProblemType,
    DatasetCharacteristicVector,
    ModelStrategyConfiguration,
    PerformanceMetrics,
    EvaluatedModelArtifact,
)
from uap.module3_training_hpo.partitioning import AdaptiveDataPartitioner


# Silence verbose Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class DynamicModelOptimizer:
    """
    Trains candidate models according to MSC instructions using Bayesian TPE HPO.
    """

    def __init__(self, n_trials: int = 15, timeout_sec: int = 60):
        self.n_trials = n_trials
        self.timeout_sec = timeout_sec
        self.scaler = None
        self.trained_models: Dict[str, Any] = {}
        self.validation_scores: Dict[str, float] = {}
        self.validation_stds: Dict[str, float] = {}
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.oof_preds: Dict[str, np.ndarray] = {}
        self.oof_probs: Dict[str, np.ndarray] = {}
        self.winning_model_name: str = ""
        self.is_ensemble: bool = False
        self.ensemble_weights: Dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dcv: DatasetCharacteristicVector,
        msc: ModelStrategyConfiguration,
    ) -> "DynamicModelOptimizer":
        problem_type = dcv.dataset_summary.problem_type
        is_classification = problem_type in [
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ]

        # 1. Apply Scaler
        if msc.routing_decision.scaling_strategy == "RobustScaler":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X.fillna(0)),
            columns=X.columns,
            index=X.index,
        )

        partitioner = AdaptiveDataPartitioner(n_splits=3, random_state=42)
        candidate_models = msc.routing_decision.selected_models

        for model_name in candidate_models:
            best_model, score, score_std, best_p, oof_p, oof_pr = self._optimize_single_model(
                model_name=model_name,
                X=X_scaled,
                y=y,
                dcv=dcv,
                msc=msc,
                partitioner=partitioner,
                is_classification=is_classification,
            )
            if best_model is not None:
                self.trained_models[model_name] = best_model
                self.validation_scores[model_name] = score
                self.validation_stds[model_name] = score_std
                self.best_params[model_name] = best_p
                self.oof_preds[model_name] = oof_p
                if oof_pr is not None:
                    self.oof_probs[model_name] = oof_pr

        # Determine Winning Model or Blend
        if not self.trained_models:
            raise RuntimeError("No candidate models succeeded during training.")

        best_single_name = max(self.validation_scores.items(), key=lambda x: x[1])[0]
        best_single_score = self.validation_scores[best_single_name]

        # Consider top models within 12% of best single score
        threshold = best_single_score - abs(best_single_score) * 0.12
        eligible_models = [m for m, s in self.validation_scores.items() if s >= threshold]
        if len(eligible_models) < 2:
            eligible_models = sorted(self.validation_scores.keys(), key=lambda x: self.validation_scores[x], reverse=True)[:2]

        use_ensemble = False
        opt_weights: Dict[str, float] = {}

        if msc.routing_decision.primary_strategy == "Stacking_Ensemble" and len(eligible_models) > 1:
            k = len(eligible_models)
            init_w = np.ones(k) / k
            bounds = [(0.0, 1.0) for _ in range(k)]
            constraints = ({'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1.0)})

            y_arr = np.array(y)

            if is_classification:
                def loss_fn(w):
                    w_norm = np.maximum(0, w)
                    s = np.sum(w_norm)
                    if s == 0:
                        return 999.0
                    w_norm = w_norm / s
                    blend = np.zeros_like(self.oof_probs[eligible_models[0]])
                    for m, weight in zip(eligible_models, w_norm):
                        blend += weight * self.oof_probs[m]
                    if blend.shape[1] == 2:
                        p_labels = (blend[:, 1] >= 0.5).astype(int)
                    else:
                        p_labels = np.argmax(blend, axis=1)
                    return -float(f1_score(y_arr, p_labels, average="macro", zero_division=0))
            else:
                def loss_fn(w):
                    w_norm = np.maximum(0, w)
                    s = np.sum(w_norm)
                    if s == 0:
                        return 999.0
                    w_norm = w_norm / s
                    blend = np.zeros(len(y_arr))
                    for m, weight in zip(eligible_models, w_norm):
                        blend += weight * self.oof_preds[m]
                    return float(root_mean_squared_error(y_arr, blend))

            try:
                res = minimize(loss_fn, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
                if res.success or res.fun is not None:
                    w_opt = np.maximum(0, res.x)
                    w_opt = w_opt / (np.sum(w_opt) if np.sum(w_opt) > 0 else 1.0)
                    ensemble_score = -res.fun if is_classification else -res.fun

                    # GUARANTEED SUPERIORITY: Only ensemble if it strictly beats the best single model on CV!
                    if ensemble_score > best_single_score + 1e-4:
                        for m, w_val in zip(eligible_models, w_opt):
                            if w_val > 0.02:
                                opt_weights[m] = float(w_val)
                        tot = sum(opt_weights.values()) or 1.0
                        opt_weights = {m: w / tot for m, w in opt_weights.items()}
                        if len(opt_weights) > 1:
                            use_ensemble = True
            except Exception:
                use_ensemble = False

        if use_ensemble and len(opt_weights) > 1:
            self.is_ensemble = True
            self.ensemble_weights = opt_weights
            top_names = sorted(opt_weights.keys(), key=lambda x: opt_weights[x], reverse=True)
            self.winning_model_name = "Stacking_Ensemble_" + "_".join(top_names[:2])
        else:
            self.is_ensemble = False
            self.winning_model_name = best_single_name
            self.ensemble_weights = {best_single_name: 1.0}

        return self

    def _optimize_single_model(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        dcv: DatasetCharacteristicVector,
        msc: ModelStrategyConfiguration,
        partitioner: AdaptiveDataPartitioner,
        is_classification: bool,
    ) -> Tuple[Any, float, float, Dict[str, Any], np.ndarray, Optional[np.ndarray]]:
        imbalance = dcv.data_health_scores.imbalance_ratio

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_hyperparameters(model_name, trial, is_classification, imbalance)
            fold_scores = []

            for train_idx, val_idx in partitioner.get_splits(X, y, dcv):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = self._create_model_instance(model_name, params, is_classification, imbalance)
                model.fit(X_tr, y_tr)

                if is_classification:
                    preds = model.predict(X_val)
                    score = f1_score(y_val, preds, average="macro", zero_division=0)
                else:
                    preds = model.predict(X_val)
                    rmse = root_mean_squared_error(y_val, preds)
                    score = -rmse

                fold_scores.append(score)

            return float(np.mean(fold_scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=SuccessiveHalvingPruner(),
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout_sec)

        best_params = self._sample_hyperparameters(
            model_name, study.best_trial, is_classification, imbalance
        )

        # Collect Out-Of-Fold (OOF) cross-validation predictions with best_params
        oof_preds = np.zeros(len(X), dtype=float)
        oof_probs = None
        classes = np.unique(y) if is_classification else None
        n_classes = len(classes) if is_classification else 0
        if is_classification:
            oof_probs = np.zeros((len(X), n_classes), dtype=float)

        for train_idx, val_idx in partitioner.get_splits(X, y, dcv):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_m = self._create_model_instance(model_name, best_params, is_classification, imbalance)
            fold_m.fit(X_tr, y_tr)
            p = fold_m.predict(X_val)
            oof_preds[val_idx] = p

            if is_classification:
                if hasattr(fold_m, "predict_proba"):
                    pr = fold_m.predict_proba(X_val)
                    if pr.shape[1] == n_classes:
                        oof_probs[val_idx] = pr
                    else:
                        oof_probs[val_idx, :pr.shape[1]] = pr
                else:
                    for idx_i, pred_val in zip(val_idx, p):
                        c_idx = np.where(classes == pred_val)[0]
                        if len(c_idx) > 0:
                            oof_probs[idx_i, c_idx[0]] = 1.0

        final_model = self._create_model_instance(model_name, best_params, is_classification, imbalance)
        final_model.fit(X, y)

        best_score = study.best_value
        return final_model, best_score, 0.02, study.best_params, oof_preds, oof_probs

    def _sample_hyperparameters(
        self, model_name: str, trial_or_dict: Any, is_classification: bool, imbalance: float
    ) -> Dict[str, Any]:
        params = {}
        if isinstance(trial_or_dict, optuna.Trial):
            trial = trial_or_dict
            if "LightGBM" in model_name:
                # Expanded: up to 300 trees, deeper leaves, regularisation
                params["n_estimators"]    = trial.suggest_int("n_estimators", 100, 300, step=50)
                params["learning_rate"]   = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
                params["num_leaves"]      = trial.suggest_int("num_leaves", 31, 127)
                params["max_depth"]       = trial.suggest_int("max_depth", 4, 10)
                params["min_child_samples"] = trial.suggest_int("min_child_samples", 10, 50)
                params["subsample"]       = trial.suggest_float("subsample", 0.6, 1.0)
                params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
                params["reg_alpha"]       = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True)
                params["reg_lambda"]      = trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True)
                params["verbose"] = -1
            elif "XGBoost" in model_name:
                params["n_estimators"]    = trial.suggest_int("n_estimators", 100, 300, step=50)
                params["learning_rate"]   = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
                params["max_depth"]       = trial.suggest_int("max_depth", 3, 9)
                params["subsample"]       = trial.suggest_float("subsample", 0.6, 1.0)
                params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
                params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
                params["gamma"]           = trial.suggest_float("gamma", 0.0, 0.5)
                params["reg_alpha"]       = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True)
                params["verbosity"] = 0
            elif "CatBoost" in model_name and CatBoostClassifier is not None:
                params["iterations"]  = trial.suggest_int("iterations", 100, 300, step=50)
                params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
                params["depth"]       = trial.suggest_int("depth", 4, 8)
                params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)
                params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 1.0)
                # silent=True is the correct way to suppress CatBoost output (verbose=0 prints all)
                params["silent"] = True
            elif "GradientBoosting" in model_name:
                params["n_estimators"] = trial.suggest_int("n_estimators", 100, 250, step=50)
                params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
                params["max_depth"]    = trial.suggest_int("max_depth", 3, 6)
                params["subsample"]    = trial.suggest_float("subsample", 0.6, 1.0)
            elif "GaussianNB" in model_name:
                params = {}
            elif "ElasticNet" in model_name:
                params["alpha"]    = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 0.95)
            elif "Lasso" in model_name:
                params["alpha"] = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
            elif "DynamicFeatureGraphModel" in model_name:
                params["embed_dim"] = trial.suggest_categorical("embed_dim", [16, 32])
                params["epochs"]    = trial.suggest_categorical("epochs", [25, 40])
            elif "SVC" in model_name or "SVR" in model_name:
                params["C"]     = trial.suggest_float("C", 0.01, 100.0, log=True)
                params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
            elif "MLP" in model_name:
                params["alpha"]               = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
                params["learning_rate_init"]   = trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True)
            elif "KNeighbors" in model_name:
                params["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 21, step=2)
                params["weights"]     = trial.suggest_categorical("weights", ["uniform", "distance"])
            elif "ExtraTrees" in model_name or "RandomForest" in model_name:
                params["n_estimators"] = trial.suggest_int("n_estimators", 100, 300, step=50)
                params["max_depth"]    = trial.suggest_int("max_depth", 5, 20)
                params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
                params["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            elif "HuberRegressor" in model_name:
                params["epsilon"] = trial.suggest_float("epsilon", 1.1, 2.0)
                params["alpha"]   = trial.suggest_float("alpha", 1e-5, 1.0, log=True)
            elif "LogisticRegression" in model_name or "Ridge" in model_name:
                params["C" if is_classification else "alpha"] = trial.suggest_float(
                    "reg_strength", 0.001, 100.0, log=True
                )
        else:
            params = dict(trial_or_dict.params)

        return params

    def _create_model_instance(
        self, model_name: str, params: Dict[str, Any], is_classification: bool, imbalance: float
    ) -> Any:
        p = params.copy()
        use_balanced = is_classification and imbalance > 1.5

        if "CatBoostClassifier" in model_name and CatBoostClassifier is not None:
            if use_balanced:
                p["scale_pos_weight"] = min(20.0, imbalance)
            p.pop("verbose", None)
            p.pop("silent", None)
            return CatBoostClassifier(**p, random_seed=42, silent=True)
        elif "CatBoostRegressor" in model_name and CatBoostRegressor is not None:
            p.pop("verbose", None)
            p.pop("silent", None)
            return CatBoostRegressor(**p, random_seed=42, silent=True)
        elif "GradientBoostingClassifier" in model_name:
            return GradientBoostingClassifier(**p, random_state=42)
        elif "GradientBoostingRegressor" in model_name:
            return GradientBoostingRegressor(**p, random_state=42)
        elif "GaussianNB" in model_name:
            return GaussianNB()
        elif "ElasticNet" in model_name:
            return ElasticNet(**p, random_state=42)
        elif "Lasso" in model_name:
            return Lasso(**p, random_state=42)
        elif "DynamicFeatureGraphModel" in model_name and DynamicFeatureGraphModel is not None:
            return DynamicFeatureGraphModel(**p, is_classification=is_classification, random_state=42)
        elif "SVC" in model_name:
            cw = "balanced" if use_balanced else None
            return SVC(**p, class_weight=cw, probability=True, random_state=42)
        elif "SVR" in model_name:
            return SVR(**p)
        elif "MLPClassifier" in model_name:
            return MLPClassifier(**p, hidden_layer_sizes=(64, 32), max_iter=250, random_state=42)
        elif "MLPRegressor" in model_name:
            return MLPRegressor(**p, hidden_layer_sizes=(64, 32), max_iter=250, random_state=42)
        elif "KNeighborsClassifier" in model_name:
            return KNeighborsClassifier(**p)
        elif "KNeighborsRegressor" in model_name:
            return KNeighborsRegressor(**p)
        elif "LightGBMClassifier" in model_name:
            if use_balanced:
                p["scale_pos_weight"] = min(20.0, imbalance)
            return lgb.LGBMClassifier(**p, random_state=42)
        elif "LightGBMRegressor" in model_name:
            return lgb.LGBMRegressor(**p, random_state=42)
        elif "XGBoostClassifier" in model_name:
            if use_balanced:
                p["scale_pos_weight"] = min(20.0, imbalance)
            return xgb.XGBClassifier(**p, random_state=42, eval_metric="logloss")
        elif "XGBoostRegressor" in model_name:
            return xgb.XGBRegressor(**p, random_state=42)
        elif "ExtraTreesClassifier" in model_name:
            cw = "balanced" if use_balanced else None
            return ExtraTreesClassifier(**p, class_weight=cw, random_state=42)
        elif "ExtraTreesRegressor" in model_name:
            return ExtraTreesRegressor(**p, random_state=42)
        elif "RandomForestClassifier" in model_name:
            cw = "balanced" if use_balanced else None
            return RandomForestClassifier(**p, class_weight=cw, random_state=42)
        elif "RandomForestRegressor" in model_name:
            return RandomForestRegressor(**p, random_state=42)
        elif "HuberRegressor" in model_name:
            return HuberRegressor(**p)
        elif "LogisticRegression" in model_name:
            reg = p.pop("reg_strength", 1.0)
            cw = "balanced" if use_balanced else None
            return LogisticRegression(C=reg, class_weight=cw, max_iter=500, random_state=42)
        elif "RidgeRegressor" in model_name:
            alpha = p.pop("reg_strength", 1.0)
            return Ridge(alpha=alpha, random_state=42)
        elif "TMRMClassifier" in model_name:
            return TopologicalManifoldResonantMachine(task_type="classification", n_resonators="auto", random_state=42, **p)
        elif "TMRMRegressor" in model_name:
            return TopologicalManifoldResonantMachine(task_type="regression", n_resonators="auto", random_state=42, **p)
        else:
            cw = "balanced" if use_balanced else None
            return ExtraTreesClassifier(class_weight=cw, random_state=42) if is_classification else ExtraTreesRegressor(random_state=42)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Executes prediction using winning model or weighted ensemble.
        """
        X_scaled = pd.DataFrame(
            self.scaler.transform(X.fillna(0)),
            columns=X.columns,
            index=X.index,
        )

        if not self.is_ensemble:
            return self.trained_models[self.winning_model_name].predict(X_scaled)

        # Classification: Soft-voting using predict_proba
        first_m = list(self.trained_models.values())[0]
        if hasattr(first_m, "predict_proba"):
            probs = self.predict_proba(X)
            if probs is not None:
                if probs.ndim == 2 and probs.shape[1] == 2:
                    return (probs[:, 1] >= 0.5).astype(int)
                return np.argmax(probs, axis=1)

        # Regression: Weighted average of predictions
        blended = np.zeros(len(X_scaled), dtype=float)
        total_w = sum(self.ensemble_weights.values()) or 1.0
        for name, w in self.ensemble_weights.items():
            if name in self.trained_models and w > 0:
                blended += w * self.trained_models[name].predict(X_scaled)
        return blended / total_w

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        X_scaled = pd.DataFrame(
            self.scaler.transform(X.fillna(0)),
            columns=X.columns,
            index=X.index,
        )
        if not self.is_ensemble:
            m = self.trained_models[self.winning_model_name]
            if hasattr(m, "predict_proba"):
                return m.predict_proba(X_scaled)
            return None

        probs_sum = None
        weights_sum = 0.0
        for name, w in self.ensemble_weights.items():
            if name in self.trained_models and w > 0:
                model = self.trained_models[name]
                if hasattr(model, "predict_proba"):
                    p = model.predict_proba(X_scaled)
                    if probs_sum is None:
                        probs_sum = np.zeros_like(p, dtype=float)
                    probs_sum += w * p
                    weights_sum += w

        if probs_sum is not None and weights_sum > 0:
            return probs_sum / weights_sum
        return None
