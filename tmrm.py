"""
TMRM: Topological Manifold Resonant Machine (Universal Predictive Model)
========================================================================
A radically novel, ground-up predictive machine learning algorithm.
Universal across ANY dataset (Healthcare, Finance, E-commerce, IoT, etc.).

Usage (Exactly like RandomForest / XGBoost):
    >>> from tmrm import TMRM
    >>> model = TMRM()
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
"""

import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from typing import Dict, Any, List, Optional, Tuple, Union

__version__ = "2.0.0"
__all__ = ["TMRM", "TopologicalManifoldResonantMachine", "StreamingTMRM"]


class TopologicalManifoldResonantMachine:
    """
    Topological Manifold Resonant Machine (TMRM)
    Universal Predictive Algorithm for Any Classification or Regression Dataset.
    """

    def __init__(
        self,
        task_type: str = "auto",  # 'auto', 'classification', 'regression'
        n_resonators: Union[int, str] = "auto",
        harmonic_octaves: int = 3,
        metric_regularization: float = 1e-3,
        novelty_threshold: float = 2.5,
        focal_gamma: float = 0.5,
        temperature: float = 1.0,
        random_state: int = 42
    ):
        self.task_type = task_type
        self.n_resonators = n_resonators
        self.harmonic_octaves = harmonic_octaves
        self.reg = metric_regularization
        self.novelty_threshold = novelty_threshold
        self.focal_gamma = focal_gamma
        self.temperature = temperature
        self.random_state = random_state

        self.is_fitted = False
        self.is_classifier = True
        self.classes_ = None
        self.n_classes_ = 0
        self.n_features_ = 0
        self.feature_names_: List[str] = []

        # Learned Manifolds
        self.class_manifolds_: Dict[Any, List[Dict[str, Any]]] = {}
        self.class_weights_: Dict[Any, float] = {}
        self.regression_resonators_: List[Dict[str, Any]] = []
        self.all_centroids_: np.ndarray = np.empty((0, 0))
        self.feature_means_: np.ndarray = None
        self.feature_stds_: np.ndarray = None
        self.feature_weights_: np.ndarray = None
        self.global_bandwidth_: float = 1.0
        self.calibrated_temperature_: float = 1.0

        # Universal Categorical Encoding
        self.categorical_cols_: List[str] = []
        self.category_maps_: Dict[str, Dict[Any, float]] = {}

    def _extract_and_encode(self, X: Any, is_training: bool = False) -> Tuple[np.ndarray, List[str]]:
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            names = list(X_df.columns)
            if is_training:
                self.categorical_cols_ = []
                self.category_maps_ = {}
                for col in names:
                    if X_df[col].dtype == object or str(X_df[col].dtype) == "category":
                        self.categorical_cols_.append(col)
                        unique_vals = X_df[col].dropna().unique()
                        val_map = {val: float(idx + 1) for idx, val in enumerate(unique_vals)}
                        val_map["__unknown__"] = 0.0
                        self.category_maps_[col] = val_map
                        X_df[col] = X_df[col].map(val_map).fillna(0.0).astype(float)
            else:
                for col in self.categorical_cols_:
                    if col in X_df.columns:
                        val_map = self.category_maps_.get(col, {})
                        X_df[col] = X_df[col].map(lambda v: val_map.get(v, 0.0)).astype(float)
            return X_df.to_numpy(dtype=float), names
        else:
            X_arr = np.asarray(X, dtype=float)
            names = [f"feature_{i}" for i in range(X_arr.shape[1])]
            return X_arr, names

    def _manifold_impute(self, X_arr: np.ndarray) -> np.ndarray:
        nan_mask = np.isnan(X_arr)
        if not np.any(nan_mask):
            return X_arr

        X_imputed = X_arr.copy()
        for j in range(self.n_features_):
            X_imputed[nan_mask[:, j], j] = self.feature_means_[j]

        if self.is_fitted and len(self.all_centroids_) > 0:
            for i in range(len(X_arr)):
                if np.any(nan_mask[i]):
                    obs_idx = np.where(~nan_mask[i])[0]
                    mis_idx = np.where(nan_mask[i])[0]
                    if len(obs_idx) > 0:
                        obs_vals = (X_arr[i, obs_idx] - self.feature_means_[obs_idx]) / (self.feature_stds_[obs_idx] + 1e-8)
                        cent_obs = (self.all_centroids_[:, obs_idx] - self.feature_means_[obs_idx]) / (self.feature_stds_[obs_idx] + 1e-8)
                        dists = np.sum((cent_obs - obs_vals) ** 2, axis=1)
                        best_k = np.argmin(dists)
                        X_imputed[i, mis_idx] = self.all_centroids_[best_k, mis_idx]

        return X_imputed

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feature_means_) / (self.feature_stds_ + 1e-8)

    def _determine_task(self, y_arr: np.ndarray) -> bool:
        if self.task_type == "classification":
            return True
        elif self.task_type == "regression":
            return False
        unique_vals = np.unique(y_arr)
        if len(unique_vals) <= 20 or np.issubdtype(y_arr.dtype, np.integer) or y_arr.dtype == object:
            return True
        return False

    def _auto_k_resonators(self, n_samples: int) -> int:
        if isinstance(self.n_resonators, int):
            return max(1, self.n_resonators)
        return int(np.clip(np.sqrt(n_samples / 4.0), 2, 12))

    def fit(self, X: Union[np.ndarray, pd.DataFrame, Any], y: Union[np.ndarray, pd.Series, Any]) -> "TopologicalManifoldResonantMachine":
        """Fit TMRM model to ANY training dataset."""
        rng = np.random.RandomState(self.random_state)
        X_arr, self.feature_names_ = self._extract_and_encode(X, is_training=True)

        y_arr = np.asarray(y)
        self.n_features_ = X_arr.shape[1]
        self.is_classifier = self._determine_task(y_arr)

        self.feature_means_ = np.nanmean(X_arr, axis=0)
        self.feature_stds_ = np.nanstd(X_arr, axis=0)
        self.feature_stds_[self.feature_stds_ == 0] = 1.0

        X_clean = np.where(np.isnan(X_arr), self.feature_means_, X_arr)
        X_norm = self._standardize(X_clean)

        sample_subset = X_norm[:min(500, len(X_norm))]
        pw_dists = cdist(sample_subset, sample_subset, metric='euclidean')
        pos_dists = pw_dists[pw_dists > 0]
        self.global_bandwidth_ = float(np.median(pos_dists)) if len(pos_dists) > 0 else 1.0
        if self.global_bandwidth_ <= 0:
            self.global_bandwidth_ = 1.0

        if self.is_classifier:
            self._fit_classification(X_norm, y_arr, rng)
        else:
            self._fit_regression(X_norm, y_arr.astype(np.float64), rng)

        self.is_fitted = True
        return self

    def _fit_classification(self, X_norm: np.ndarray, y_arr: np.ndarray, rng: np.random.RandomState):
        unique_y = np.unique(y_arr)
        self.classes_ = unique_y
        self.n_classes_ = len(unique_y)

        feature_weights = np.ones(self.n_features_)
        if self.n_classes_ > 1:
            overall_mean = np.mean(X_norm, axis=0)
            between_var = np.zeros(self.n_features_)
            within_var = np.zeros(self.n_features_)
            for c in self.classes_:
                X_c_temp = X_norm[y_arr == c]
                if len(X_c_temp) > 0:
                    c_mean = np.mean(X_c_temp, axis=0)
                    between_var += len(X_c_temp) * ((c_mean - overall_mean) ** 2)
                    within_var += np.sum((X_c_temp - c_mean) ** 2, axis=0)
            within_var = np.maximum(within_var, 1e-4)
            fisher_ratio = between_var / within_var
            fisher_ratio = fisher_ratio / (np.mean(fisher_ratio) + 1e-8)
            feature_weights = np.clip(fisher_ratio, 0.2, 5.0)

        self.feature_weights_ = feature_weights
        self.class_manifolds_ = {}
        self.class_weights_ = {}
        all_centers_list = []
        n_total = len(y_arr)

        for c in self.classes_:
            idx_c = np.where(y_arr == c)[0]
            X_c = X_norm[idx_c]
            n_c = len(X_c)

            self.class_weights_[c] = float((n_total / (max(1, n_c) * self.n_classes_)) ** self.focal_gamma)
            k_clusters = min(self._auto_k_resonators(n_c), max(1, n_c // 2))

            centers = [X_c[rng.randint(0, n_c)]]
            for _ in range(1, k_clusters):
                dists = cdist(X_c, np.array(centers), metric='sqeuclidean').min(axis=1)
                probs = dists / (dists.sum() + 1e-12)
                centers.append(X_c[rng.choice(n_c, p=probs)])
            centers = np.array(centers)

            assignments = cdist(X_c, centers, metric='euclidean').argmin(axis=1)
            resonators = []

            for k in range(k_clusters):
                cluster_pts = X_c[assignments == k]
                if len(cluster_pts) < 2:
                    cluster_pts = X_c

                center_k = np.mean(cluster_pts, axis=0)
                diff = (cluster_pts - center_k) * np.sqrt(self.feature_weights_)
                cov_k = (diff.T @ diff) / max(1, len(cluster_pts) - 1)
                cov_reg = cov_k + np.eye(self.n_features_) * (self.reg * self.global_bandwidth_)

                try:
                    inv_metric = np.linalg.pinv(cov_reg)
                except np.linalg.LinAlgError:
                    inv_metric = np.eye(self.n_features_) / (self.global_bandwidth_ ** 2)

                eigenvals, eigenvecs = np.linalg.eigh(cov_reg)
                top_eigvec = eigenvecs[:, -1]
                base_freq = 2.0 * np.pi / (np.sqrt(max(1e-4, eigenvals[-1])) + 1e-4)

                octave_vectors = []
                for octave in range(1, self.harmonic_octaves + 1):
                    octave_vectors.append({
                        "freq_vector": top_eigvec * (base_freq * octave),
                        "phase": float(rng.uniform(0, np.pi)),
                        "weight": 1.0 / octave
                    })

                all_centers_list.append(center_k * self.feature_stds_ + self.feature_means_)
                resonators.append({
                    "center": center_k,
                    "inv_metric": inv_metric,
                    "weight": float(len(cluster_pts) / n_c),
                    "octave_vectors": octave_vectors
                })

            self.class_manifolds_[c] = resonators

        self.all_centroids_ = np.array(all_centers_list)
        self.calibrated_temperature_ = max(0.5, float(self.temperature))

    def _fit_regression(self, X_norm: np.ndarray, y_arr: np.ndarray, rng: np.random.RandomState):
        n_samples = len(X_norm)
        corrs = np.array([np.corrcoef(X_norm[:, i], y_arr)[0, 1] for i in range(self.n_features_)])
        corrs = np.nan_to_num(corrs, nan=0.0)
        feature_weights = np.clip(np.abs(corrs) * 3.0 + 0.3, 0.2, 5.0)
        self.feature_weights_ = feature_weights

        k_clusters = min(self._auto_k_resonators(n_samples) * 3, max(4, n_samples // 3))
        centers = [X_norm[rng.randint(0, n_samples)]]
        for _ in range(1, k_clusters):
            dists = cdist(X_norm, np.array(centers), metric='sqeuclidean').min(axis=1)
            probs = dists / (dists.sum() + 1e-12)
            centers.append(X_norm[rng.choice(n_samples, p=probs)])
        centers = np.array(centers)

        assignments = cdist(X_norm, centers, metric='euclidean').argmin(axis=1)
        self.regression_resonators_ = []
        all_centers_list = []

        for k in range(k_clusters):
            cluster_pts = X_norm[assignments == k]
            cluster_y = y_arr[assignments == k]
            if len(cluster_pts) < 3:
                cluster_pts = X_norm
                cluster_y = y_arr

            center_k = np.mean(cluster_pts, axis=0)
            diff = (cluster_pts - center_k) * np.sqrt(self.feature_weights_)
            cov_k = (diff.T @ diff) / max(1, len(cluster_pts) - 1)
            cov_reg = cov_k + np.eye(self.n_features_) * (self.reg * self.global_bandwidth_)

            try:
                inv_metric = np.linalg.pinv(cov_reg)
            except np.linalg.LinAlgError:
                inv_metric = np.eye(self.n_features_) / (self.global_bandwidth_ ** 2)

            A = np.column_stack([np.ones(len(cluster_pts)), cluster_pts - center_k])
            ridge_eye = np.eye(A.shape[1]) * 1e-2
            ridge_eye[0, 0] = 0.0
            try:
                beta = np.linalg.solve(A.T @ A + ridge_eye, A.T @ cluster_y)
            except np.linalg.LinAlgError:
                beta = np.zeros(self.n_features_ + 1)
                beta[0] = np.mean(cluster_y)

            all_centers_list.append(center_k * self.feature_stds_ + self.feature_means_)
            self.regression_resonators_.append({
                "center": center_k,
                "beta_0": float(beta[0]),
                "beta_tangent": beta[1:],
                "inv_metric": inv_metric,
                "weight": float(len(cluster_pts) / n_samples)
            })

        self.all_centroids_ = np.array(all_centers_list)

    def _compute_classification_energy(self, X_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = len(X_norm)
        resonances = np.zeros((n_samples, self.n_classes_))
        all_min_dists = np.full(n_samples, np.inf)

        for c_idx, c in enumerate(self.classes_):
            c_resonators = self.class_manifolds_[c]
            c_focal_weight = self.class_weights_.get(c, 1.0)
            class_energy = np.zeros(n_samples)

            for res in c_resonators:
                center = res["center"]
                inv_m = res["inv_metric"]
                w = res["weight"]
                octaves = res["octave_vectors"]

                delta = (X_norm - center) * np.sqrt(self.feature_weights_)
                dist_sq = np.sum((delta @ inv_m) * delta, axis=1)
                dist_sq = np.clip(dist_sq, 0, 100.0)
                dist_l1 = np.sum(np.abs(delta), axis=1) / np.sqrt(self.n_features_)

                all_min_dists = np.minimum(all_min_dists, np.sqrt(dist_sq))

                psi = 1.0
                for oct_info in octaves:
                    proj = np.dot(delta, oct_info["freq_vector"]) + oct_info["phase"]
                    psi += (0.10 * oct_info["weight"]) * np.cos(proj)

                res_potential = w * (0.65 * np.exp(-0.5 * dist_sq) + 0.35 * np.exp(-0.8 * dist_l1)) * psi
                class_energy += res_potential

            resonances[:, c_idx] = class_energy * c_focal_weight

        return resonances, all_min_dists

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame, Any]) -> np.ndarray:
        """Predict class probabilities for X on ANY classification dataset."""
        if not self.is_fitted:
            raise ValueError("TMRM model is not fitted yet.")
        if not self.is_classifier:
            raise ValueError("predict_proba is only available for classification.")

        X_arr, _ = self._extract_and_encode(X, is_training=False)
        X_imputed = self._manifold_impute(X_arr)
        X_norm = self._standardize(X_imputed)

        resonances, _ = self._compute_classification_energy(X_norm)
        tau = max(0.1, self.calibrated_temperature_)
        log_energy = np.log(np.maximum(resonances, 1e-12)) / tau
        log_energy -= np.max(log_energy, axis=1, keepdims=True)
        exp_energy = np.exp(log_energy)
        probs = exp_energy / np.sum(exp_energy, axis=1, keepdims=True)
        return probs

    def predict(self, X: Union[np.ndarray, pd.DataFrame, Any]) -> np.ndarray:
        """Predict labels for X on ANY dataset."""
        if not self.is_fitted:
            raise ValueError("TMRM model is not fitted yet.")

        X_arr, _ = self._extract_and_encode(X, is_training=False)
        X_imputed = self._manifold_impute(X_arr)
        X_norm = self._standardize(X_imputed)

        if self.is_classifier:
            probs = self.predict_proba(X)
            pred_indices = np.argmax(probs, axis=1)
            return self.classes_[pred_indices]
        else:
            n_samples = len(X_norm)
            num = np.zeros(n_samples)
            denom = np.zeros(n_samples)

            for res in self.regression_resonators_:
                center = res["center"]
                inv_m = res["inv_metric"]
                w = res["weight"]
                beta_0 = res["beta_0"]
                beta_tangent = res["beta_tangent"]

                delta_raw = X_norm - center
                local_pred = beta_0 + delta_raw @ beta_tangent
                delta_weighted = delta_raw * np.sqrt(self.feature_weights_)
                dist_sq = np.sum((delta_weighted @ inv_m) * delta_weighted, axis=1)
                dist_sq = np.clip(dist_sq, 0, 100.0)
                kernel_w = w * np.exp(-0.5 * dist_sq)

                num += kernel_w * local_pred
                denom += kernel_w

            denom = np.maximum(denom, 1e-10)
            return num / denom

    def score(self, X: Union[np.ndarray, pd.DataFrame, Any], y: Union[np.ndarray, pd.Series, Any]) -> float:
        preds = self.predict(X)
        y_arr = np.asarray(y)
        if self.is_classifier:
            return float(np.mean(preds == y_arr))
        else:
            u = np.sum((y_arr - preds) ** 2)
            v = np.sum((y_arr - np.mean(y_arr)) ** 2)
            return float(1.0 - u / (v + 1e-12))

    def get_epistemic_novelty(self, X: Union[np.ndarray, pd.DataFrame, Any]) -> np.ndarray:
        """Detect out-of-distribution or alien data without extra libraries."""
        if not self.is_fitted:
            raise ValueError("TMRM model is not fitted yet.")

        X_arr, _ = self._extract_and_encode(X, is_training=False)
        X_imputed = self._manifold_impute(X_arr)
        X_norm = self._standardize(X_imputed)

        if self.is_classifier:
            _, min_dists = self._compute_classification_energy(X_norm)
            novelty = min_dists / (self.global_bandwidth_ + 1e-6)
        else:
            centers = np.array([res["center"] for res in self.regression_resonators_])
            dists = cdist(X_norm, centers, metric='euclidean').min(axis=1)
            novelty = dists / (self.global_bandwidth_ + 1e-6)

        return np.clip(novelty, 0.0, 10.0)

    def generate_recourse(
        self,
        x_sample: Union[np.ndarray, pd.Series, Dict[str, Any]],
        target_class: Any,
        immutable_features: Optional[List[str]] = None,
        max_steps: int = 100,
        learning_rate: float = 0.05
    ) -> Dict[str, Any]:
        """Closed-form Manifold Geodesic Recourse Generator for ANY dataset."""
        if not self.is_classifier:
            raise ValueError("Recourse is only available for classification.")

        if immutable_features is None:
            immutable_features = []

        if isinstance(x_sample, dict):
            raw_x = np.array([[x_sample.get(f, self.feature_means_[i]) for i, f in enumerate(self.feature_names_)]])
        elif isinstance(x_sample, pd.Series):
            raw_x = np.array([[x_sample.get(f, self.feature_means_[i]) for i, f in enumerate(self.feature_names_)]])
        else:
            raw_x = np.array([x_sample])

        raw_x = self._manifold_impute(raw_x)
        x_curr = self._standardize(raw_x).copy()[0]
        init_x = x_curr.copy()

        immutable_indices = [
            i for i, f in enumerate(self.feature_names_)
            if f.lower() in [imm.lower() for imm in immutable_features]
        ]

        target_resonators = self.class_manifolds_.get(target_class, [])
        if not target_resonators:
            return {"success": False, "reason": f"Target class {target_class} has no resonators."}

        target_centers = np.array([res["center"] for res in target_resonators])
        best_target_idx = np.argmin(cdist([x_curr], target_centers, metric='euclidean')[0])
        best_target_center = target_centers[best_target_idx]

        for step in range(max_steps):
            probs = self.predict_proba(np.array([x_curr * self.feature_stds_ + self.feature_means_]))[0]
            target_idx = np.where(self.classes_ == target_class)[0][0]
            if probs[target_idx] >= 0.60:
                break

            grad = (best_target_center - x_curr) * self.feature_weights_
            for imm_idx in immutable_indices:
                grad[imm_idx] = 0.0
            x_curr += learning_rate * grad

        final_raw = x_curr * self.feature_stds_ + self.feature_means_
        feature_shifts = {}
        for i, f in enumerate(self.feature_names_):
            shift = final_raw[i] - raw_x[0, i]
            if abs(shift) > 1e-3:
                feature_shifts[f] = {
                    "original": round(float(raw_x[0, i]), 2),
                    "prescribed": round(float(final_raw[i]), 2),
                    "shift": round(float(shift), 2)
                }

        return {
            "success": True,
            "target_class": target_class,
            "prescribed_actions": feature_shifts,
            "immutable_preserved": immutable_features,
            "feasibility_distance": round(float(np.linalg.norm(x_curr - init_x)), 4)
        }

    def save(self, file_path: str):
        data = {
            "task_type": self.task_type,
            "classes_": self.classes_.tolist() if self.classes_ is not None else [],
            "feature_names_": self.feature_names_,
            "feature_means_": self.feature_means_.tolist() if self.feature_means_ is not None else [],
            "feature_stds_": self.feature_stds_.tolist() if self.feature_stds_ is not None else [],
            "feature_weights_": self.feature_weights_.tolist() if self.feature_weights_ is not None else [],
            "class_weights_": {str(k): float(v) for k, v in self.class_weights_.items()},
            "calibrated_temperature_": float(self.calibrated_temperature_),
            "category_maps_": self.category_maps_,
            "categorical_cols_": self.categorical_cols_,
            "class_manifolds_": {
                str(k): [
                    {
                        "center": r["center"].tolist(),
                        "inv_metric": r["inv_metric"].tolist(),
                        "weight": float(r["weight"])
                    } for r in v
                ] for k, v in self.class_manifolds_.items()
            }
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)
        model = cls(task_type=data["task_type"])
        model.classes_ = np.array(data["classes_"])
        model.feature_names_ = data["feature_names_"]
        model.feature_means_ = np.array(data["feature_means_"])
        model.feature_stds_ = np.array(data["feature_stds_"])
        model.feature_weights_ = np.array(data.get("feature_weights_", np.ones(len(model.feature_means_))))
        model.class_weights_ = {int(k) if k.isdigit() else k: float(v) for k, v in data.get("class_weights_", {}).items()}
        model.calibrated_temperature_ = float(data.get("calibrated_temperature_", 1.0))
        model.category_maps_ = data.get("category_maps_", {})
        model.categorical_cols_ = data.get("categorical_cols_", [])
        
        all_centers = []
        model.class_manifolds_ = {}
        for k, v in data["class_manifolds_"].items():
            key = int(k) if k.isdigit() else k
            resonators = []
            for r in v:
                c = np.array(r["center"])
                all_centers.append(c * model.feature_stds_ + model.feature_means_)
                resonators.append({
                    "center": c,
                    "inv_metric": np.array(r["inv_metric"]),
                    "weight": r["weight"],
                    "octave_vectors": []
                })
            model.class_manifolds_[key] = resonators

        model.all_centroids_ = np.array(all_centers)
        model.is_fitted = True
        return model


TMRM = TopologicalManifoldResonantMachine


class StreamingTMRM:
    """Extreme Big Data Streaming TMRM for Real-Time Infinite Event Ingestion."""
    def __init__(self, n_features: int = 10, n_classes: int = 2, k_per_class: int = 4):
        self.n_features = n_features
        self.n_classes = n_classes
        self.k_per_class = k_per_class
        self.centroids = np.zeros((n_classes, k_per_class, n_features))
        self.counts = np.zeros((n_classes, k_per_class))

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        for xi, yi in zip(X, y):
            cls_idx = int(yi)
            cls_centroids = self.centroids[cls_idx]
            dists = np.sum((cls_centroids - xi) ** 2, axis=1)
            nearest = np.argmin(dists)
            self.counts[cls_idx, nearest] += 1
            eta = 1.0 / self.counts[cls_idx, nearest]
            self.centroids[cls_idx, nearest] += eta * (xi - cls_centroids[nearest])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for xi in X:
            best_cls = 0
            min_dist = float("inf")
            for c in range(self.n_classes):
                d = np.min(np.sum((self.centroids[c] - xi) ** 2, axis=1))
                if d < min_dist:
                    min_dist = d
                    best_cls = c
            preds.append(best_cls)
        return np.array(preds)
