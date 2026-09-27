import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error


class PrototypeTMRMRegressor:
    """
    TMRM v4.0 Continuous Topological Manifold Resonant Regressor
    """
    def __init__(self, n_resonators="auto", n_subspaces=4, harmonic_octaves=3, reg=1e-3, random_state=42):
        self.n_resonators = n_resonators
        self.n_subspaces = n_subspaces
        self.harmonic_octaves = harmonic_octaves
        self.reg = reg
        self.random_state = random_state

        self.ricci_curvature_ = 0.0
        self.metric_weights_ = (0.40, 0.35, 0.25)
        self.global_bandwidth_ = 1.0
        self.centroids_ = None
        self.resonators_ = []
        self.subspaces_ = []
        self.dual_weights_ = None
        self.feature_weights_ = None
        self.means_ = None
        self.stds_ = None

    def _estimate_ricci(self, X_sample, k=5, n_pairs=40):
        n = len(X_sample)
        if n < k + 2:
            return 0.0
        dists = cdist(X_sample, X_sample)
        curvs = []
        rng = np.random.RandomState(self.random_state)
        for _ in range(n_pairs):
            i = rng.randint(0, n)
            nbrs_i = np.argsort(dists[i])[1:k+1]
            j = nbrs_i[0]
            nbrs_j = np.argsort(dists[j])[1:k+1]
            cost = dists[np.ix_(nbrs_i, nbrs_j)]
            r, c = linear_sum_assignment(cost)
            w1 = cost[r, c].mean()
            d_ij = dists[i, j]
            if d_ij > 1e-6:
                curvs.append(1.0 - (w1 / d_ij))
        return float(np.median(curvs)) if len(curvs) > 0 else 0.0

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n_samples, n_features = X_arr.shape

        self.means_ = np.nanmean(X_arr, axis=0)
        self.stds_ = np.nanstd(X_arr, axis=0)
        self.stds_[self.stds_ == 0] = 1.0

        # Geometric soft-log scale shield
        X_clean = np.where(np.isnan(X_arr), self.means_, X_arr)
        z = (X_clean - self.means_) / self.stds_
        X_norm = np.sign(z) * np.log1p(np.abs(z))

        # Auto-Ricci Curvature Estimation
        subset = X_norm[:min(200, n_samples)]
        self.ricci_curvature_ = self._estimate_ricci(subset)

        # Dynamic Minkowski Mix based on Curvature
        if self.ricci_curvature_ > 0.15:
            # Spherical manifold: boost smooth Riemannian wave
            self.metric_weights_ = (0.60, 0.25, 0.15)
        elif self.ricci_curvature_ < 0.0:
            # Hyperbolic/Tree manifold: boost Chebyshev hyper-box
            self.metric_weights_ = (0.30, 0.45, 0.25)
        else:
            # Balanced Euclidean
            self.metric_weights_ = (0.40, 0.35, 0.25)

        # Bandwidth
        pw_dists = cdist(subset[:100], subset[:100])
        pos_d = pw_dists[pw_dists > 0]
        self.global_bandwidth_ = float(np.median(pos_d)) if len(pos_d) > 0 else 1.0

        # Feature relevance via correlation
        corrs = np.array([np.corrcoef(X_norm[:, i], y_arr)[0, 1] for i in range(n_features)])
        corrs = np.nan_to_num(corrs, nan=0.0)
        self.feature_weights_ = np.clip(np.abs(corrs) * 3.0 + 0.3, 0.2, 5.0)

        # Resonator Centroid Allocation via K-Means++
        k_res = max(6, min(40, int(np.sqrt(n_samples) * 1.5)))
        centers = [X_norm[rng.randint(0, n_samples)]]
        for _ in range(1, k_res):
            dists = cdist(X_norm, np.array(centers), metric='sqeuclidean').min(axis=1)
            probs = dists / (dists.sum() + 1e-12)
            centers.append(X_norm[rng.choice(n_samples, p=probs)])
        self.centroids_ = np.array(centers)

        # Cluster assignments and local tangent fitting
        assignments = cdist(X_norm, self.centroids_, metric='euclidean').argmin(axis=1)
        self.resonators_ = []

        for k in range(k_res):
            pts = X_norm[assignments == k]
            pts_y = y_arr[assignments == k]
            if len(pts) < 4:
                pts = X_norm
                pts_y = y_arr

            c_k = self.centroids_[k]
            diff = (pts - c_k) * np.sqrt(self.feature_weights_)
            cov_k = (diff.T @ diff) / max(1, len(pts) - 1)
            cov_reg = cov_k + np.eye(n_features) * (self.reg * self.global_bandwidth_)

            inv_m = np.linalg.pinv(cov_reg)
            eigenvals, eigenvecs = np.linalg.eigh(cov_reg)
            top_eig = eigenvecs[:, -1]
            base_freq = 2.0 * np.pi / (np.sqrt(max(1e-4, eigenvals[-1])) + 1e-4)

            octaves = []
            for oct_idx in range(1, self.harmonic_octaves + 1):
                octaves.append({
                    "freq_vec": top_eig * (base_freq * oct_idx),
                    "phase": float(rng.uniform(0, np.pi)),
                    "weight": 1.0 / oct_idx
                })

            # Local Tangent Taylor fit
            A = np.column_stack([np.ones(len(pts)), pts - c_k])
            ridge = np.eye(A.shape[1]) * 1e-2
            ridge[0, 0] = 0.0
            try:
                beta = np.linalg.solve(A.T @ A + ridge, A.T @ pts_y)
            except np.linalg.LinAlgError:
                beta = np.zeros(n_features + 1)
                beta[0] = np.mean(pts_y)

            self.resonators_.append({
                "center": c_k,
                "inv_metric": inv_m,
                "beta_0": float(beta[0]),
                "beta_tangent": beta[1:],
                "octaves": octaves,
                "weight": float(len(pts) / n_samples)
            })

        # Build Multi-Faceted Minkowski Spectral Basis
        Phi_primary = self._compute_primary_basis(X_norm)

        # Topological Subspace Wave-Packets for Regression
        sub_dim = max(2, int(np.sqrt(n_features) * 1.5))
        self.subspaces_ = []
        Phi_sub = []
        if sub_dim < n_features and self.n_subspaces > 1:
            gamma_sub = 1.0 / (2.0 * (self.global_bandwidth_ ** 2))
            for _ in range(self.n_subspaces):
                feats = rng.choice(n_features, size=sub_dim, replace=False)
                X_s = (X_norm * np.sqrt(self.feature_weights_))[:, feats]
                C_s = (self.centroids_ * np.sqrt(self.feature_weights_))[:, feats]
                D2 = cdist(X_s, C_s, metric='sqeuclidean')
                D_cheb = cdist(X_s, C_s, metric='chebyshev')
                phi_s = 0.50 * np.exp(-gamma_sub * D2) + 0.50 * np.exp(-0.8 * D_cheb)
                Phi_sub.append(phi_s)
                self.subspaces_.append((feats, gamma_sub))
            Phi_full = np.hstack([Phi_primary] + Phi_sub)
        else:
            Phi_full = Phi_primary

        # Closed-Form Dual Ridge Potential Superposition for Regression
        lambda_reg = 0.02 * np.mean(np.diag(Phi_full.T @ Phi_full))
        self.dual_weights_ = np.linalg.solve(
            Phi_full.T @ Phi_full + np.eye(Phi_full.shape[1]) * lambda_reg,
            Phi_full.T @ y_arr
        )
        return self

    def _compute_primary_basis(self, X_norm):
        n_samples = len(X_norm)
        n_res = len(self.resonators_)
        Phi = np.zeros((n_samples, n_res))
        w_l2, w_cheb, w_l1 = self.metric_weights_

        for j, res in enumerate(self.resonators_):
            delta = (X_norm - res["center"]) * np.sqrt(self.feature_weights_)
            d_l2 = np.clip(np.sum((delta @ res["inv_metric"]) * delta, axis=1), 0, 100.0)
            d_cheb = np.max(np.abs(delta), axis=1)
            d_l1 = np.sum(np.abs(delta), axis=1) / np.sqrt(len(self.feature_weights_))

            psi = 1.0
            for oct_info in res["octaves"]:
                proj = np.dot(delta, oct_info["freq_vec"]) + oct_info["phase"]
                psi += 0.05 * oct_info["weight"] * np.cos(proj)

            res_phi = (w_l2 * np.exp(-0.5 * d_l2) + w_cheb * np.exp(-0.75 * d_cheb) + w_l1 * np.exp(-0.9 * d_l1)) * psi
            Phi[:, j] = res_phi
        return Phi

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        X_clean = np.where(np.isnan(X_arr), self.means_, X_arr)
        z = (X_clean - self.means_) / self.stds_
        X_norm = np.sign(z) * np.log1p(np.abs(z))

        # 1. Global Spectral Wave Superposition
        Phi_primary = self._compute_primary_basis(X_norm)
        if len(self.subspaces_) > 0:
            Phi_sub = []
            for feats, gamma_sub in self.subspaces_:
                X_s = (X_norm * np.sqrt(self.feature_weights_))[:, feats]
                C_s = (self.centroids_ * np.sqrt(self.feature_weights_))[:, feats]
                D2 = cdist(X_s, C_s, metric='sqeuclidean')
                D_cheb = cdist(X_s, C_s, metric='chebyshev')
                phi_s = 0.50 * np.exp(-gamma_sub * D2) + 0.50 * np.exp(-0.8 * D_cheb)
                Phi_sub.append(phi_s)
            Phi_full = np.hstack([Phi_primary] + Phi_sub)
        else:
            Phi_full = Phi_primary

        spectral_pred = Phi_full @ self.dual_weights_

        # 2. Local Tangent Taylor Field
        n_samples = len(X_norm)
        tangent_num = np.zeros(n_samples)
        tangent_denom = np.zeros(n_samples)

        for res in self.resonators_:
            c_k = res["center"]
            delta = X_norm - c_k
            local_val = res["beta_0"] + delta @ res["beta_tangent"]
            delta_w = delta * np.sqrt(self.feature_weights_)
            d_l2 = np.clip(np.sum((delta_w @ res["inv_metric"]) * delta_w, axis=1), 0, 100.0)
            kw = res["weight"] * np.exp(-0.5 * d_l2)
            tangent_num += kw * local_val
            tangent_denom += kw

        tangent_denom = np.maximum(tangent_denom, 1e-10)
        tangent_pred = tangent_num / tangent_denom

        # Smooth Unified Field: 65% Spectral Superposition + 35% Local Tangent Field
        return 0.65 * spectral_pred + 0.35 * tangent_pred


# Execute Head-to-Head Benchmark on Real Datasets
print("=" * 70)
print("BENCHMARK 1: Diabetes Progression Dataset (Clinical Continuous Target)")
d = load_diabetes()
X_tr, X_te, y_tr, y_te = train_test_split(d.data, d.target, test_size=0.25, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
tmrm = PrototypeTMRMRegressor(random_state=42).fit(X_tr, y_tr)

rf_r2 = r2_score(y_te, rf.predict(X_te))
gb_r2 = r2_score(y_te, gb.predict(X_te))
tmrm_r2 = r2_score(y_te, tmrm.predict(X_te))

print(f"Random Forest Regressor R^2: {rf_r2:.4f}")
print(f"Gradient Boosting Regressor R^2: {gb_r2:.4f}")
print(f"TMRM v4.0 Regressor R^2:     {tmrm_r2:.4f} (Ricci Curvature: {tmrm.ricci_curvature_:.3f})")
print(f"Minkowski Dynamic Mix: {tmrm.metric_weights_}")

print("\n" + "=" * 70)
print("BENCHMARK 2: California Housing Dataset (2,000 Sample Evaluation)")
cal = fetch_california_housing()
X_c, y_c = cal.data[:2000], cal.target[:2000]
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_c, y_c, test_size=0.25, random_state=42)

rf_c = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr_c, y_tr_c)
tmrm_c = PrototypeTMRMRegressor(random_state=42).fit(X_tr_c, y_tr_c)

print(f"Random Forest California R^2: {r2_score(y_te_c, rf_c.predict(X_te_c)):.4f}")
print(f"TMRM v4.0 California R^2:     {r2_score(y_te_c, tmrm_c.predict(X_te_c)):.4f} (Ricci Curvature: {tmrm_c.ricci_curvature_:.3f})")
