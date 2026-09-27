import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import fetch_california_housing, make_friedman1, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load California Housing subset (2500 samples)
cal = fetch_california_housing()
X_c, y_c = cal.data[:2500], cal.target[:2500]
X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_c, test_size=0.25, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
rf_score = r2_score(y_te, rf.predict(X_te))
print(f"Random Forest Target R^2: {rf_score:.4f}")

# Test Multi-Scale Cascade Formulation
class MultiScaleWaveletTMRM:
    def __init__(self, n_subspaces=16, n_scales=2, random_state=42):
        self.n_subspaces = n_subspaces
        self.n_scales = n_scales
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.stds[self.stds == 0] = 1.0
        z = (X - self.means) / self.stds
        X_norm = np.sign(z) * np.log1p(np.abs(z))

        n_samples, n_features = X_norm.shape
        self.feature_weights = np.ones(n_features)
        corrs = np.array([np.corrcoef(X_norm[:, i], y)[0, 1] for i in range(n_features)])
        corrs = np.nan_to_num(corrs, nan=0.0)
        self.feature_weights = np.clip(np.abs(corrs) * 3.0 + 0.3, 0.2, 5.0)

        # Scale 1: Macro Manifold Resonators (Coarse)
        k_macro = max(8, min(25, int(np.sqrt(n_samples) * 0.8)))
        # Scale 2: Micro Manifold Resonators (Fine)
        k_micro = max(16, min(60, int(np.sqrt(n_samples) * 1.8)))

        self.scales = []
        for k_res, scale_weight in [(k_macro, 1.0), (k_micro, 0.5)]:
            # K-means++ selection
            centers = [X_norm[rng.randint(0, n_samples)]]
            for _ in range(1, k_res):
                dists = cdist(X_norm, np.array(centers), metric='sqeuclidean').min(axis=1)
                probs = dists / (dists.sum() + 1e-12)
                centers.append(X_norm[rng.choice(n_samples, p=probs)])
            centers = np.array(centers)
            self.scales.append({"centers": centers, "k": k_res})

        # Build Multi-Faceted Basis across all scales
        all_centers = np.vstack([s["centers"] for s in self.scales])
        self.all_centroids = all_centers

        # Subspaces (Multi-manifold dimensional bagging)
        sub_dim = max(2, int(np.sqrt(n_features) * 1.5))
        self.subspaces = []
        pw_d = cdist(X_norm[:200], X_norm[:200])
        bandwidth = float(np.median(pw_d[pw_d > 0])) if np.any(pw_d > 0) else 1.0
        self.bandwidth = bandwidth
        gamma_sub = 1.0 / (2.0 * (bandwidth ** 2))

        Phi_list = []
        # Primary Multi-scale basis
        D2 = cdist(X_norm * np.sqrt(self.feature_weights), all_centers * np.sqrt(self.feature_weights), metric='sqeuclidean')
        D_cheb = cdist(X_norm * np.sqrt(self.feature_weights), all_centers * np.sqrt(self.feature_weights), metric='chebyshev')
        D_l1 = cdist(X_norm * np.sqrt(self.feature_weights), all_centers * np.sqrt(self.feature_weights), metric='cityblock') / np.sqrt(n_features)

        Phi_primary = 0.40 * np.exp(-gamma_sub * D2) + 0.40 * np.exp(-0.8 * D_cheb) + 0.20 * np.exp(-0.9 * D_l1)
        Phi_list.append(Phi_primary)

        # 16-32 Subspace wave-packets
        for _ in range(self.n_subspaces):
            feats = rng.choice(n_features, size=sub_dim, replace=False)
            Xs = (X_norm * np.sqrt(self.feature_weights))[:, feats]
            Cs = (all_centers * np.sqrt(self.feature_weights))[:, feats]
            d2_s = cdist(Xs, Cs, metric='sqeuclidean')
            d_cheb_s = cdist(Xs, Cs, metric='chebyshev')
            phi_s = 0.50 * np.exp(-gamma_sub * d2_s) + 0.50 * np.exp(-0.8 * d_cheb_s)
            Phi_list.append(phi_s)
            self.subspaces.append((feats, gamma_sub))

        Phi_full = np.hstack(Phi_list)

        # Dual solve with adaptive Tikhonov ridge regularization
        lambda_reg = 0.005 * np.mean(np.diag(Phi_full.T @ Phi_full))
        self.w_dual = np.linalg.solve(Phi_full.T @ Phi_full + np.eye(Phi_full.shape[1]) * lambda_reg, Phi_full.T @ y)

        # Local Tangent Taylor field
        assignments = cdist(X_norm, all_centers).argmin(axis=1)
        self.tangent_resonators = []
        for k in range(len(all_centers)):
            pts = X_norm[assignments == k]
            pts_y = y[assignments == k]
            if len(pts) < 4:
                pts = X_norm
                pts_y = y
            ck = all_centers[k]
            A = np.column_stack([np.ones(len(pts)), pts - ck])
            ridge = np.eye(A.shape[1]) * 1e-2
            ridge[0, 0] = 0.0
            try:
                beta = np.linalg.solve(A.T @ A + ridge, A.T @ pts_y)
            except:
                beta = np.zeros(n_features + 1)
                beta[0] = np.mean(pts_y)
            self.tangent_resonators.append({"center": ck, "beta_0": beta[0], "beta": beta[1:], "weight": len(pts)/n_samples})

        return self

    def predict(self, X):
        z = (X - self.means) / self.stds
        X_norm = np.sign(z) * np.log1p(np.abs(z))
        n_features = X_norm.shape[1]

        D2 = cdist(X_norm * np.sqrt(self.feature_weights), self.all_centroids * np.sqrt(self.feature_weights), metric='sqeuclidean')
        D_cheb = cdist(X_norm * np.sqrt(self.feature_weights), self.all_centroids * np.sqrt(self.feature_weights), metric='chebyshev')
        D_l1 = cdist(X_norm * np.sqrt(self.feature_weights), self.all_centroids * np.sqrt(self.feature_weights), metric='cityblock') / np.sqrt(n_features)
        gamma = 1.0 / (2.0 * (self.bandwidth ** 2))

        Phi_primary = 0.40 * np.exp(-gamma * D2) + 0.40 * np.exp(-0.8 * D_cheb) + 0.20 * np.exp(-0.9 * D_l1)
        Phi_list = [Phi_primary]

        for feats, gamma_sub in self.subspaces:
            Xs = (X_norm * np.sqrt(self.feature_weights))[:, feats]
            Cs = (self.all_centroids * np.sqrt(self.feature_weights))[:, feats]
            d2_s = cdist(Xs, Cs, metric='sqeuclidean')
            d_cheb_s = cdist(Xs, Cs, metric='chebyshev')
            phi_s = 0.50 * np.exp(-gamma_sub * d2_s) + 0.50 * np.exp(-0.8 * d_cheb_s)
            Phi_list.append(phi_s)

        Phi_full = np.hstack(Phi_list)
        spectral_pred = Phi_full @ self.w_dual

        # Tangent field
        num = np.zeros(len(X_norm))
        denom = np.zeros(len(X_norm))
        for res in self.tangent_resonators:
            ck = res["center"]
            delta = X_norm - ck
            local_y = res["beta_0"] + delta @ res["beta"]
            d = np.sum((delta * np.sqrt(self.feature_weights))**2, axis=1)
            kw = res["weight"] * np.exp(-gamma * d)
            num += kw * local_y
            denom += kw
        denom = np.maximum(denom, 1e-10)
        tangent_pred = num / denom

        return 0.70 * spectral_pred + 0.30 * tangent_pred

tmrm_multi = MultiScaleWaveletTMRM(n_subspaces=16).fit(X_tr, y_tr)
tmrm_score = r2_score(y_te, tmrm_multi.predict(X_te))
print(f"Multi-Scale Wavelet TMRM R^2: {tmrm_score:.4f}")
