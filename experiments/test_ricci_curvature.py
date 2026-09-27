import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error


def estimate_ricci_curvature(X_sample: np.ndarray, k: int = 5) -> float:
    """
    Computes an empirical discrete Ollivier-Ricci curvature proxy on the Riemannian k-NN graph.
    kappa(x, y) = 1 - W1(m_x, m_y) / d(x, y)
    Where W1 is estimated via local neighborhood distance transport.
    """
    n = len(X_sample)
    if n < k + 2:
        return 0.0

    dists = cdist(X_sample, X_sample, metric='euclidean')
    curvatures = []

    # Sample random pairs
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(n, size=min(100, n), replace=False)

    for i in sample_indices:
        # Find k nearest neighbors of i
        nbrs_i = np.argsort(dists[i])[1:k+1]
        j = nbrs_i[0] # Nearest neighbor
        d_ij = dists[i, j]
        if d_ij < 1e-6:
            continue

        nbrs_j = np.argsort(dists[j])[1:k+1]

        # Cost matrix between neighborhood i and neighborhood j
        sub_cost = dists[np.ix_(nbrs_i, nbrs_j)]
        # Greedy 1-Wasserstein approximation (average distance between matched nearest neighbors)
        w1_approx = np.mean(np.min(sub_cost, axis=1))

        kappa = 1.0 - (w1_approx / d_ij)
        curvatures.append(kappa)

    mean_kappa = float(np.median(curvatures)) if len(curvatures) > 0 else 0.0
    return np.clip(mean_kappa, -1.0, 1.0)


print("Testing discrete Ollivier-Ricci curvature...")
X_sphere = np.random.randn(200, 5)
X_sphere /= np.linalg.norm(X_sphere, axis=1, keepdims=True) # Spherical manifold
k_sphere = estimate_ricci_curvature(X_sphere)
print(f"Sphere curvature (expected positive): {k_sphere:.3f}")

X_tree = np.zeros((200, 5))
for idx in range(200):
    depth = idx % 5
    X_tree[idx, depth] = idx // 5 # Branching tree structure
k_tree = estimate_ricci_curvature(X_tree)
print(f"Tree/Grid curvature (expected negative): {k_tree:.3f}")
