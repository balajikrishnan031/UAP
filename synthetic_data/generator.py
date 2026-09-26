"""
Synthetic Dataset Generator for UAP Benchmark Testbeds.
Simulates structural challenges: Severe Class Imbalance, Outlier Corruption,
High Non-Linearity, and High-Dimensional Noise.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


class UAPSyntheticDataGenerator:
    """
    Synthetic Dataset Generator designed to test the UAP Framework against baselines.
    Generates controlled data distributions with:
    - Extreme Class Imbalance (Tests Module 1 & Module 2 Resampling)
    - Severe Feature & Target Outliers (Tests Module 2 AFWE Weighting)
    - Complex Non-Linear Interactions (Tests Module 2 Routing & Module 3 HPO)
    - High-Dimensional Noise Features (Tests Feature Selection & Regularization)
    """

    @staticmethod
    def generate_imbalanced_non_linear_classification(
        n_samples: int = 3000,
        n_features: int = 12,
        minority_ratio: float = 0.05,
        label_noise: float = 0.03,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Generates a non-linear binary classification dataset with controlled class imbalance
        and label flipping noise.
        """
        np.random.seed(random_state)

        # 1. Base classification setup
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, int(n_features * 0.6)),
            n_redundant=max(1, int(n_features * 0.2)),
            weights=[1.0 - minority_ratio, minority_ratio],
            flip_y=label_noise,
            random_state=random_state,
        )

        # 2. Inject strong non-linear transformations into feature space
        X[:, 0] = np.sin(X[:, 0] * np.pi)
        X[:, 1] = (X[:, 1] ** 2) - (X[:, 2] * X[:, 3])
        X[:, 4] = np.exp(np.clip(X[:, 4], -2, 2))

        # 3. Create DataFrame
        feature_names = [f"feature_{i+1}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        return df

    @staticmethod
    def generate_noisy_outlier_regression(
        n_samples: int = 3000,
        n_features: int = 10,
        gaussian_noise_std: float = 1.5,
        outlier_ratio: float = 0.08,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Generates a non-linear regression dataset with Gaussian feature noise and 
        extreme target outliers (impulse noise).
        """
        np.random.seed(random_state)

        # 1. Draw features from uniform distribution
        X = np.random.uniform(-3.0, 3.0, size=(n_samples, n_features))

        # 2. Define complex non-linear target response function
        # y = sin(2*x0) + cos(x1) + 0.5*x2^2 + 1.5*(x3 * x4)
        y = (
            np.sin(2.0 * X[:, 0])
            + np.cos(X[:, 1])
            + 0.5 * (X[:, 2] ** 2)
            + 1.5 * (X[:, 3] * X[:, 4])
        )

        # 3. Add Gaussian measurement noise
        y += np.random.normal(0, gaussian_noise_std, size=n_samples)

        # 4. Inject extreme target outliers (Outlier Severity Index test)
        n_outliers = int(n_samples * outlier_ratio)
        outlier_indices = np.random.choice(n_samples, size=n_outliers, replace=False)
        outlier_magnitude = np.random.uniform(15.0, 35.0, size=n_outliers)
        outlier_signs = np.random.choice([-1, 1], size=n_outliers)

        y[outlier_indices] += outlier_magnitude * outlier_signs

        # 5. Create DataFrame
        feature_names = [f"feature_{i+1}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        return df

    @staticmethod
    def generate_high_dimensional_noisy(
        n_samples: int = 2000,
        n_informative: int = 10,
        n_noise_features: int = 50,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Generates a high-dimensional dataset where noise features drastically outnumber
        informative features (DR > 0.05). Tests AFWE feature discounting.
        """
        np.random.seed(random_state)

        # 1. Informative features
        X_info, y = make_classification(
            n_samples=n_samples,
            n_features=n_informative,
            n_informative=n_informative,
            n_redundant=0,
            random_state=random_state,
        )

        # 2. Uninformative random noise features (Curse of Dimensionality)
        X_noise = np.random.normal(0, 1, size=(n_samples, n_noise_features))

        # 3. Combine matrices
        X_total = np.hstack([X_info, X_noise])
        feature_names = [f"informative_{i+1}" for i in range(n_informative)] + [
            f"noise_{i+1}" for i in range(n_noise_features)
        ]

        df = pd.DataFrame(X_total, columns=feature_names)
        df["target"] = y
        return df
