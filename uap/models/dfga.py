"""
Dynamic Feature-Graph Algorithm (DFGA) - Core Machine Learning Model.
Novel Graph-Tabular Architecture designed from scratch:
  1. Input Transformation: Row to Attributed Feature Graph (Zero-Imputation tolerant).
  2. Information Passing: Instance-conditioned message passing across dynamic edges.
  3. Graph Aggregation: Attention-weighted readout into target predictions.
  4. Custom Loss: Task loss + Causal Directionality Penalty + Graph Sparsity.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class FeatureGraphNetwork(nn.Module):
    """
    PyTorch Neural Core of the Dynamic Feature-Graph Algorithm (DFGA).
    """

    def __init__(
        self,
        n_features: int,
        embed_dim: int = 16,
        n_hops: int = 2,
        is_classification: bool = True,
        forbidden_causal_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.n_hops = n_hops
        self.is_classification = is_classification

        # 1. Feature Identity & Value Encoders (Node Initialization)
        # Learnable identity embeddings for each feature
        self.feature_embeddings = nn.Parameter(torch.randn(n_features, embed_dim) * 0.05)
        # Scalar to vector value projector per feature
        self.value_weights = nn.Parameter(torch.randn(n_features, embed_dim) * 0.05)
        self.value_biases = nn.Parameter(torch.zeros(n_features, embed_dim))
        # Missing value embedding (Zero-Imputation handler)
        self.missing_embedding = nn.Parameter(torch.zeros(embed_dim))

        # 2. Dynamic Adjacency Attention Projectors (Edge Builders)
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # 3. Message Passing Update Layers
        self.update_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
            for _ in range(n_hops)
        ])

        # 4. Graph Readout & Final Aggregator
        self.readout_attention = nn.Linear(embed_dim, 1)
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

        # 5. Causal Mask (Forbidden Directed Connections)
        if forbidden_causal_mask is not None:
            self.register_buffer("forbidden_mask", forbidden_causal_mask)
        else:
            self.register_buffer("forbidden_mask", torch.zeros(n_features, n_features))

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, n_features) - continuous values
        mask: (batch_size, n_features) - 1 if present, 0 if missing
        Returns:
            predictions: (batch_size, 1)
            adjacency_matrices: (batch_size, n_features, n_features)
        """
        batch_size = x.shape[0]

        if mask is None:
            mask = torch.ones_like(x)

        # Step 1: Input Transformation (Row to Graph Nodes)
        # Reshape for broadcasting: (batch_size, n_features, embed_dim)
        x_expanded = x.unsqueeze(-1)  # (B, D, 1)
        w_expanded = self.value_weights.unsqueeze(0)  # (1, D, embed_dim)
        b_expanded = self.value_biases.unsqueeze(0)  # (1, D, embed_dim)
        id_expanded = self.feature_embeddings.unsqueeze(0)  # (1, D, embed_dim)

        # Base node embedding: h_i = W_val * x_i + b_val + e_id
        val_embed = x_expanded * w_expanded + b_expanded + id_expanded

        # Handle missingness via mask
        mask_expanded = mask.unsqueeze(-1)  # (B, D, 1)
        missing_expanded = self.missing_embedding.view(1, 1, -1).expand(batch_size, self.n_features, -1)
        H = torch.where(mask_expanded > 0.5, val_embed, missing_expanded)  # (B, D, embed_dim)

        # Step 2: Information Passing Mechanism (Graph Message Passing)
        # Compute Dynamic Adjacency Matrix A_ij per sample
        Q = self.query_proj(H)  # (B, D, embed_dim)
        K = self.key_proj(H)    # (B, D, embed_dim)
        scale = np.sqrt(self.embed_dim)

        # Raw edge scores: (B, D, D)
        scores = torch.bmm(Q, K.transpose(1, 2)) / scale
        # Row-wise softmax gives normalized directed influence weights A_ij
        A = torch.softmax(scores, dim=-1)

        # Multi-Hop Message Passing
        H_curr = H
        for hop in range(self.n_hops):
            V = self.value_proj(H_curr)  # (B, D, embed_dim)
            # Messages aggregated from neighbors: M = A * V -> (B, D, embed_dim)
            messages = torch.bmm(A, V)
            # Node state update
            cat_feat = torch.cat([H_curr, messages], dim=-1)  # (B, D, 2*embed_dim)
            H_curr = self.update_layers[hop](cat_feat)

        # Step 3: Graph Aggregation & Readout
        attn_weights = torch.softmax(self.readout_attention(H_curr), dim=1)  # (B, D, 1)
        graph_embedding = torch.sum(attn_weights * H_curr, dim=1)  # (B, embed_dim)

        # Step 4: Final Output Prediction
        out = self.output_head(graph_embedding)
        if self.is_classification:
            out = torch.sigmoid(out)

        return out.squeeze(-1), A


class DynamicFeatureGraphModel(BaseEstimator):
    """
    Production Scikit-Learn Estimator Wrapper for DFGA.
    Supports fit, predict, predict_proba, and learned graph structure extraction.
    """

    def __init__(
        self,
        embed_dim: int = 16,
        n_hops: int = 2,
        epochs: int = 40,
        lr: float = 0.01,
        batch_size: int = 64,
        lambda_causal: float = 0.05,
        lambda_sparse: float = 0.001,
        is_classification: bool = True,
        forbidden_causal_pairs: Optional[List[Tuple[int, int]]] = None,
        random_state: int = 42,
    ):
        self.embed_dim = embed_dim
        self.n_hops = n_hops
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.lambda_causal = lambda_causal
        self.lambda_sparse = lambda_sparse
        self.is_classification = is_classification
        self.forbidden_causal_pairs = forbidden_causal_pairs or []
        self.random_state = random_state

        self.network: Optional[FeatureGraphNetwork] = None
        self.feature_names_: List[str] = []
        self.n_features_: int = 0
        self.fitted_ = False

    def _build_forbidden_mask(self, n_features: int) -> torch.Tensor:
        mask = torch.zeros(n_features, n_features)
        for src, dst in self.forbidden_causal_pairs:
            if src < n_features and dst < n_features:
                mask[src, dst] = 1.0  # Penalize edge src -> dst
        return mask

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
            self.feature_names_ = [f"feat_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y, dtype=np.float32)
        self.n_features_ = X_arr.shape[1]

        # Build mask for missing values (Zero-Imputation capability)
        mask_arr = (~np.isnan(X_arr)).astype(np.float32)
        X_clean = np.nan_to_num(X_arr, nan=0.0)

        forbidden_mask = self._build_forbidden_mask(self.n_features_)

        self.network = FeatureGraphNetwork(
            n_features=self.n_features_,
            embed_dim=self.embed_dim,
            n_hops=self.n_hops,
            is_classification=self.is_classification,
            forbidden_causal_mask=forbidden_mask,
        )

        optimizer = optim.AdamW(self.network.parameters(), lr=self.lr, weight_decay=1e-4)

        if self.is_classification:
            task_criterion = nn.BCELoss()
        else:
            task_criterion = nn.MSELoss()

        X_t = torch.tensor(X_clean, dtype=torch.float32)
        mask_t = torch.tensor(mask_arr, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_t, mask_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.network.train()
        for epoch in range(self.epochs):
            for batch_x, batch_m, batch_y in loader:
                optimizer.zero_grad()
                preds, A = self.network(batch_x, batch_m)

                # Primary task loss
                l_task = task_criterion(preds, batch_y)

                # Causal violation loss: penalize forbidden edge connections
                l_causal = torch.sum(self.network.forbidden_mask.unsqueeze(0) * (A ** 2))

                # Graph sparsity loss: penalize excessive edge connectivity
                l_sparse = torch.mean(torch.sum(A, dim=(-1, -2)))

                total_loss = l_task + (self.lambda_causal * l_causal) + (self.lambda_sparse * l_sparse)

                total_loss.backward()
                optimizer.step()

        self.fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        mask_arr = (~np.isnan(X_arr)).astype(np.float32)
        X_clean = np.nan_to_num(X_arr, nan=0.0)

        self.network.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_clean, dtype=torch.float32)
            mask_t = torch.tensor(mask_arr, dtype=torch.float32)
            preds, _ = self.network(X_t, mask_t)
            preds_np = preds.cpu().numpy()

        if self.is_classification:
            return (preds_np >= 0.5).astype(int)
        return preds_np

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_classification:
            raise ValueError("predict_proba is only available for classification.")
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        mask_arr = (~np.isnan(X_arr)).astype(np.float32)
        X_clean = np.nan_to_num(X_arr, nan=0.0)

        self.network.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_clean, dtype=torch.float32)
            mask_t = torch.tensor(mask_arr, dtype=torch.float32)
            p1, _ = self.network(X_t, mask_t)
            p1_np = p1.cpu().numpy().reshape(-1, 1)
            p0_np = 1.0 - p1_np
            return np.hstack([p0_np, p1_np])

    def get_learned_adjacency(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Extracts the instance-level or average Adjacency Matrix A (relationship graph between features).
        """
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        mask_arr = (~np.isnan(X_arr)).astype(np.float32)
        X_clean = np.nan_to_num(X_arr, nan=0.0)

        self.network.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_clean, dtype=torch.float32)
            mask_t = torch.tensor(mask_arr, dtype=torch.float32)
            _, A = self.network(X_t, mask_t)
            return A.cpu().numpy()
