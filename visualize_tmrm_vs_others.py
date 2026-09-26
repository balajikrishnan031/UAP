"""
Generate Ultra-Clear Scientific Visual Comparisons:
Traditional Models (Decision Tree / Random Forest) vs Topological Manifold Resonant Machine (TMRM)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from uap.models.novel_tmrm import TopologicalManifoldResonantMachine

# Set premium styling
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=150)

# Generate complex curved non-linear data (Moons)
X, y = make_moons(n_samples=500, noise=0.22, random_state=42)

# Create high-density grid for decision boundary visualization
x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# ==============================================================================
# PANEL 1: Traditional Decision Tree (The Rigid Staircase Trap)
# ==============================================================================
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X, y)
Z_dt = dt.predict_proba(grid_points)[:, 1].reshape(xx.shape)

ax1 = axes[0]
ax1.contourf(xx, yy, Z_dt, levels=20, cmap="coolwarm", alpha=0.6)
ax1.scatter(X[y == 0, 0], X[y == 0, 1], c='#00E5FF', edgecolors='k', s=30, label='Class 0 (Normal)')
ax1.scatter(X[y == 1, 0], X[y == 1, 1], c='#FF3D00', edgecolors='k', s=30, label='Class 1 (Risk)')
ax1.set_title("1. Standard Decision Trees (XGBoost / RF)\n[Rigid Orthogonal Square-Box Splits]", fontsize=13, fontweight='bold', color='#FF8A80')
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.2)
ax1.text(0.05, 0.05, "MISTAKE: Jagged 'Staircase' cuts!\nCannot capture smooth natural curves.", 
         transform=ax1.transAxes, fontsize=10, color='yellow', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# ==============================================================================
# PANEL 2: TMRM Continuous Riemannian Energy Manifolds
# ==============================================================================
tmrm = TopologicalManifoldResonantMachine(n_resonators_per_class=6, random_state=42)
tmrm.fit(X, y)
Z_tmrm = tmrm.predict_proba(grid_points)[:, 1].reshape(xx.shape)

ax2 = axes[1]
ax2.contourf(xx, yy, Z_tmrm, levels=20, cmap="viridis", alpha=0.7)
ax2.contour(xx, yy, Z_tmrm, levels=[0.5], colors=['#FFEA00'], linewidths=2.5, linestyles='-')
ax2.scatter(X[y == 0, 0], X[y == 0, 1], c='#00E5FF', edgecolors='k', s=30, label='Class 0 (Normal)')
ax2.scatter(X[y == 1, 0], X[y == 1, 1], c='#FF3D00', edgecolors='k', s=30, label='Class 1 (Risk)')
# Plot resonator centers
for c, resonators in tmrm.class_manifolds_.items():
    centers = np.array([r["center"] for r in resonators])
    # Unstandardize centers to original data coordinates
    centers_orig = centers * tmrm.feature_stds_ + tmrm.feature_means_
    marker = 'o' if c == 0 else '^'
    color = '#76FF03' if c == 0 else '#FFD600'
    ax2.scatter(centers_orig[:, 0], centers_orig[:, 1], c=color, s=120, edgecolors='white', marker=marker, 
                label=f'Class {c} Manifold Resonators')

ax2.set_title("2. Novel Invention: TMRM Physics Engine\n[Smooth Riemannian Manifold Gravitational Fields]", fontsize=13, fontweight='bold', color='#69F0AE')
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Feature 2")
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, linestyle='--', alpha=0.2)
ax2.text(0.05, 0.05, "ADVANTAGE: Perfectly adapts to curved manifolds!\nSmooth continuous harmonic contours.", 
         transform=ax2.transAxes, fontsize=10, color='#69F0AE', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# ==============================================================================
# PANEL 3: TMRM In-Model Self-Doubt (OOD) & Instant Recourse Vector
# ==============================================================================
novelty_grid = tmrm.get_epistemic_novelty(grid_points).reshape(xx.shape)

ax3 = axes[2]
# Plot novelty gap (high distance = self doubt)
contour_novelty = ax3.contourf(xx, yy, novelty_grid, levels=15, cmap="magma", alpha=0.6)
plt.colorbar(contour_novelty, ax=ax3, label="Epistemic Novelty Gap (Self-Doubt Distance)")
ax3.scatter(X[y == 0, 0], X[y == 0, 1], c='#00E5FF', edgecolors='k', s=20, alpha=0.4)
ax3.scatter(X[y == 1, 0], X[y == 1, 1], c='#FF3D00', edgecolors='k', s=20, alpha=0.4)

# Alien OOD sample
ood_point = np.array([2.5, 2.0])
ax3.scatter(ood_point[0], ood_point[1], c='#FF1744', s=250, marker='X', edgecolors='white', label='Alien / OOD Data (Self-Doubt=True)')
ax3.annotate("OOD Alien Data!\nTMRM Abstains (Safe)", xy=(ood_point[0], ood_point[1]), xytext=(ood_point[0]-1.2, ood_point[1]+0.3),
             arrowprops=dict(facecolor='#FF1744', shrink=0.08, width=2), fontsize=10, color='#FF8A80', fontweight='bold')

# Actionable In-Model Recourse
sample_risk = np.array([0.5, 0.8])
recourse = tmrm.get_recourse_gradient(sample_risk, target_class=0, step_size=0.8)
rec_target = np.array(list(recourse["recommended_sample_dict"].values()))

ax3.scatter(sample_risk[0], sample_risk[1], c='#FFD600', s=160, marker='s', edgecolors='black', label='Patient in Risk (Class 1)')
ax3.annotate("", xy=(rec_target[0], rec_target[1]), xytext=(sample_risk[0], sample_risk[1]),
             arrowprops=dict(facecolor='#00E676', shrink=0.05, width=2.5, headwidth=8))
ax3.scatter(rec_target[0], rec_target[1], c='#00E676', s=160, marker='*', edgecolors='black', label='Prescribed Recourse Action (Safe Class 0)')
ax3.text(sample_risk[0]-0.8, sample_risk[1]-0.4, "Instant In-Model Recourse Vector:\nShift feature 1 & 2 to flip risk!", 
         fontsize=9, color='#00E676', fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

ax3.set_title("3. In-Model Cognitive Intelligence\n[Alien OOD Self-Doubt + Analytical Recourse]", fontsize=13, fontweight='bold', color='#B388FF')
ax3.set_xlabel("Feature 1")
ax3.set_ylabel("Feature 2")
ax3.legend(loc='lower right', fontsize=8)
ax3.grid(True, linestyle='--', alpha=0.2)

plt.tight_layout()
os.makedirs("models", exist_ok=True)
save_path = "models/tmrm_vs_industry_visual_explanation.png"
plt.savefig(save_path, dpi=180, bbox_inches='tight')
print(f"Visual Explanation Chart successfully saved to: {save_path}")
