import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

print("Generating 3D Diagram 1: Discrete Ollivier-Ricci Curvature Topologies...")
fig1 = plt.figure(figsize=(15, 5), dpi=300)

# 1. Positive Curvature: Sphere
ax1 = fig1.add_subplot(1, 3, 1, projection='3d')
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 40)
x_sph = np.outer(np.cos(u), np.sin(v))
y_sph = np.outer(np.sin(u), np.sin(v))
z_sph = np.outer(np.ones(np.size(u)), np.cos(v))
surf1 = ax1.plot_surface(x_sph, y_sph, z_sph, cmap='viridis', alpha=0.85, edgecolor='none')
ax1.set_title("Positive Ricci Curvature ($\\kappa > 0$)\nSpherical Geometry\nAuto-Boost: $L_2$ Riemannian Wave (60%)", fontsize=11, fontweight='bold', pad=12)
ax1.set_axis_off()

# 2. Zero Curvature: Flat Plane
ax2 = fig1.add_subplot(1, 3, 2, projection='3d')
x_flat = np.linspace(-1, 1, 30)
y_flat = np.linspace(-1, 1, 30)
X_f, Y_f = np.meshgrid(x_flat, y_flat)
Z_f = 0.15 * X_f - 0.10 * Y_f
surf2 = ax2.plot_surface(X_f, Y_f, Z_f, cmap='plasma', alpha=0.85, edgecolor='gray', linewidth=0.2)
ax2.set_title("Zero Ricci Curvature ($\\kappa \\approx 0$)\nFlat Euclidean Space\nBalanced Mix: (40% $L_2$, 35% $L_\\infty$, 25% $L_1$)", fontsize=11, fontweight='bold', pad=12)
ax2.set_axis_off()

# 3. Negative Curvature: Hyperbolic Saddle
ax3 = fig1.add_subplot(1, 3, 3, projection='3d')
x_sad = np.linspace(-1, 1, 35)
y_sad = np.linspace(-1, 1, 35)
X_s, Y_s = np.meshgrid(x_sad, y_sad)
Z_s = X_s**2 - Y_s**2
surf3 = ax3.plot_surface(X_s, Y_s, Z_s, cmap='coolwarm', alpha=0.85, edgecolor='none')
ax3.set_title("Negative Ricci Curvature ($\\kappa < 0$)\nHyperbolic Saddle / Tree Domain\nAuto-Boost: $L_\\infty$ Chebyshev Hyper-Box (45%)", fontsize=11, fontweight='bold', pad=12)
ax3.set_axis_off()

plt.tight_layout()
fig1.savefig("docs/assets/fig_ollivier_ricci_curvature.png", bbox_inches='tight')
plt.close(fig1)
print("Saved docs/assets/fig_ollivier_ricci_curvature.png")

print("\nGenerating 3D Diagram 2: Continuous Resonant Potential Field vs Tree Staircase...")
fig2 = plt.figure(figsize=(14, 6), dpi=300)

x = np.linspace(-2, 2, 60)
y = np.linspace(-2, 2, 60)
X, Y = np.meshgrid(x, y)

# Smooth Harmonic Field
Z_smooth = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-0.15 * (X**2 + Y**2)) + 0.3 * X

# Jagged Tree Staircase Step Function
Z_step = np.floor(Z_smooth * 4.0) / 4.0

# 1. Tree Staircase Plot
ax_tree = fig2.add_subplot(1, 2, 1, projection='3d')
ax_tree.plot_surface(X, Y, Z_step, cmap='copper', alpha=0.85, edgecolor='black', linewidth=0.3)
ax_tree.set_title("Traditional Tree Regression (RF / XGBoost)\nDiscontinuous Axis-Aligned Staircase Noise\n$(\\nabla f$ undefined at boundaries)", fontsize=11, fontweight='bold', pad=12)
ax_tree.set_xlabel("Feature $x_1$", fontsize=9)
ax_tree.set_ylabel("Feature $x_2$", fontsize=9)
ax_tree.set_zlabel("Predicted $y$", fontsize=9)
ax_tree.view_init(elev=28, azim=-55)

# 2. TMRM Continuous Resonant Field
ax_tmrm = fig2.add_subplot(1, 2, 2, projection='3d')
ax_tmrm.plot_surface(X, Y, Z_smooth, cmap='viridis', alpha=0.85, edgecolor='none')
ax_tmrm.contour(X, Y, Z_smooth, zdir='z', offset=np.min(Z_smooth)-0.2, cmap='viridis', alpha=0.6)
ax_tmrm.set_title("TMRM v4.0 Continuous Resonant Field\nSmooth Riemannian Energy Landscape\n$(\\mathcal{C}^\\infty$ Differentiable Geodesic Flow)", fontsize=11, fontweight='bold', pad=12)
ax_tmrm.set_xlabel("Feature $x_1$", fontsize=9)
ax_tmrm.set_ylabel("Feature $x_2$", fontsize=9)
ax_tmrm.set_zlabel("Predicted $y$", fontsize=9)
ax_tmrm.view_init(elev=28, azim=-55)

plt.tight_layout()
fig2.savefig("docs/assets/fig_continuous_resonant_field_vs_trees.png", bbox_inches='tight')
plt.close(fig2)
print("Saved docs/assets/fig_continuous_resonant_field_vs_trees.png")
