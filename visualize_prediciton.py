"""
Visualize tree segmentation predictions
Loads predictions.npy and displays in 3D
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("="*60)
print("VISUALIZING TREE SEGMENTATION")
print("="*60)

# Load predictions
predictions_file = "predictions.npy"
print(f"\nLoading: {predictions_file}")

try:
    data = np.load(predictions_file)
    print(f"✅ Loaded {len(data):,} points")
    print(f"   Data shape: {data.shape}")
except:
    print(f"❌ Could not load {predictions_file}")
    print("   Make sure you ran inference first!")
    exit(1)

# Extract coordinates and labels
xyz = data[:, :3]  # X, Y, Z coordinates
labels = data[:, 3].astype(int)  # Predicted labels

print(f"\nData ranges:")
print(f"   X: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
print(f"   Y: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]")
print(f"   Z: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")

# Count classes
unique, counts = np.unique(labels, return_counts=True)
print(f"\nPredictions:")
class_names = {0: "Conifer", 1: "Broadleaf"}
for label, count in zip(unique, counts):
    percentage = 100 * count / len(labels)
    class_name = class_names.get(label, f"Class {label}")
    print(f"   {class_name}: {count} points ({percentage:.1f}%)")

# ============================================================
# CREATE VISUALIZATION
# ============================================================
print("\n" + "="*60)
print("Creating 3D visualization...")
print("="*60)

# Create figure with subplots
fig = plt.figure(figsize=(16, 8))

# Define colors
colors_conifer = np.array([34, 139, 34]) / 255.0      # Forest Green
colors_broadleaf = np.array([255, 140, 0]) / 255.0    # Dark Orange

# Color map for points
point_colors = np.zeros((len(labels), 3))
point_colors[labels == 0] = colors_conifer     # Conifer = Green
point_colors[labels == 1] = colors_broadleaf   # Broadleaf = Orange

# ============================================================
# Plot 1: 3D View
# ============================================================
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
           c=point_colors, s=10, alpha=0.6, edgecolors='none')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Point Cloud - Colored by Prediction', fontsize=14, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors_conifer, label='Conifer (Class 0)'),
    Patch(facecolor=colors_broadleaf, label='Broadleaf (Class 1)')
]
ax1.legend(handles=legend_elements, loc='upper right')

# ============================================================
# Plot 2: Top-Down View
# ============================================================
ax2 = fig.add_subplot(122)
ax2.scatter(xyz[:, 0], xyz[:, 1], 
           c=point_colors, s=20, alpha=0.6, edgecolors='none')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Top-Down View (X-Y Plane)', fontsize=14, fontweight='bold')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()

# Save figure
output_image = "predictions_visualization.png"
plt.savefig(output_image, dpi=150, bbox_inches='tight')
print(f"\n✅ Visualization saved to: {output_image}")

# Show interactive plot
print("\n💡 Displaying interactive 3D plot...")
print("   - Rotate: Click and drag")
print("   - Zoom: Scroll wheel or right-click drag")
print("   - Close window to exit")
plt.show()

print("\n✨ Visualization complete!")