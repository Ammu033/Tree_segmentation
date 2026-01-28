"""
Visualize first 10 trees from BioDiv-3DTrees dataset
Shows ORIGINAL point clouds (no predictions)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ============================================================
# CONFIGURATION
# ============================================================
LAZ_DIR = r"C:\Users\Student\Downloads\TLS"
NUM_TREES = 10
NUM_POINTS_PER_TREE = 5000  # Show more points for better visualization

print("="*70)
print("VISUALIZING FIRST 10 ORIGINAL TREES (NO PREDICTIONS)")
print("="*70)

# ============================================================
# LOAD TREES
# ============================================================
print(f"\nLoading first {NUM_TREES} trees from {LAZ_DIR}...")

# Get LAZ files
if not os.path.exists(LAZ_DIR):
    print(f"❌ Directory not found: {LAZ_DIR}")
    exit(1)

all_files = sorted([f for f in os.listdir(LAZ_DIR) if f.endswith('.laz')])
tree_files = all_files[:NUM_TREES]

if len(tree_files) == 0:
    print("❌ No LAZ files found!")
    exit(1)

print(f"✅ Found {len(tree_files)} trees to visualize\n")

# ============================================================
# LOAD LAZ FILES
# ============================================================
try:
    import laspy
except ImportError:
    print("❌ laspy not installed!")
    print("   Install: pip install laspy[lazrs]")
    exit(1)

trees_data = []

for i, filename in enumerate(tree_files):
    filepath = os.path.join(LAZ_DIR, filename)
    print(f"   [{i+1}/{len(tree_files)}] Loading {filename}...", end=" ")
    
    try:
        # Load LAZ
        las = laspy.read(filepath)
        x = np.array(las.x)
        y = np.array(las.y)
        z = np.array(las.z)
        points = np.column_stack([x, y, z])
        
        total_points = len(points)
        
        # Sample points for visualization
        if len(points) > NUM_POINTS_PER_TREE:
            indices = np.random.choice(len(points), NUM_POINTS_PER_TREE, replace=False)
            sampled_points = points[indices]
        else:
            sampled_points = points
        
        # Center the point cloud
        centroid = np.mean(sampled_points, axis=0)
        centered_points = sampled_points - centroid
        
        trees_data.append({
            'filename': filename,
            'points': centered_points,
            'total_points': total_points,
            'z_range': points[:, 2].max() - points[:, 2].min()
        })
        
        print(f"✅ {total_points:,} points (height: {trees_data[-1]['z_range']:.1f}m)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# ============================================================
# CREATE VISUALIZATION
# ============================================================
print("\n" + "="*70)
print("Creating 3D visualization...")
print("="*70)

# Calculate grid size
n_trees = len(trees_data)
n_cols = 5
n_rows = (n_trees + n_cols - 1) // n_cols

# Create large figure
fig = plt.figure(figsize=(20, 4 * n_rows))
fig.suptitle('First 10 Trees - Original Point Clouds (Colored by Height)', 
             fontsize=16, fontweight='bold', y=0.995)

# Plot each tree
for i, tree in enumerate(trees_data):
    ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
    
    points = tree['points']
    
    # Color by height (Z coordinate)
    z_values = points[:, 2]
    colors = plt.cm.viridis((z_values - z_values.min()) / (z_values.max() - z_values.min()))
    
    # Plot
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c=colors, s=2, alpha=0.7)
    
    # Title with filename and stats
    title = f"{tree['filename'][:25]}\n"
    title += f"{tree['total_points']:,} points | H:{tree['z_range']:.1f}m"
    ax.set_title(title, fontsize=9)
    
    # Clean axes
    ax.set_xlabel('X', fontsize=7)
    ax.set_ylabel('Y', fontsize=7)
    ax.set_zlabel('Z (Height)', fontsize=7)
    ax.tick_params(labelsize=6)
    
    # Set equal aspect ratio
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()

# Save figure
output_file = "original_10_trees_visualization.png"
plt.savefig(output_file, dpi=200, bbox_inches='tight')
print(f"\n✅ Visualization saved to: {output_file}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

avg_points = np.mean([t['total_points'] for t in trees_data])
avg_height = np.mean([t['z_range'] for t in trees_data])
total_points = sum([t['total_points'] for t in trees_data])

print(f"\nDataset statistics:")
print(f"   Total trees:      {len(trees_data)}")
print(f"   Total points:     {total_points:,}")
print(f"   Avg points/tree:  {avg_points:,.0f}")
print(f"   Avg tree height:  {avg_height:.1f}m")

print(f"\nIndividual trees:")
print(f"   {'#':<3} {'Filename':<30} {'Points':>10} {'Height':>8}")
print(f"   {'-'*3} {'-'*30} {'-'*10} {'-'*8}")
for i, tree in enumerate(trees_data):
    print(f"   {i+1:<3} {tree['filename']:<30} {tree['total_points']:>10,} {tree['z_range']:>7.1f}m")

# Save summary
summary_file = "original_10_trees_summary.txt"
with open(summary_file, 'w') as f:
    f.write("First 10 Trees - Original Point Clouds\n")
    f.write("="*70 + "\n\n")
    f.write(f"Dataset statistics:\n")
    f.write(f"   Total trees:      {len(trees_data)}\n")
    f.write(f"   Total points:     {total_points:,}\n")
    f.write(f"   Avg points/tree:  {avg_points:,.0f}\n")
    f.write(f"   Avg tree height:  {avg_height:.1f}m\n\n")
    f.write(f"Individual trees:\n")
    f.write(f"   {'#':<3} {'Filename':<30} {'Points':>10} {'Height':>8}\n")
    f.write(f"   {'-'*3} {'-'*30} {'-'*10} {'-'*8}\n")
    for i, tree in enumerate(trees_data):
        f.write(f"   {i+1:<3} {tree['filename']:<30} {tree['total_points']:>10,} {tree['z_range']:>7.1f}m\n")

print(f"\n✅ Summary saved to: {summary_file}")

print("\n" + "="*70)
print("✨ Complete! Showing visualization...")
print("="*70)
print("\n💡 Color coding:")
print("   🟣 Purple/Blue  = Lower points (ground level)")
print("   🟢 Green/Yellow = Middle points")
print("   🟡 Yellow/Red   = Higher points (tree top)")
print("\n   Colors show HEIGHT gradient (Z-axis)")
print("\n   You can rotate the 3D plots with your mouse!")

# Show plot
plt.show()

print("\n✨ Done!")