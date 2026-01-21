#!/usr/bin/env python3
"""
Generate synthetic tree point clouds for testing the pipeline
Use this if you don't have the BioDiv-3DTrees dataset yet
"""

import numpy as np
import os


def generate_tree_point_cloud(n_points=5000, noise=0.05):
    """
    Generate a synthetic tree point cloud with labels
    
    Returns:
        points: (n_points, 4) array with [X, Y, Z, Label]
    """
    points = []
    labels = []
    
    # Generate trunk (cylinder)
    trunk_height = 3.0
    trunk_radius = 0.3
    n_trunk = int(n_points * 0.2)
    
    for _ in range(n_trunk):
        height = np.random.uniform(0, trunk_height)
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(0, trunk_radius)
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        
        points.append([x, y, z])
        labels.append(1)  # Tree
    
    # Generate crown (ellipsoid)
    crown_center = trunk_height
    crown_height = 2.0
    crown_radius = 1.5
    n_crown = int(n_points * 0.5)
    
    for _ in range(n_crown):
        # Sample from unit sphere
        u = np.random.uniform(0, 1)
        v = np.random.uniform(0, 1)
        theta = 2 * np.pi * u
        phi = np.arccos(2*v - 1)
        
        x = crown_radius * np.sin(phi) * np.cos(theta)
        y = crown_radius * np.sin(phi) * np.sin(theta)
        z = crown_center + crown_height * np.cos(phi)
        
        points.append([x, y, z])
        labels.append(1)  # Tree
    
    # Generate ground points (background)
    n_ground = int(n_points * 0.3)
    ground_range = 3.0
    
    for _ in range(n_ground):
        x = np.random.uniform(-ground_range, ground_range)
        y = np.random.uniform(-ground_range, ground_range)
        z = np.random.uniform(-0.2, 0.2)  # Ground level
        
        points.append([x, y, z])
        labels.append(0)  # Background
    
    # Convert to numpy arrays
    points = np.array(points)
    labels = np.array(labels).reshape(-1, 1)
    
    # Add noise
    points += np.random.normal(0, noise, points.shape)
    
    # Combine points and labels
    point_cloud = np.hstack([points, labels])
    
    return point_cloud


def generate_forest_scene(n_points=10000, n_trees=3):
    """
    Generate a forest scene with multiple trees
    """
    all_points = []
    all_labels = []
    
    # Generate multiple trees at different positions
    for i in range(n_trees):
        tree = generate_tree_point_cloud(n_points=n_points//n_trees)
        
        # Random position
        offset_x = np.random.uniform(-5, 5)
        offset_y = np.random.uniform(-5, 5)
        tree[:, 0] += offset_x
        tree[:, 1] += offset_y
        
        all_points.append(tree[:, :3])
        all_labels.append(tree[:, 3])
    
    # Add background points
    n_background = n_points // 4
    background = np.random.uniform(-10, 10, (n_background, 3))
    background[:, 2] = np.random.uniform(-0.3, 0.3, n_background)  # Ground level
    
    all_points.append(background)
    all_labels.append(np.zeros(n_background))
    
    # Combine all
    points = np.vstack(all_points)
    labels = np.hstack(all_labels).reshape(-1, 1)
    
    point_cloud = np.hstack([points, labels])
    
    return point_cloud


def main():
    """Generate sample dataset"""
    print("🌲 Generating synthetic tree dataset...")
    
    # Create output directory
    output_dir = './data/BioDiv-3DTrees'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate different types of samples
    n_samples = 20
    
    print(f"\nGenerating {n_samples} synthetic point clouds...")
    
    for i in range(n_samples):
        if i % 2 == 0:
            # Single tree
            point_cloud = generate_tree_point_cloud(n_points=5000)
            scene_type = "single_tree"
        else:
            # Forest scene
            n_trees = np.random.randint(2, 5)
            point_cloud = generate_forest_scene(n_points=8000, n_trees=n_trees)
            scene_type = f"forest_{n_trees}trees"
        
        # Save as .npy file
        filename = f'synthetic_{scene_type}_{i:03d}.npy'
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, point_cloud)
        
        if (i + 1) % 5 == 0:
            print(f"   Generated {i+1}/{n_samples} files...")
    
    print(f"\n✅ Generated {n_samples} synthetic point clouds")
    print(f"📁 Saved to: {output_dir}")
    
    # Print statistics
    sample = np.load(os.path.join(output_dir, 'synthetic_single_tree_000.npy'))
    print(f"\n📊 Sample statistics:")
    print(f"   Points per cloud: {sample.shape[0]}")
    print(f"   Dimensions: {sample.shape[1]} (X, Y, Z, Label)")
    print(f"   Label distribution:")
    unique, counts = np.unique(sample[:, 3], return_counts=True)
    for label, count in zip(unique, counts):
        label_name = "Tree" if label == 1 else "Background"
        percentage = 100 * count / len(sample)
        print(f"      {label_name} (class {int(label)}): {count} points ({percentage:.1f}%)")
    
    print(f"\n✨ You can now train the model with:")
    print(f"   python train.py")
    
    # Optional: Visualize one sample
    print(f"\n💡 To visualize a sample (requires open3d):")
    print(f"   python visualize_sample.py")


def visualize_sample():
    """Visualize a sample point cloud (requires open3d)"""
    try:
        import open3d as o3d
        
        # Load sample
        sample_path = './data/BioDiv-3DTrees/synthetic_single_tree_000.npy'
        if not os.path.exists(sample_path):
            print("No sample found. Generate samples first!")
            return
        
        data = np.load(sample_path)
        points = data[:, :3]
        labels = data[:, 3]
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Color by labels
        colors = np.zeros((len(labels), 3))
        colors[labels == 0] = [0.7, 0.7, 0.7]  # Gray for background
        colors[labels == 1] = [0.0, 0.8, 0.0]  # Green for tree
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Visualize
        print("Visualizing sample... (close window to continue)")
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="Synthetic Tree Sample",
            width=800,
            height=600
        )
    
    except ImportError:
        print("Open3D not installed. Install with: pip install open3d")
    except Exception as e:
        print(f"Error visualizing: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--visualize':
        visualize_sample()
    else:
        main()
        
        # Ask if user wants to visualize
        try:
            response = input("\nWould you like to visualize a sample? (y/n): ")
            if response.lower() == 'y':
                visualize_sample()
        except:
            pass
