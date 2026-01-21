import torch
import numpy as np
import os
from pointnet2_model import PointNet2Segmentation
import argparse


def load_model(checkpoint_path, num_classes=2, device='cuda'):
    """Load trained model from checkpoint"""
    model = PointNet2Segmentation(num_classes=num_classes).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']+1} epochs")
    print(f"Best validation IoU: {checkpoint.get('best_val_iou', 'N/A')}")
    
    return model


def normalize_point_cloud(points):
    """Normalize point cloud to unit sphere"""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / max_dist
    return points


def predict_tree(model, points, device='cuda', num_points=2048):
    """
    Predict segmentation for a single point cloud
    
    Args:
        model: Trained PointNet++ model
        points: numpy array of shape (N, 3) - point cloud
        device: cuda or cpu
        num_points: number of points to sample
    
    Returns:
        predictions: numpy array of shape (N,) - predicted labels for each point
    """
    original_size = points.shape[0]
    
    # Normalize
    points = normalize_point_cloud(points)
    
    # Sample points
    if points.shape[0] >= num_points:
        indices = np.random.choice(points.shape[0], num_points, replace=False)
    else:
        indices = np.random.choice(points.shape[0], num_points, replace=True)
    
    sampled_points = points[indices]
    
    # Convert to tensor
    points_tensor = torch.from_numpy(sampled_points.astype(np.float32)).unsqueeze(0)
    points_tensor = points_tensor.transpose(1, 2).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(points_tensor)
        predictions = outputs.argmax(dim=2).squeeze().cpu().numpy()
    
    return predictions, indices


def predict_file(model, filepath, device='cuda'):
    """Predict segmentation for a point cloud file"""
    # Load point cloud
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.npy':
        data = np.load(filepath)
    elif ext == '.txt' or ext == '.xyz':
        data = np.loadtxt(filepath)
    elif ext == '.ply':
        from plyfile import PlyData
        plydata = PlyData.read(filepath)
        vertex = plydata['vertex']
        data = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Extract XYZ coordinates
    if data.shape[1] >= 3:
        points = data[:, :3]
    else:
        raise ValueError("Point cloud must have at least 3 dimensions (X, Y, Z)")
    
    # Predict
    predictions, indices = predict_tree(model, points, device)
    
    # Print statistics
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"\nPrediction statistics:")
    for label, count in zip(unique, counts):
        percentage = 100 * count / len(predictions)
        print(f"  Class {label}: {count} points ({percentage:.2f}%)")
    
    return predictions, indices, points


def visualize_predictions(points, predictions, save_path=None):
    """Visualize predicted segmentation (requires open3d)"""
    try:
        import open3d as o3d
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Color based on predictions
        colors = np.zeros((len(predictions), 3))
        colors[predictions == 0] = [0.7, 0.7, 0.7]  # Gray for background
        colors[predictions == 1] = [0.0, 0.8, 0.0]  # Green for tree
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Visualize
        o3d.visualization.draw_geometries([pcd], window_name="Tree Segmentation")
        
        # Save if requested
        if save_path:
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"Saved visualization to {save_path}")
    
    except ImportError:
        print("Open3D not installed. Skipping visualization.")
        print("Install with: pip install open3d")


def main():
    parser = argparse.ArgumentParser(description='Tree Segmentation Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input point cloud file')
    parser.add_argument('--output', type=str, default=None, help='Path to save predictions')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device=args.device)
    
    # Predict
    print(f"Processing {args.input}...")
    predictions, indices, points = predict_file(model, args.input, device=args.device)
    
    # Save predictions
    if args.output:
        output_data = np.column_stack([points[indices], predictions])
        np.save(args.output, output_data)
        print(f"Predictions saved to {args.output}")
    
    # Visualize
    if args.visualize:
        visualize_predictions(points[indices], predictions)


if __name__ == "__main__":
    # Example usage without command line
    checkpoint_path = "./output/checkpoints/best.pth"
    
    if os.path.exists(checkpoint_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(checkpoint_path, device=device)
        print("\nModel loaded successfully!")
        print("\nTo use inference:")
        print("python inference.py --checkpoint ./output/checkpoints/best.pth --input your_tree.npy --visualize")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first using train.py")
