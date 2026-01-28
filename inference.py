"""
Simple inference script for testing tree segmentation model
"""

import torch
import numpy as np
import os
import sys

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(__file__))

from pointnet2_model import PointNet2Segmentation

# ============================================================
# CONFIGURATION - Update these paths
# ============================================================
CHECKPOINT = "output/checkpoints/best.pth"
INPUT_FILE = r"C:\Users\Student\Downloads\TLS\SEW03_G_138_TLS.laz"
OUTPUT_FILE = "predictions.npy"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*60)
print("TREE SEGMENTATION INFERENCE")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT}")
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print("="*60)

# ============================================================
# LOAD MODEL
# ============================================================
print("\n[1/3] Loading model...")

if not os.path.exists(CHECKPOINT):
    print(f"❌ ERROR: Checkpoint not found: {CHECKPOINT}")
    exit(1)

model = PointNet2Segmentation(num_classes=2).to(DEVICE)
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded successfully!")
print(f"   Epochs trained: {checkpoint['epoch'] + 1}")
print(f"   Best IoU: {checkpoint.get('best_val_iou', 'N/A'):.4f}")

# ============================================================
# LOAD POINT CLOUD
# ============================================================
print(f"\n[2/3] Loading point cloud...")

if not os.path.exists(INPUT_FILE):
    print(f"❌ ERROR: Input file not found: {INPUT_FILE}")
    exit(1)

# Load LAZ file
try:
    import laspy
    las = laspy.read(INPUT_FILE)
    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)
    points = np.column_stack([x, y, z])
    print(f"✅ Loaded {len(points):,} points")
except ImportError:
    print("❌ ERROR: laspy not installed!")
    print("   Install with: pip install laspy[lazrs]")
    exit(1)
except Exception as e:
    print(f"❌ ERROR loading file: {e}")
    exit(1)

# ============================================================
# NORMALIZE AND SAMPLE
# ============================================================
print(f"\n[3/3] Running inference...")

# Normalize to unit sphere
centroid = np.mean(points, axis=0)
points_normalized = points - centroid
max_dist = np.max(np.sqrt(np.sum(points_normalized**2, axis=1)))
points_normalized = points_normalized / max_dist

# Sample 2048 points
num_points = 2048
if len(points) >= num_points:
    indices = np.random.choice(len(points), num_points, replace=False)
else:
    indices = np.random.choice(len(points), num_points, replace=True)

sampled_points = points_normalized[indices]

# Convert to tensor
points_tensor = torch.from_numpy(sampled_points.astype(np.float32)).unsqueeze(0)
points_tensor = points_tensor.transpose(1, 2).to(DEVICE)

# ============================================================
# PREDICT
# ============================================================
with torch.no_grad():
    outputs = model(points_tensor)
    predictions = outputs.argmax(dim=2).squeeze().cpu().numpy()

print(f"✅ Inference complete!")

# ============================================================
# RESULTS
# ============================================================
print("\n" + "="*60)
print("RESULTS")
print("="*60)

class_names = {0: "Conifer", 1: "Broadleaf"}

unique, counts = np.unique(predictions, return_counts=True)
for label, count in zip(unique, counts):
    percentage = 100 * count / len(predictions)
    class_name = class_names.get(label, f"Class {label}")
    print(f"  {class_name:12} (Class {label}): {count:4} points ({percentage:5.1f}%)")

# ============================================================
# SAVE OUTPUT
# ============================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Save predictions with coordinates
output_data = np.column_stack([
    sampled_points,      # X, Y, Z (normalized)
    predictions          # Predicted class
])

np.save(OUTPUT_FILE, output_data)
print(f"✅ Predictions saved to: {OUTPUT_FILE}")
print(f"   Shape: {output_data.shape}")
print(f"   Format: [X, Y, Z, Label]")

# Also save summary
summary_file = OUTPUT_FILE.replace('.npy', '_summary.txt')
with open(summary_file, 'w') as f:
    f.write(f"Tree Segmentation Results\n")
    f.write(f"="*60 + "\n")
    f.write(f"Input file: {INPUT_FILE}\n")
    f.write(f"Model: {CHECKPOINT}\n")
    f.write(f"Epochs trained: {checkpoint['epoch'] + 1}\n")
    f.write(f"Best IoU: {checkpoint.get('best_val_iou', 'N/A'):.4f}\n")
    f.write(f"Total points: {len(points):,}\n")
    f.write(f"Sampled points: {num_points:,}\n")
    f.write(f"\nPredictions:\n")
    for label, count in zip(unique, counts):
        percentage = 100 * count / len(predictions)
        class_name = class_names.get(label, f"Class {label}")
        f.write(f"  {class_name}: {count} points ({percentage:.1f}%)\n")

print(f"✅ Summary saved to: {summary_file}")

print("\n" + "="*60)
print("✨ INFERENCE COMPLETE!")
print("="*60)

# Load and verify
print("\n💡 To load predictions:")
print(f"   data = np.load('{OUTPUT_FILE}')")
print(f"   xyz = data[:, :3]  # Coordinates")
print(f"   labels = data[:, 3]  # Predicted labels")