# PointNet++ for Tree Segmentation (BioDiv-3DTrees)

Complete training pipeline for tree segmentation using PointNet++ on the BioDiv-3DTrees dataset.

## 🚀 Quick Start (2-Hour Training)

### 1. Installation (5 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# If you have GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Optional: for visualization
pip install open3d
```

### 2. Prepare Your Data (10-15 minutes)

#### Expected Directory Structure:
```
tree_segmentation/
├── data/
│   └── BioDiv-3DTrees/
│       ├── tree1.npy  (or .txt, .ply, .xyz)
│       ├── tree2.npy
│       └── ...
├── train.py
├── inference.py
└── ...
```

#### Supported Data Formats:

**Option A: NumPy format (.npy) - RECOMMENDED**
```python
# Shape: (N, 4) where columns are [X, Y, Z, Label]
# Label: 0 = background, 1 = tree
import numpy as np
point_cloud = np.random.randn(10000, 4)
np.save('data/BioDiv-3DTrees/tree1.npy', point_cloud)
```

**Option B: Text format (.txt or .xyz)**
```
# Format: X Y Z Label (space or comma separated)
0.5 0.3 1.2 1
0.6 0.4 1.3 1
...
```

**Option C: PLY format (.ply)**
- Standard PLY format with x, y, z coordinates
- Optional 'label' property for ground truth labels

#### If You Don't Have Labels:
The dataset loader will automatically create pseudo-labels based on distance from centroid. For best results, add labels manually or use a labeling tool.

### 3. Train the Model (90-100 minutes)

```bash
# Edit train.py to set your data path
# config['data_path'] = './data/BioDiv-3DTrees'

# Start training
python train.py
```

#### Training Configuration:

```python
config = {
    'data_path': './data/BioDiv-3DTrees',  # UPDATE THIS
    'num_classes': 2,        # Background and Tree
    'num_points': 2048,      # Points per sample
    'batch_size': 8,         # Reduce if GPU memory is low
    'num_epochs': 50,        # Will train as many as time allows
    'lr': 0.001,             # Learning rate
    'time_limit': 7200       # 2 hours in seconds
}
```

#### What Happens During Training:

1. **Automatic train/val split**: 80% training, 20% validation
2. **Data augmentation**: Random rotation, jittering, scaling
3. **Checkpointing**: Saves every 10 epochs + best model
4. **Time management**: Automatically stops at 2-hour limit
5. **Monitoring**: Loss, IoU, and accuracy tracked

#### Training Output:
```
Using device: cuda
Model parameters: 1,480,450
Training samples: 80
Validation samples: 20
Batch size: 8
Number of epochs: 50
...

Epoch 1/50 Summary:
  Train Loss: 0.6543
  Val Loss: 0.5234
  Val IoU: 0.4567
  Val Acc: 78.34%
  Time: 2.3min / 120.0min
```

### 4. Monitor Training

Training logs are saved in `./output/runs/`. View with TensorBoard:

```bash
tensorboard --logdir ./output/runs
```

Open browser to `http://localhost:6006` to see:
- Training/validation loss curves
- IoU progression
- Learning rate schedule

### 5. Use Trained Model for Inference

```bash
# Run inference on a single file
python inference.py \
    --checkpoint ./output/checkpoints/best.pth \
    --input data/test_tree.npy \
    --output predictions.npy \
    --visualize
```

## 📊 Expected Results (2-Hour Training)

| Metric | Expected Range |
|--------|----------------|
| Training Loss | 0.3 - 0.5 |
| Validation IoU | 0.5 - 0.7 |
| Validation Accuracy | 75% - 85% |
| Epochs Completed | 30 - 50 (depends on GPU) |

**Note**: Results improve significantly with:
- More training time (4-8 hours)
- Larger dataset
- Better quality labels
- Hyperparameter tuning

## 🔧 Troubleshooting

### GPU Out of Memory
```python
# Reduce batch size in train.py
config['batch_size'] = 4  # or even 2
```

### Slow Training
```python
# Reduce number of points
config['num_points'] = 1024  # instead of 2048

# Reduce workers if CPU limited
config['num_workers'] = 2
```

### No Data Found
```
ERROR: No point cloud files found in ./data/BioDiv-3DTrees
```
**Solution**: Check that your data path is correct and contains supported file formats (.npy, .txt, .ply, .xyz)

### Low Validation Accuracy
- Check if labels are correct (0 and 1, not other values)
- Ensure point clouds are properly normalized
- Increase training time
- Add more training data

## 📁 Output Files

After training, you'll find:

```
output/
├── checkpoints/
│   ├── best.pth          # Best model (highest IoU)
│   ├── latest.pth        # Most recent epoch
│   ├── epoch_10.pth      # Periodic checkpoints
│   └── epoch_20.pth
└── runs/                 # TensorBoard logs
```

## 🎯 Key Features

✅ **Automatic data handling**: Supports multiple formats
✅ **Time management**: Stops automatically at 2-hour limit
✅ **Data augmentation**: Improves generalization
✅ **Checkpointing**: Never lose your progress
✅ **Monitoring**: Real-time training metrics
✅ **Easy inference**: Simple prediction interface

## 🔬 Model Architecture

PointNet++ with:
- 3 Set Abstraction layers (hierarchical feature learning)
- 3 Feature Propagation layers (upsampling)
- Input: (B, 3, N) - batch of point clouds
- Output: (B, N, num_classes) - per-point predictions

## 📈 Improving Results

**After initial 2-hour training:**

1. **Train longer**: Remove or increase time_limit
```python
config['time_limit'] = 14400  # 4 hours
```

2. **Tune hyperparameters**:
```python
config['lr'] = 0.0005           # Lower learning rate
config['num_points'] = 4096     # More points
config['batch_size'] = 16       # Larger batches
```

3. **Better data**:
- Add more labeled samples
- Improve label quality
- Balance classes (equal tree/background)

4. **Advanced techniques**:
- Class weighting for imbalanced data
- Learning rate warmup
- Stronger data augmentation

## 💡 Tips for BioDiv-3DTrees Dataset

1. **Point cloud density**: Normalize point densities across samples
2. **Tree size variation**: Augment with random scaling
3. **Multiple tree species**: Consider multi-class segmentation
4. **Partial scans**: Handle incomplete point clouds with dropout

## 📚 Citation

If you use this code, please cite:

```bibtex
@article{pointnet++,
  title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
  author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
  journal={Advances in Neural Information Processing Systems},
  year={2017}
}
```

## 🆘 Need Help?

Common issues and solutions:

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size or num_points |
| Slow training | Use GPU, reduce num_workers |
| Poor accuracy | Train longer, check labels, add data |
| No convergence | Check learning rate, inspect data |

## 📞 Quick Reference Commands

```bash
# Train model
python train.py

# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Test dataset loading
python dataset.py

# Run inference
python inference.py --checkpoint output/checkpoints/best.pth --input test.npy

# View training logs
tensorboard --logdir output/runs
```

Good luck with your training! 🌲🚀
