# 🚀 QUICK START GUIDE - PointNet++ Tree Segmentation

## Complete 2-Hour Training Pipeline for BioDiv-3DTrees

---

## ⏱️ Timeline Overview

| Time | Task | Duration |
|------|------|----------|
| 0:00-0:05 | Setup environment | 5 min |
| 0:05-0:20 | Prepare data | 15 min |
| 0:20-2:00 | Train model | 100 min |
| 2:00-2:10 | Test inference | 10 min |

---

## 📦 What's Included

```
tree_segmentation/
├── pointnet2_model.py          # PointNet++ architecture
├── dataset.py                  # Data loader for BioDiv-3DTrees
├── train.py                    # Main training script ⭐
├── inference.py                # Prediction script
├── check_setup.py              # Verify setup
├── generate_sample_data.py     # Generate test data
├── requirements.txt            # Dependencies
└── README.md                   # Full documentation
```

---

## 🎯 Step-by-Step Instructions

### STEP 1: Install Dependencies (5 minutes)

```bash
# Navigate to project directory
cd tree_segmentation

# Install required packages
pip install -r requirements.txt

# For GPU support (HIGHLY RECOMMENDED)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Optional: for visualization
pip install open3d
```

**Verify installation:**
```bash
python check_setup.py
```

---

### STEP 2: Prepare Your Data (15 minutes)

#### Option A: Use Your BioDiv-3DTrees Dataset

1. Create data directory:
```bash
mkdir -p data/BioDiv-3DTrees
```

2. Place your point cloud files there:
```
data/BioDiv-3DTrees/
├── tree_001.npy
├── tree_002.npy
└── ...
```

**Supported formats:**
- `.npy` - NumPy array (shape: N×4 with X,Y,Z,Label)
- `.txt` - Text file (format: X Y Z Label per line)
- `.ply` - PLY format with optional label property
- `.xyz` - XYZ format

**Label requirements:**
- 0 = Background
- 1 = Tree

#### Option B: Generate Synthetic Test Data

```bash
python generate_sample_data.py
```

This creates 20 synthetic point clouds for testing the pipeline.

---

### STEP 3: Configure Training (2 minutes)

Open `train.py` and update the configuration:

```python
config = {
    'data_path': './data/BioDiv-3DTrees',  # ← Update this!
    'num_classes': 2,
    'num_points': 2048,      # Points per sample
    'batch_size': 8,         # Reduce if GPU memory issues
    'num_epochs': 50,
    'lr': 0.001,
    'time_limit': 7200       # 2 hours
}
```

**GPU Memory Guide:**
- 12GB GPU: batch_size = 16
- 8GB GPU: batch_size = 8
- 6GB GPU: batch_size = 4
- CPU: batch_size = 2

---

### STEP 4: Start Training (100 minutes)

```bash
python train.py
```

**What you'll see:**

```
Using device: cuda
Model parameters: 1,480,450
Training samples: 80
Validation samples: 20

Epoch 1/50 Summary:
  Train Loss: 0.6543
  Val Loss: 0.5234
  Val IoU: 0.4567
  Val Acc: 78.34%
  Time: 2.3min / 120.0min
✓ New best model saved! IoU: 0.4567
```

**The training will:**
- ✅ Automatically split data (80% train, 20% val)
- ✅ Apply data augmentation
- ✅ Save checkpoints every 10 epochs
- ✅ Save best model (highest IoU)
- ✅ Stop at 2-hour time limit
- ✅ Track metrics with TensorBoard

**Monitor training in real-time:**
```bash
# In another terminal
tensorboard --logdir ./output/runs
# Open: http://localhost:6006
```

---

### STEP 5: Test Your Model (10 minutes)

```bash
# Run inference on a test file
python inference.py \
    --checkpoint ./output/checkpoints/best.pth \
    --input data/BioDiv-3DTrees/test_tree.npy \
    --output predictions.npy \
    --visualize
```

**Output:**
```
Model loaded from ./output/checkpoints/best.pth
Trained for 45 epochs
Best validation IoU: 0.6234

Prediction statistics:
  Class 0: 1024 points (50.0%)
  Class 1: 1024 points (50.0%)
```

---

## 📊 Expected Results

After 2 hours of training:

| Metric | Expected Value |
|--------|----------------|
| Validation IoU | 0.50 - 0.70 |
| Validation Accuracy | 75% - 85% |
| Training Loss | 0.30 - 0.50 |
| Epochs Completed | 30-50 (GPU) / 8-12 (CPU) |

---

## 🔧 Common Issues & Solutions

### Issue 1: "CUDA out of memory"
**Solution:**
```python
# In train.py, reduce batch size
config['batch_size'] = 4  # or 2
```

### Issue 2: "No data found"
**Solution:**
```bash
# Check your data path
ls data/BioDiv-3DTrees/

# Generate sample data to test
python generate_sample_data.py
```

### Issue 3: Training too slow
**Solution:**
- Use GPU (10-20x faster)
- Reduce num_points to 1024
- Reduce num_workers to 2

### Issue 4: Low accuracy
**Solutions:**
- Train longer (increase time_limit)
- Check label quality
- Add more training data
- Adjust learning rate

---

## 📁 Output Files

After training, check these locations:

```
output/
├── checkpoints/
│   ├── best.pth          ← Use this for inference
│   ├── latest.pth
│   ├── epoch_10.pth
│   └── epoch_20.pth
└── runs/                 ← TensorBoard logs
```

---

## 🎓 Tips for Better Results

### Immediate improvements:
1. **Use GPU** - 10-20x faster training
2. **More data** - Add more labeled samples
3. **Better labels** - Ensure accurate labeling

### After first 2 hours:
1. **Train longer** - 4-8 hours for best results
2. **Tune hyperparameters** - Adjust learning rate, batch size
3. **Add augmentation** - More rotation, scaling variations

---

## 🔍 Verify Everything Works

Before starting your 2-hour training:

```bash
python check_setup.py
```

This checks:
- ✅ Python version
- ✅ Dependencies installed
- ✅ CUDA/GPU available
- ✅ Data directory setup
- ✅ Model creation works
- ✅ Disk space sufficient

---

## 📞 Quick Command Reference

```bash
# Generate test data
python generate_sample_data.py

# Verify setup
python check_setup.py

# Train model
python train.py

# Monitor training
tensorboard --logdir ./output/runs

# Run inference
python inference.py --checkpoint output/checkpoints/best.pth --input test.npy

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🎯 What to Expect

### During Training:
- Progress bar with loss and accuracy
- Regular validation checks
- Automatic checkpoint saving
- Time tracking

### After Training:
- Best model saved based on IoU
- Training curves in TensorBoard
- Ready-to-use model for inference

### Inference:
- Per-point predictions (tree vs background)
- Statistics on segmentation
- Optional 3D visualization

---

## ✨ Ready to Start!

```bash
# 1. Verify setup
python check_setup.py

# 2. Prepare data (if needed)
python generate_sample_data.py

# 3. Start training
python train.py

# That's it! The training will run for 2 hours automatically.
```

---

## 📚 Need More Help?

- **Full documentation**: See `README.md`
- **Model architecture**: See `pointnet2_model.py`
- **Data format**: See `dataset.py`
- **Setup issues**: Run `python check_setup.py`

---

**Good luck with your training! 🌲🚀**

The model will automatically stop after 2 hours and save the best checkpoint.
You can resume training later by loading the checkpoint if needed.
