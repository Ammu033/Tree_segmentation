#!/usr/bin/env python3
"""
Setup verification script for BioDiv-3DTrees (LAZ format)
Run this to check if everything is ready for training
"""

import sys
import os


def check_python_version():
    """Check Python version"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor} (need 3.7+)")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("\n🔍 Checking dependencies...")
    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm'
    }
    
    all_installed = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} (not installed)")
            all_installed = False
    
    # Check CRITICAL dependency for LAZ files
    print("\n   Critical for BioDiv-3DTrees:")
    try:
        import laspy
        print("   ✅ laspy (LAZ file support)")
        try:
            # Check if lazrs backend is available
            import lazrs
            print("   ✅ lazrs (LAZ compression support)")
        except ImportError:
            print("   ⚠️  lazrs (install with: pip install laspy[lazrs])")
    except ImportError:
        print("   ❌ laspy (CRITICAL - install with: pip install laspy[lazrs])")
        all_installed = False
    
    # Check optional dependencies
    print("\n   Optional packages:")
    try:
        import open3d
        print("   ✅ Open3D (for visualization)")
    except ImportError:
        print("   ⚠️  Open3D (visualization will not work)")
    
    return all_installed


def check_cuda():
    """Check CUDA availability"""
    print("\n🔍 Checking CUDA/GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available")
            print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   ✅ CUDA version: {torch.version.cuda}")
            
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   ✅ GPU Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory >= 6:
                print(f"   ✅ Sufficient GPU memory (recommend batch_size=8)")
            elif gpu_memory >= 4:
                print(f"   ⚠️  Limited GPU memory (recommend batch_size=4)")
            else:
                print(f"   ⚠️  Low GPU memory (recommend batch_size=2)")
            
            return True
        else:
            print("   ⚠️  CUDA not available (will use CPU)")
            print("   ⚠️  Training will be MUCH slower on CPU (~10-15x)")
            print("   💡 Tip: Consider using Google Colab or cloud GPU")
            return False
    except Exception as e:
        print(f"   ❌ Could not check CUDA: {e}")
        return False


def check_biodiv_dataset():
    """Check if BioDiv-3DTrees dataset exists"""
    print("\n🔍 Checking BioDiv-3DTrees dataset...")
    
    # Expected paths
    laz_dir = r"C:\Users\Student\Downloads\TLS"
    labels_csv = r"C:\Users\Student\Downloads\labels.csv"
    
    dataset_ok = True
    
    # Check LAZ directory
    if os.path.exists(laz_dir):
        print(f"   ✅ LAZ directory found: {laz_dir}")
        
        # Count LAZ files
        laz_files = []
        for root, _, files in os.walk(laz_dir):
            for f in files:
                if f.endswith('.laz') or f.endswith('.LAZ'):
                    laz_files.append(f)
        
        print(f"   ✅ Found {len(laz_files)} LAZ files")
        
        if len(laz_files) == 0:
            print("   ❌ No LAZ files found in directory!")
            dataset_ok = False
        elif len(laz_files) < 100:
            print(f"   ⚠️  Only {len(laz_files)} files - expected ~4900+")
    else:
        print(f"   ❌ LAZ directory not found: {laz_dir}")
        print("   💡 Update path in train_biodiv.py config")
        dataset_ok = False
    
    # Check labels CSV
    if os.path.exists(labels_csv):
        print(f"   ✅ Labels CSV found: {labels_csv}")
        
        # Try to read CSV
        try:
            import pandas as pd
            df = pd.read_csv(labels_csv)
            print(f"   ✅ CSV loaded: {len(df)} entries")
            
            # Check for required columns
            if 'treeID' in df.columns:
                print(f"   ✅ 'treeID' column found")
            else:
                print(f"   ⚠️  'treeID' column not found")
            
            if 'type' in df.columns:
                types = df['type'].unique()
                print(f"   ✅ 'type' column found: {types}")
            elif 'species' in df.columns:
                n_species = df['species'].nunique()
                print(f"   ✅ 'species' column found: {n_species} species")
            else:
                print(f"   ⚠️  No 'type' or 'species' column found")
                
        except Exception as e:
            print(f"   ⚠️  Could not read CSV: {e}")
    else:
        print(f"   ❌ Labels CSV not found: {labels_csv}")
        print("   💡 Update path in train_biodiv.py config")
        dataset_ok = False
    
    return dataset_ok


def test_laz_reading():
    """Test if LAZ files can be read"""
    print("\n🔍 Testing LAZ file reading...")
    
    laz_dir = r"C:\Users\Student\Downloads\TLS"
    
    if not os.path.exists(laz_dir):
        print("   ⚠️  LAZ directory not found, skipping test")
        return False
    
    # Find first LAZ file
    test_file = None
    for root, _, files in os.walk(laz_dir):
        for f in files:
            if f.endswith('.laz') or f.endswith('.LAZ'):
                test_file = os.path.join(root, f)
                break
        if test_file:
            break
    
    if not test_file:
        print("   ⚠️  No LAZ files found to test")
        return False
    
    try:
        import laspy
        las = laspy.read(test_file)
        
        print(f"   ✅ Successfully read LAZ file")
        print(f"   ✅ Sample file: {os.path.basename(test_file)}")
        print(f"   ✅ Points: {len(las.points):,}")
        print(f"   ✅ Coordinates: X, Y, Z")
        
        # Check point cloud size
        if len(las.points) > 100000:
            print(f"   ✅ Good point cloud density ({len(las.points):,} points)")
        else:
            print(f"   ⚠️  Low point density ({len(las.points):,} points)")
        
        return True
        
    except ImportError:
        print("   ❌ laspy not installed!")
        print("   💡 Install with: pip install laspy[lazrs]")
        return False
    except Exception as e:
        print(f"   ❌ Error reading LAZ file: {e}")
        print("   💡 Make sure laspy[lazrs] is properly installed")
        return False


def check_disk_space():
    """Check available disk space"""
    print("\n🔍 Checking disk space...")
    try:
        import shutil
        total, used, free = shutil.disk_usage("./")
        free_gb = free // (2**30)
        
        if free_gb > 10:
            print(f"   ✅ {free_gb} GB free space")
            return True
        elif free_gb > 5:
            print(f"   ⚠️  {free_gb} GB free (recommend 10+ GB for large dataset)")
            return True
        else:
            print(f"   ⚠️  Only {free_gb} GB free (may need more space)")
            return False
    except:
        print("   ⚠️  Could not check disk space")
        return True


def test_model_creation():
    """Test if model can be created"""
    print("\n🔍 Testing PointNet++ model creation...")
    try:
        from pointnet2_model import PointNet2Segmentation
        import torch
        
        model = PointNet2Segmentation(num_classes=2)
        print(f"   ✅ Model created successfully")
        print(f"   ✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        test_input = torch.randn(1, 3, 2048).to(device)
        
        import time
        start = time.time()
        output = model(test_input)
        elapsed = time.time() - start
        
        print(f"   ✅ Forward pass successful")
        print(f"   ✅ Output shape: {output.shape}")
        print(f"   ✅ Inference time: {elapsed*1000:.1f}ms")
        
        return True
    except Exception as e:
        print(f"   ❌ Model test failed: {str(e)}")
        print("   💡 Make sure pointnet2_model.py is in the same directory")
        return False


def test_dataset_loading():
    """Test if dataset can be loaded"""
    print("\n🔍 Testing dataset loading...")
    
    try:
        from dataset_biodiv import BioDivTreeDataset
        
        laz_dir = r"C:\Users\Student\Downloads\TLS"
        labels_csv = r"C:\Users\Student\Downloads\labels.csv"
        
        if not os.path.exists(laz_dir) or not os.path.exists(labels_csv):
            print("   ⚠️  Dataset paths not found, skipping test")
            return False
        
        print("   Loading dataset (this may take a moment)...")
        dataset = BioDivTreeDataset(laz_dir, labels_csv, num_points=1024, split='train')
        
        print(f"   ✅ Dataset loaded: {len(dataset)} samples")
        
        if len(dataset) > 0:
            print("   Testing first sample...")
            points, labels = dataset[0]
            print(f"   ✅ Points shape: {points.shape}")
            print(f"   ✅ Labels shape: {labels.shape}")
            print(f"   ✅ Ready for training!")
            return True
        else:
            print("   ❌ No samples in dataset")
            return False
            
    except Exception as e:
        print(f"   ❌ Dataset loading failed: {e}")
        print("   💡 Check that dataset_biodiv.py is present")
        return False


def estimate_training_time():
    """Estimate training time for BioDiv-3DTrees"""
    print("\n⏱️  Estimated training time (with 3900+ trees):")
    try:
        import torch
        if torch.cuda.is_available():
            print("   With GPU:")
            print("     ~2-3 minutes per epoch")
            print("     ~40-50 epochs in 2 hours")
            print("     Expected IoU: 0.65-0.80")
            print("   ✅ Good for 2-hour training!")
        else:
            print("   With CPU:")
            print("     ~20-30 minutes per epoch")
            print("     ~4-6 epochs in 2 hours")
            print("     Expected IoU: 0.40-0.55")
            print("   ⚠️  STRONGLY recommend using GPU!")
            print("   💡 Consider Google Colab or cloud GPU service")
    except:
        print("   Could not estimate")


def print_next_steps():
    """Print what to do next"""
    print("\n" + "="*60)
    print("📋 NEXT STEPS FOR BIODIV-3DTREES")
    print("="*60)
    
    print("\n1. Update paths in train_biodiv.py (if needed):")
    print("   config = {")
    print("       'laz_dir': r'C:\\Users\\Student\\Downloads\\TLS\\TLS',")
    print("       'labels_csv': r'C:\\Users\\Student\\Downloads\\labels.csv',")
    print("       'batch_size': 4,  # Adjust based on GPU memory")
    print("       'num_workers': 0,  # Keep 0 on Windows")
    print("   }")
    
    print("\n2. Start training:")
    print("   python train_biodiv.py")
    
    print("\n3. Monitor training (in another terminal):")
    print("   tensorboard --logdir ./output/runs")
    print("   Then open: http://localhost:6006")
    
    print("\n4. After training, use your model:")
    print("   python inference.py \\")
    print("       --checkpoint output/checkpoints/best.pth \\")
    print("       --input C:\\Users\\Student\\Downloads\\TLS\\TLS\\AEW03_GD_104_TLS.laz")
    
    print("\n" + "="*60)


def main():
    print("="*70)
    print("🌲 BioDiv-3DTrees PointNet++ Setup Checker")
    print("="*70)
    
    checks = {
        'Python version': check_python_version(),
        'Dependencies': check_dependencies(),
        'CUDA/GPU': check_cuda(),
        'BioDiv dataset': check_biodiv_dataset(),
        'LAZ reading': test_laz_reading(),
        'Disk space': check_disk_space(),
        'Model creation': test_model_creation(),
        'Dataset loading': test_dataset_loading()
    }
    
    estimate_training_time()
    
    # Summary
    print("\n" + "="*70)
    print("📊 SETUP SUMMARY")
    print("="*70)
    
    passed = sum(checks.values())
    total = len(checks)
    
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
    
    print(f"\n{passed}/{total} checks passed")
    
    # Determine readiness
    critical_checks = [
        checks['Dependencies'],
        checks['BioDiv dataset'],
        checks['LAZ reading'],
        checks['Model creation']
    ]
    
    if all(critical_checks):
        print("\n🎉 All critical checks passed! You're ready to train!")
        if not checks['CUDA/GPU']:
            print("⚠️  However, training will be SLOW without GPU")
    elif sum(critical_checks) >= len(critical_checks) - 1:
        print("\n⚠️  Almost ready! Check warnings above.")
    else:
        print("\n❌ Please fix critical issues before training.")
        print("\nCritical requirements:")
        print("  - laspy[lazrs] installed")
        print("  - LAZ files and labels.csv accessible")
        print("  - Model files (pointnet2_model.py, dataset_biodiv.py) present")
    
    print_next_steps()


if __name__ == "__main__":
    main()