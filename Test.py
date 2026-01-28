"""
Quick test script for latest.pth model
Tests on BioDiv-3DTrees LAZ files
"""

import torch
import os
from inference import load_model, predict_file

# Configuration
CHECKPOINT = "output/checkpoints/latest.pth"  # Using latest.pth
LAZ_DIR = r"C:\Users\Student\Downloads\TLS\SEW03_G_127_TLS.laz"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*60)
print("TESTING LATEST MODEL")
print("="*60)

# Check if checkpoint exists
if not os.path.exists(CHECKPOINT):
    print(f"❌ Checkpoint not found: {CHECKPOINT}")
    print("\nAvailable checkpoints:")
    checkpoint_dir = "output/checkpoints"
    if os.path.exists(checkpoint_dir):
        files = os.listdir(checkpoint_dir)
        for f in files:
            print(f"  - {f}")
    else:
        print("  No checkpoints directory found!")
    exit(1)

# Load model
print(f"\n📦 Loading model from: {CHECKPOINT}")
print(f"🖥️  Device: {DEVICE}")
model = load_model(CHECKPOINT, device=DEVICE)

# Get test files
test_files = []
if os.path.exists(LAZ_DIR):
    all_files = os.listdir(LAZ_DIR)
    laz_files = [f for f in all_files if f.endswith('.laz')]
    # Take first 3 files for testing
    test_files = [os.path.join(LAZ_DIR, f) for f in laz_files[:3]]
else:
    print(f"❌ LAZ directory not found: {LAZ_DIR}")
    exit(1)

if len(test_files) == 0:
    print("❌ No LAZ files found!")
    exit(1)

print(f"\n🌲 Testing on {len(test_files)} trees...")
print("="*60)

# Test each file
results = []
for i, filepath in enumerate(test_files):
    filename = os.path.basename(filepath)
    print(f"\n[{i+1}/{len(test_files)}] Testing: {filename}")
    print("-"*60)
    
    try:
        predictions, indices, points = predict_file(model, filepath, device=DEVICE)
        
        # Store results
        unique, counts = torch.unique(torch.from_numpy(predictions), return_counts=True)
        results.append({
            'file': filename,
            'predictions': predictions,
            'class_0_pct': (predictions == 0).sum() / len(predictions) * 100,
            'class_1_pct': (predictions == 1).sum() / len(predictions) * 100,
        })
        
        print("✅ Success!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append({
            'file': filename,
            'error': str(e)
        })

# Summary
print("\n" + "="*60)
print("TESTING SUMMARY")
print("="*60)

successful = sum(1 for r in results if 'error' not in r)
print(f"✅ Successful: {successful}/{len(results)}")

if successful > 0:
    print("\nPrediction Distribution:")
    for r in results:
        if 'error' not in r:
            print(f"  {r['file'][:30]:30} | Conifer: {r['class_0_pct']:.1f}% | Broadleaf: {r['class_1_pct']:.1f}%")

print("\n" + "="*60)
print("✨ Testing Complete!")
print("="*60)

# Instructions for full inference
print("\n💡 To test on a specific file:")
print(f'python inference.py --checkpoint {CHECKPOINT} --input "path/to/your/tree.laz"')
print("\n💡 To test with visualization:")
print(f'python inference.py --checkpoint {CHECKPOINT} --input "path/to/your/tree.laz" --visualize')
print("\n💡 To save predictions:")
print(f'python inference.py --checkpoint {CHECKPOINT} --input "path/to/your/tree.laz" --output results.npy')