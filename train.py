import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime

from pointnet2_model import PointNet2Segmentation
from dataset_biodiv import create_biodiv_dataloaders


class TreeSegmentationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = PointNet2Segmentation(num_classes=config['num_classes']).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Loss and optimizer
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'], weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.7)
        
        # Metrics tracking
        self.best_val_iou = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        
        # Create output directory
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'runs'))
        
        # Timing
        self.start_time = time.time()
        self.time_limit = config.get('time_limit', 7200)  # 2 hours default
    
    def calculate_iou(self, pred, target, num_classes):
        """Calculate IoU for each class"""
        ious = []
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        for cls in range(num_classes):
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = (pred_inds & target_inds).sum()
            union = (pred_inds | target_inds).sum()
            
            if union == 0:
                ious.append(float('nan'))
            else:
                ious.append(intersection / union)
        
        return np.nanmean(ious)
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
        
        for batch_idx, (points, labels) in enumerate(pbar):
            # Check time limit
            elapsed_time = time.time() - self.start_time
            if elapsed_time > self.time_limit:
                print(f"\nTime limit reached ({self.time_limit/3600:.2f} hours)")
                return total_loss / (batch_idx + 1), True
            
            points, labels = points.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(points)
            
            # Calculate loss
            loss = self.criterion(outputs.reshape(-1, self.config['num_classes']), labels.reshape(-1))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = outputs.argmax(dim=2)
            correct += (pred == labels).sum().item()
            total += labels.numel()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%",
                'time': f"{elapsed_time/60:.1f}min"
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, False
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_iou = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for points, labels in tqdm(val_loader, desc="Validating"):
                points, labels = points.to(self.device), labels.to(self.device)
                
                outputs = self.model(points)
                loss = self.criterion(outputs.reshape(-1, self.config['num_classes']), labels.reshape(-1))
                
                total_loss += loss.item()
                pred = outputs.argmax(dim=2)
                
                # Calculate accuracy
                correct += (pred == labels).sum().item()
                total += labels.numel()
                
                # Calculate IoU
                iou = self.calculate_iou(pred, labels, self.config['num_classes'])
                total_iou += iou
        
        avg_loss = total_loss / len(val_loader)
        avg_iou = total_iou / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, avg_iou, accuracy
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_iou': self.best_val_iou,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'checkpoints', 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoints', 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ New best model saved! IoU: {self.best_val_iou:.4f}")
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            periodic_path = os.path.join(self.output_dir, 'checkpoints', f'epoch_{epoch+1}.pth')
            torch.save(checkpoint, periodic_path)
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training - BioDiv-3DTrees")
        print("="*60)
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Number of epochs: {self.config['num_epochs']}")
        print(f"Learning rate: {self.config['lr']}")
        print(f"Device: {self.device}")
        print(f"Time limit: {self.time_limit/3600:.2f} hours")
        print("="*60 + "\n")
        
        for epoch in range(self.config['num_epochs']):
            # Training
            train_loss, time_exceeded = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            if time_exceeded:
                print("Time limit exceeded, stopping training...")
                break
            
            # Validation
            val_loss, val_iou, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('IoU/val', val_iou, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # Print epoch summary
            elapsed = time.time() - self.start_time
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val IoU: {val_iou:.4f}")
            print(f"  Val Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Time: {elapsed/60:.1f}min / {self.time_limit/60:.1f}min")
            
            # Save checkpoint
            is_best = val_iou > self.best_val_iou
            if is_best:
                self.best_val_iou = val_iou
            
            self.save_checkpoint(epoch, is_best)
        
        # Final summary
        total_time = time.time() - self.start_time
        print("\n" + "="*60)
        print("Training Completed!")
        print("="*60)
        print(f"Total training time: {total_time/60:.2f} minutes")
        print(f"Best validation IoU: {self.best_val_iou:.4f}")
        print(f"Checkpoints saved to: {os.path.join(self.output_dir, 'checkpoints')}")
        print("="*60)
        
        self.writer.close()
        
        return self.best_val_iou


def main():
    # ========================================================================
    # CONFIGURATION - UPDATE THESE PATHS FOR YOUR SYSTEM
    # ========================================================================
    config = {
        # DATA PATHS - UPDATE THESE!
        'laz_dir': r'C:\Users\Student\Downloads\TLS\TLS',  # Path to LAZ files
        'labels_csv': r'C:\Users\Student\Downloads\labels.csv',  # Path to labels CSV
        
        # MODEL CONFIGURATION
        'output_dir': './output',
        'num_classes': 2,        # Background and Tree
        'num_points': 2048,      # Points per sample (reduce if memory issues)
        'batch_size': 10,         # Reduce to 2 if GPU memory issues
        'num_epochs': 10,        # Will train as many as time allows
        'lr': 0.001,             # Learning rate
        'num_workers': 0,        # Set to 0 on Windows to avoid multiprocessing issues
        'time_limit': 7200       # 2 hours in seconds (7200)
    }
    
    # ========================================================================
    # VALIDATE PATHS
    # ========================================================================
    print("Checking configuration...")
    print(f"LAZ directory: {config['laz_dir']}")
    print(f"Labels CSV: {config['labels_csv']}")
    
    if not os.path.exists(config['laz_dir']):
        print(f"\n✗ ERROR: LAZ directory not found!")
        print(f"  Expected: {config['laz_dir']}")
        print("\n  Please update 'laz_dir' in the config section above.")
        return
    
    if not os.path.exists(config['labels_csv']):
        print(f"\n✗ ERROR: Labels CSV not found!")
        print(f"  Expected: {config['labels_csv']}")
        print("\n  Please update 'labels_csv' in the config section above.")
        return
    
    print("✓ Paths validated successfully!\n")
    
    # ========================================================================
    # CHECK DEPENDENCIES
    # ========================================================================
    print("Checking dependencies...")
    try:
        import laspy
        print("✓ laspy installed")
    except ImportError:
        print("\n✗ ERROR: laspy not installed!")
        print("  Install with: pip install laspy[lazrs]")
        return
    
    # ========================================================================
    # CREATE DATALOADERS
    # ========================================================================
    print("\nLoading dataset...")
    try:
        train_loader, val_loader = create_biodiv_dataloaders(
            config['laz_dir'],
            config['labels_csv'],
            batch_size=config['batch_size'],
            num_points=config['num_points'],
            num_workers=config['num_workers']
        )
    except Exception as e:
        print(f"\n✗ ERROR loading dataset: {e}")
        print("\nPlease check:")
        print("  1. LAZ files are accessible")
        print("  2. Labels CSV is properly formatted")
        print("  3. laspy is installed: pip install laspy[lazrs]")
        return
    
    if len(train_loader) == 0:
        print("\n✗ ERROR: No training data found!")
        print(f"Please check that LAZ files exist in: {config['laz_dir']}")
        return
    
    print(f"✓ Data loaded successfully!")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    
    # ========================================================================
    # START TRAINING
    # ========================================================================
    print("\n" + "="*60)
    print("READY TO START TRAINING")
    print("="*60)
    print(f"Training will run for {config['time_limit']/3600:.1f} hours")
    print(f"Press Ctrl+C to stop early (progress will be saved)")
    print("="*60 + "\n")
    
    try:
        trainer = TreeSegmentationTrainer(config)
        best_iou = trainer.train(train_loader, val_loader)
        
        print(f"\n✓ Training finished successfully!")
        print(f"✓ Best IoU: {best_iou:.4f}")
        print(f"✓ Best model saved to: {config['output_dir']}/checkpoints/best.pth")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Progress has been saved in checkpoints.")
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
