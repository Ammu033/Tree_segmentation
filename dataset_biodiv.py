import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import glob


class BioDivTreeDataset(Dataset):
    """
    Dataset for BioDiv-3DTrees with LAZ files and separate CSV labels
    
    Structure expected:
    - TLS/TLS/*.laz (point cloud files)
    - labels.csv (labels for each tree)
    """
    def __init__(self, laz_dir, labels_csv, num_points=2048, split='train', test_size=0.2):
        self.num_points = num_points
        self.laz_dir = laz_dir
        
        # Check for laspy/laszip
        self._check_laz_support()
        
        # Load labels CSV
        print(f"Loading labels from {labels_csv}...")
        self.labels_df = pd.read_csv(labels_csv)
        print(f"Found {len(self.labels_df)} entries in labels CSV")
        print(f"Label columns: {list(self.labels_df.columns)}")
        
        # Find all LAZ files
        self.laz_files = self._load_file_list(laz_dir)
        print(f"Found {len(self.laz_files)} LAZ files")
        
        # Match files with labels
        self.matched_data = self._match_files_with_labels()
        print(f"Successfully matched {len(self.matched_data)} files with labels")
        
        # Split into train/val
        if len(self.matched_data) > 1:
            train_data, val_data = train_test_split(
                self.matched_data, 
                test_size=test_size, 
                random_state=42
            )
            self.data = train_data if split == 'train' else val_data
        else:
            self.data = self.matched_data
        
        print(f"{split} dataset: {len(self.data)} samples")
    
    def _check_laz_support(self):
        """Check if LAZ reading is supported"""
        try:
            import laspy
            self.laspy = laspy
            print("✓ laspy library found")
        except ImportError:
            print("ERROR: laspy not found. Installing...")
            print("Run: pip install laspy[lazrs]")
            raise ImportError("Please install laspy: pip install laspy[lazrs]")
    
    def _load_file_list(self, laz_dir):
        """Load all LAZ files"""
        laz_pattern = os.path.join(laz_dir, '**', '*.laz')
        files = glob.glob(laz_pattern, recursive=True)
        return sorted(files)
    
    def _extract_tree_id(self, filename):
        """
        Extract tree ID from filename to match CSV format
        Filename: AEW03_GD_104_TLS.laz -> AEW03_G_104 (matches CSV)
        """
        basename = os.path.basename(filename)
        name = basename.replace('.laz', '').replace('.LAZ', '')
        parts = name.split('_')
        
        # Find numeric ID
        numeric_id = None
        for part in parts:
            if part.isdigit():
                numeric_id = part
                break
        
        if numeric_id and len(parts) >= 3:
            # Format: AEW03_GD_104_TLS
            # CSV format: AEW03_G_104
            prefix = parts[0]  # AEW03
            middle = parts[1]   # GD
            
            # Take first letter of middle part
            if len(middle) > 0 and middle[0].isalpha():
                letter = middle[0]  # G
                return f"{prefix}_{letter}_{numeric_id}"
        
        # Fallback: try full name without TLS suffix
        if 'TLS' in parts:
            parts.remove('TLS')
        return '_'.join(parts)
    
    def _match_files_with_labels(self):
        """Match LAZ files with labels from CSV"""
        matched = []
        
        # Try to understand the CSV structure
        print("\nAnalyzing CSV structure...")
        print(f"First few rows:\n{self.labels_df.head()}")
        
        # For BioDiv-3DTrees, we know the structure
        id_col = 'treeID'  # Main identifier
        
        # Choose label column based on what's available and makes sense
        label_col = None
        
        # Priority order for label columns
        if 'type' in self.labels_df.columns:
            # Use tree type (conifer/deciduous) for binary classification
            label_col = 'type'
            print(f"✓ Using 'type' column for labels (conifer/deciduous)")
            
            # Convert type to binary labels
            unique_types = self.labels_df['type'].unique()
            print(f"  Found types: {unique_types}")
            
            # Create label mapping
            type_mapping = {}
            for i, tree_type in enumerate(unique_types):
                type_mapping[tree_type] = i
            print(f"  Label mapping: {type_mapping}")
            
        elif 'species' in self.labels_df.columns:
            # Use species for multi-class classification
            label_col = 'species'
            unique_species = self.labels_df['species'].unique()
            print(f"✓ Using 'species' column for labels")
            print(f"  Found {len(unique_species)} species: {unique_species[:5]}...")
            
            # Create label mapping
            type_mapping = {}
            for i, species in enumerate(unique_species):
                type_mapping[species] = i
            print(f"  Will map to {len(type_mapping)} classes")
            
        else:
            # Fallback to binary (all trees = 1)
            label_col = None
            type_mapping = None
            print("⚠ No suitable label column found, using binary (all=tree)")
        
        # Create ID to label mapping
        id_to_label = {}
        if label_col and type_mapping:
            for _, row in self.labels_df.iterrows():
                tree_id = str(row[id_col])
                label_value = row[label_col]
                # Map to numeric label
                if label_value in type_mapping:
                    id_to_label[tree_id] = type_mapping[label_value]
                else:
                    id_to_label[tree_id] = 0  # Default
        elif label_col:
            # Direct numeric labels
            for _, row in self.labels_df.iterrows():
                tree_id = str(row[id_col])
                id_to_label[tree_id] = row[label_col]
        
        # Match files
        unmatched = 0
        for laz_file in self.laz_files:
            tree_id = self._extract_tree_id(laz_file)
            
            # Try exact match
            if tree_id in id_to_label:
                label = id_to_label[tree_id]
            else:
                # Try partial matching
                matched_id = None
                for csv_id in id_to_label.keys():
                    if tree_id in csv_id or csv_id in tree_id:
                        matched_id = csv_id
                        break
                
                if matched_id:
                    label = id_to_label[matched_id]
                else:
                    # Default: assume all are trees (label=1)
                    label = 1
                    unmatched += 1
            
            matched.append({
                'file': laz_file,
                'tree_id': tree_id,
                'label': label
            })
        
        if unmatched > 0:
            print(f"⚠ Warning: {unmatched} files could not be matched with CSV labels")
            print(f"  These will be assigned default label=1 (tree)")
        
        return matched
    
    def _load_laz_file(self, filepath):
        """Load LAZ file and extract XYZ coordinates"""
        try:
            las = self.laspy.read(filepath)
            
            # Extract XYZ coordinates
            x = np.array(las.x)
            y = np.array(las.y)
            z = np.array(las.z)
            
            points = np.column_stack([x, y, z])
            
            return points
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def _normalize_point_cloud(self, points):
        """Normalize point cloud to unit sphere"""
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_dist > 0:
            points = points / max_dist
        return points
    
    def _random_sample(self, points, n_points):
        """Randomly sample fixed number of points"""
        n = points.shape[0]
        
        if n >= n_points:
            choice = np.random.choice(n, n_points, replace=False)
        else:
            choice = np.random.choice(n, n_points, replace=True)
        
        return points[choice, :]
    
    def _create_segmentation_labels(self, points, tree_label):
        """
        Create per-point segmentation labels
        For tree segmentation: all points in a tree LAZ file are tree points
        For complete scene: need to identify tree vs background
        """
        n_points = points.shape[0]
        
        # If tree_label indicates this is a tree, mark all points as tree
        if tree_label == 1:
            labels = np.ones(n_points, dtype=np.int64)
        else:
            # If background or mixed scene, use height-based heuristic
            # Points above median height are more likely to be tree
            z_coords = points[:, 2]
            threshold = np.median(z_coords)
            labels = (z_coords > threshold).astype(np.int64)
        
        return labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        filepath = item['file']
        tree_label = item['label']
        
        # Load point cloud
        points = self._load_laz_file(filepath)
        
        if points is None or len(points) == 0:
            # Return dummy data if file couldn't be loaded
            points = np.random.randn(self.num_points, 3).astype(np.float32)
            labels = np.ones(self.num_points, dtype=np.int64)
            return torch.from_numpy(points).transpose(0, 1), torch.from_numpy(labels)
        
        # Normalize
        points = self._normalize_point_cloud(points)
        
        # Sample fixed number of points
        points = self._random_sample(points, self.num_points)
        
        # Create per-point labels
        labels = self._create_segmentation_labels(points, tree_label)
        
        # Data augmentation (if training)
        if hasattr(self, 'augment') and self.augment:
            points = self._augment_point_cloud(points)
        
        # Convert to torch tensors
        points = torch.from_numpy(points.astype(np.float32))
        labels = torch.from_numpy(labels[:self.num_points])  # Ensure correct size
        
        # Transpose points to (3, N) for PointNet++
        points = points.transpose(0, 1)
        
        return points, labels
    
    def _augment_point_cloud(self, points):
        """Apply data augmentation"""
        # Random rotation around Z-axis
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        points = points @ rotation_matrix.T
        
        # Random jittering
        points += np.random.normal(0, 0.02, size=points.shape)
        
        # Random scaling
        scale = np.random.uniform(0.8, 1.2)
        points *= scale
        
        return points


def create_biodiv_dataloaders(laz_dir, labels_csv, batch_size=8, num_points=2048, num_workers=4):
    """
    Create train and validation dataloaders for BioDiv-3DTrees
    
    Args:
        laz_dir: Path to directory containing LAZ files (e.g., 'C:/Users/Student/Downloads/TLS/TLS')
        labels_csv: Path to labels CSV file (e.g., 'C:/Users/Student/Downloads/labels.csv')
        batch_size: Batch size for training
        num_points: Number of points to sample per cloud
        num_workers: Number of workers for data loading
    """
    train_dataset = BioDivTreeDataset(
        laz_dir, 
        labels_csv, 
        num_points=num_points, 
        split='train'
    )
    train_dataset.augment = True
    
    val_dataset = BioDivTreeDataset(
        laz_dir, 
        labels_csv, 
        num_points=num_points, 
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing BioDiv-3DTrees dataset loader...")
    
    # Update these paths to match your setup
    laz_dir = r"C:\Users\Student\Downloads\TLS\TLS"
    labels_csv = r"C:\Users\Student\Downloads\labels.csv"
    
    if os.path.exists(laz_dir) and os.path.exists(labels_csv):
        print(f"\n✓ LAZ directory found: {laz_dir}")
        print(f"✓ Labels CSV found: {labels_csv}")
        
        dataset = BioDivTreeDataset(laz_dir, labels_csv, num_points=2048, split='train')
        print(f"\nDataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            print("\nLoading first sample...")
            points, labels = dataset[0]
            print(f"✓ Points shape: {points.shape}")
            print(f"✓ Labels shape: {labels.shape}")
            print(f"✓ Unique labels: {torch.unique(labels)}")
            print(f"✓ Points range: [{points.min():.3f}, {points.max():.3f}]")
    else:
        print(f"\n✗ Could not find data:")
        if not os.path.exists(laz_dir):
            print(f"  - LAZ directory not found: {laz_dir}")
        if not os.path.exists(labels_csv):
            print(f"  - Labels CSV not found: {labels_csv}")
        print("\nPlease update the paths in the script.")