"""
data_loader.py
Handles loading and splitting the APTOS 2019 Diabetic Retinopathy dataset

Usage:
    from src.data_loader import APTOSDataLoader
    
    loader = APTOSDataLoader(data_dir='data/raw')
    train_df = loader.load_train_data()
    train_df, val_df = loader.train_val_split(train_df)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from typing import Tuple, List, Dict, Optional
from pathlib import Path

class APTOSDataLoader:
    """
    Data loader for APTOS 2019 Diabetic Retinopathy Detection dataset
    
    Attributes:
        data_dir: Root directory containing the raw data
        train_csv_path: Path to training labels [CSV]
        test_csv_path: Path to test labels [CSV]  
        train_images_dir: Directory with training images
        test_images_dir: Directory with test images
    """
    
    def __init__(self, data_dir: str = 'data/raw'):
        """
        Initialize the data loader
        
        Args:
            data_dir: Path to the raw data directory (default: 'data/raw')
        """
        self.data_dir = Path(data_dir)
        self.train_csv_path = self.data_dir / 'train.csv'
        self.test_csv_path = self.data_dir / 'test.csv'
        self.train_images_dir = self.data_dir / 'train_images'
        self.test_images_dir = self.data_dir / 'test_images'
        
        # Class names for reference
        self.class_names = {
            0: 'No DR',
            1: 'Mild',
            2: 'Moderate', 
            3: 'Severe',
            4: 'Proliferative DR'
        }
        
        # Verify paths exist
        self._verify_paths()
        
    def _verify_paths(self):
        """Check if all required files and folders exist"""
        required_paths = {
            'Training CSV': self.train_csv_path,
            'Test CSV': self.test_csv_path,
            'Training Images': self.train_images_dir,
            'Test Images': self.test_images_dir
        }
        
        missing_paths = []
        for name, path in required_paths.items():
            if not path.exists():
                missing_paths.append(f"{name}: {path}")
        
        if missing_paths:
            raise FileNotFoundError(
                f"Missing required paths:\n" + "\n".join(missing_paths)
            )
        
        print("=" * 60)
        print(" Dataset Verification Complete")
        print("=" * 60)
        for name, path in required_paths.items():
            print(f"   {name}: {path}")
        print("=" * 60)
    
    def load_train_data(self, verify_images: bool = True) -> pd.DataFrame:
        """
        Load training CSV with full image paths and optional verification
        
        Args:
            verify_images: Whether to check if all images exist (default: True)
            
        Return Type:
            DataFrame with columns: id_code, diagnosis, image_path, class_name
        """
        print("\nLoading Training Data...")
        
        # Read CSV
        df = pd.read_csv(self.train_csv_path)
        
        # Add full image paths
        df['image_path'] = df['id_code'].apply(
            lambda x: str(self.train_images_dir / f"{x}.png")
        )
        
        # Add human-readable class names
        df['class_name'] = df['diagnosis'].map(self.class_names)
        
        # Verify images exist
        if verify_images:
            print("   Verifying image files...")
            df['exists'] = df['image_path'].apply(os.path.exists)
            missing = df[~df['exists']]
            
            if len(missing) > 0:
                print(f"WARNING: {len(missing)} images not found!")
                print(f"Missing IDs: {missing['id_code'].tolist()[:5]}...")
                df = df[df['exists']].copy()
            else:
                print("All images verified")
            
            df = df.drop('exists', axis=1)
        
        # Display statistics
        print(f"\nLoaded {len(df)} training samples")
        print(f"\nClass Distribution:")
        print("-" * 50)
        for grade in sorted(df['diagnosis'].unique()):
            count = len(df[df['diagnosis'] == grade])
            percentage = (count / len(df)) * 100
            class_name = self.class_names[grade]
            print(f"  Grade {grade} ({class_name:20s}): {count:5d} ({percentage:5.2f}%)")
        print("-" * 50)
        
        return df
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load test CSV with full image paths
        Note: Test set has no labels (for Kaggle submission)
        
        Returns:
            DataFrame with columns: id_code, image_path
        """
        print("\nLoading Test Data...")
        
        df = pd.read_csv(self.test_csv_path)
        
        # Add full image paths
        df['image_path'] = df['id_code'].apply(
            lambda x: str(self.test_images_dir / f"{x}.png")
        )
        
        print(f"Loaded {len(df)} test samples")
        
        return df
    
    def train_val_split(
        self, 
        df: pd.DataFrame, 
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split training data into train and validation sets with stratification
        Stratification ensures each class is proportionally represented in both sets
        
        Args:
            df: Training DataFrame
            val_size: Validation set proportion (default: 0.15 = 15%)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df)
        """
        print(f"\nSplitting Data (Train: {(1-val_size)*100:.0f}%, Val: {val_size*100:.0f}%)...")
        
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            stratify=df['diagnosis'],  # Critical: maintains class balance
            random_state=random_state
        )
        
        # Reset indices
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        
        print(f"\nSplit Complete:")
        print(f"  Training Set:   {len(train_df):5d} samples")
        print(f"  Validation Set: {len(val_df):5d} samples")
        
        # Show class distribution in both sets
        print(f"\n Training Set Distribution:")
        print("-" * 50)
        for grade in sorted(train_df['diagnosis'].unique()):
            count = len(train_df[train_df['diagnosis'] == grade])
            percentage = (count / len(train_df)) * 100
            print(f"  Grade {grade}: {count:5d} ({percentage:5.2f}%)")
        
        print(f"\n Validation Set Distribution:")
        print("-" * 50)
        for grade in sorted(val_df['diagnosis'].unique()):
            count = len(val_df[val_df['diagnosis'] == grade])
            percentage = (count / len(val_df)) * 100
            print(f"  Grade {grade}: {count:5d} ({percentage:5.2f}%)")
        print("-" * 50)
        
        return train_df, val_df
    
    def get_class_weights(self, df: pd.DataFrame, method: str = 'balanced') -> Dict[int, float]:
        """
        Calculate class weights to handle imbalanced dataset
        APTOS dataset is heavily skewed toward Grade 0
        
        Args:
            df: Training DataFrame
            method: 'balanced' (sklearn-style) or 'inverse' (simple inverse frequency)
            
        Returns:
            Dictionary mapping class index to weight
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        print(f"\n  Calculating Class Weights (method: {method})...")
        
        if method == 'balanced':
            # Sklearn's balanced approach
            classes = np.unique(df['diagnosis'])
            weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=df['diagnosis']
            )
            class_weights = dict(zip(classes, weights))
        else:
            # Simple inverse frequency
            class_counts = df['diagnosis'].value_counts().sort_index()
            total = len(df)
            class_weights = {
                i: total / (len(class_counts) * count) 
                for i, count in class_counts.items()
            }
        
        print(" Class Weights:")
        print("-" * 50)
        for cls in sorted(class_weights.keys()):
            print(f"  Grade {cls} ({self.class_names[cls]:20s}): {class_weights[cls]:.4f}")
        print("-" * 50)
        print("  Higher weights = model pays more attention to that class")
        
        return class_weights
    
    def create_augmentation_generator(
        self,
        mode: str = 'train'
    ) -> ImageDataGenerator:
        """
        Create ImageDataGenerator with appropriate augmentation settings
        
        Args:
            mode: 'train' (with augmentation) or 'val' (no augmentation, only rescaling)
            
        Returns:
            Configured ImageDataGenerator
        """
        if mode == 'train':
            print("\n Creating Training Data Generator (with augmentation)...")
            generator = ImageDataGenerator(
                rescale=1./255,
                rotation_range=15,           # Rotate ±15 degrees
                width_shift_range=0.1,       # Shift horizontally by 10%
                height_shift_range=0.1,      # Shift vertically by 10%
                horizontal_flip=True,        # Mirror image horizontally
                vertical_flip=True,          # Mirror image vertically
                zoom_range=0.1,              # Zoom in/out by 10%
                fill_mode='constant',        # Fill empty pixels with black
                cval=0                       # Black color value
            )
            print("   Augmentation enabled: rotation, flips, shifts, zoom")
        else:
            print("\n Creating Validation Data Generator (no augmentation)...")
            generator = ImageDataGenerator(
                rescale=1./255  # Only normalize pixel values
            )
            print("   Only rescaling applied")
        
        return generator
    
    def check_image_quality(
        self, 
        df: pd.DataFrame, 
        sample_size: int = 100,
        check_dimensions: bool = True
    ) -> Dict[str, List]:
        """
        Perform quality checks on images
        
        Args:
            df: DataFrame with image paths
            sample_size: Number of images to check (None = check all)
            check_dimensions: Whether to record image dimensions
            
        Returns:
            Dictionary with 'corrupted', 'dimensions' lists
        """
        if sample_size is None or sample_size > len(df):
            sample_size = len(df)
            
        print(f"\n Quality Check: Inspecting {sample_size} images...")
        
        sample = df.sample(min(sample_size, len(df)), random_state=42)
        corrupted = []
        dimensions = []
        
        for idx, row in sample.iterrows():
            try:
                img = cv2.imread(row['image_path'])
                if img is None:
                    corrupted.append(row['id_code'])
                elif check_dimensions:
                    dimensions.append({
                        'id': row['id_code'],
                        'height': img.shape[0],
                        'width': img.shape[1],
                        'channels': img.shape[2] if len(img.shape) == 3 else 1
                    })
            except Exception as e:
                corrupted.append(row['id_code'])
                print(f"   Error reading {row['id_code']}: {e}")
        
        # Report results
        if len(corrupted) > 0:
            print(f"    Found {len(corrupted)} corrupted images: {corrupted[:5]}")
        else:
            print(f"   All {sample_size} sampled images are readable")
        
        if check_dimensions and len(dimensions) > 0:
            dims_df = pd.DataFrame(dimensions)
            print(f"\n Image Dimensions Summary:")
            print(f"  Heights: {dims_df['height'].min()} - {dims_df['height'].max()}")
            print(f"  Widths:  {dims_df['width'].min()} - {dims_df['width'].max()}")
            print(f"  Mode:    {dims_df['height'].mode().values[0]}x{dims_df['width'].mode().values[0]}")
        
        return {
            'corrupted': corrupted,
            'dimensions': dimensions
        }
    
    def get_sample_images(
        self, 
        df: pd.DataFrame, 
        n_per_class: int = 2
    ) -> Dict[int, List[str]]:
        """
        Get sample image paths for each class
        For visualization in EDA
        
        Args:
            df: DataFrame with images
            n_per_class: Number of samples per class
            
        Returns:
            Dictionary mapping class -> list of image paths
        """
        samples = {}
        for grade in sorted(df['diagnosis'].unique()):
            grade_df = df[df['diagnosis'] == grade]
            sample_paths = grade_df.sample(
                min(n_per_class, len(grade_df)), 
                random_state=42
            )['image_path'].tolist()
            samples[grade] = sample_paths
        
        return samples
    
    def save_split_info(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame,
        output_dir: str = 'data/processed'
    ):
        """
        Save train/val split information for reproducibility
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            output_dir: Directory to save CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_path = output_path / 'train_split.csv'
        val_path = output_path / 'val_split.csv'
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        print(f"\n Saved split information:")
        print(f"  Training:   {train_path}")
        print(f"  Validation: {val_path}")


def main():
    """
    Demonstration of APTOSDataLoader usage
    Run this script directly to test: python src/data_loader.py
    """
    print("\n" + "=" * 60)
    print("APTOS Data Loader")
    print("=" * 60)
    
    # Initialize loader
    loader = APTOSDataLoader(data_dir='data/raw')
    
    # Load training data
    train_df = loader.load_train_data(verify_images=True)
    
    # Check image quality
    quality_report = loader.check_image_quality(
        train_df, 
        sample_size=100,
        check_dimensions=True
    )
    
    # Split into train/validation
    train_df, val_df = loader.train_val_split(train_df, val_size=0.15)
    
    # Calculate class weights
    class_weights = loader.get_class_weights(train_df, method='balanced')
    
    # Create data generators
    train_gen = loader.create_augmentation_generator(mode='train')
    val_gen = loader.create_augmentation_generator(mode='val')
    
    # Get sample images for visualization
    samples = loader.get_sample_images(train_df, n_per_class=3)
    
    # Save split info
    loader.save_split_info(train_df, val_df)
    
    print("\n" + "=" * 60)
    print(" Data Loading Pipeline Ready!")
    print("=" * 60)
    print("\nNext Steps:")
    print("  1. Run notebooks/01_eda.ipynb for exploratory analysis")
    print("  2. Use train_df and val_df for model training")
    print("  3. Apply class_weights during model.fit()")
    print("=" * 60 + "\n")
    

if __name__ == "__main__":
    main()