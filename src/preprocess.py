"""
preprocess.py
This implements the Ben Graham's preprocessing method for DR detection

Ben Graham's Method is as below:
1. Crop to remove black borders (circle cropping)
2. Resize to uniform dimensions
3. Apply local averaging and color normalization
4. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)

"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import os
from tqdm import tqdm


def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Here, we crop out the black borders from image
    
    Args:
        img: Input image (BGR or grayscale)
        tol: Tolerance for black pixel detection (0-255)
        
    Returns:
        Cropped image
    """
    # Convert to grayscale if needed
    if img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    
    # Create mask of non-black pixels
    mask = gray_img > tol
    
    # Find coordinates of non-black pixels
    coords = np.argwhere(mask)
    
    if len(coords) == 0:
        return img  # Return original if all black
    
    # Get bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Crop the original image
    cropped_img = img[y_min:y_max+1, x_min:x_max+1]
    
    return cropped_img


def circle_crop(img: np.ndarray, sigmaX: int = 10) -> np.ndarray:
    """
    Here, we perform circular crop focusing on the retina area
    It reduces irrelevant background and focuses model on retinal features
    
    Args:
        img: Input image
        sigmaX: Gaussian blur sigma (smoothing parameter)
        
    Returns:
        Circle-cropped image
    """
    height, width = img.shape[:2]
    
    # Find center and radius
    x_center = width // 2
    y_center = height // 2
    radius = min(x_center, y_center)
    
    # Create circular mask
    y_coords, x_coords = np.ogrid[:height, :width]
    mask = (x_coords - x_center)**2 + (y_coords - y_center)**2 <= radius**2
    
    # Apply mask
    masked_img = img.copy()
    masked_img[~mask] = 0  # Set pixels outside circle to black
    
    # Apply Gaussian blur to smooth edges
    blurred = cv2.GaussianBlur(masked_img, (0, 0), sigmaX)
    
    # Crop to bounding box of circle
    cropped = crop_image_from_gray(blurred, tol=7)
    
    return cropped


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    It enhances local contrast, making lesions more visible
    
    Args:
        img: Input BGR image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        CLAHE-enhanced image
    """
    # Convert to LAB color space (better for medical images)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split into L, A, B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel (lightness)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    # Merge channels back
    limg = cv2.merge((cl, a, b))
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced


def subtract_local_average(img: np.ndarray, kernel_size: int = 50) -> np.ndarray:
    """
    Subtract local average color to normalize illumination
    It removes uneven lighting and background variations
    
    Args:
        img: Input BGR image
        kernel_size: Size of averaging kernel (odd number)
        
    Returns:
        Normalized image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Apply Gaussian blur to get local average
    local_avg = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # Subtract local average (with offset to avoid negative values)
    normalized = cv2.addWeighted(img, 4, local_avg, -4, 128)
    
    return normalized


def ben_graham_preprocessing(
    image_path: Union[str, Path],
    output_size: Tuple[int, int] = (512, 512),
    apply_clahe_flag: bool = True,
    apply_local_avg: bool = True,
    sigmaX: int = 10,
    save_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    Complete Ben Graham's preprocessing pipeline
    
    Pipeline:
    1. Load image
    2. Circle crop (remove black borders, focus on retina)
    3. Resize to uniform dimensions
    4. Subtract local average (normalize illumination) - optional
    5. Apply CLAHE (enhance contrast) - optional
    
    Args:
        image_path: Path to input image
        output_size: Target dimensions (height, width)
        apply_clahe_flag: Whether to apply CLAHE enhancement
        apply_local_avg: Whether to apply local average subtraction
        sigmaX: Gaussian blur sigma for circle crop
        save_path: Optional path to save processed image
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    img = cv2.imread(str(image_path))
    
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Step 1: Circle crop and remove black borders
    img = circle_crop(img, sigmaX=sigmaX)
    
    # Step 2: Resize to uniform dimensions
    img = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)
    
    # Step 3: Subtract local average (optional)
    if apply_local_avg:
        img = subtract_local_average(img, kernel_size=51)
    
    # Step 4: Apply CLAHE (optional)
    if apply_clahe_flag:
        img = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img)
    
    return img


def preprocess_batch(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    image_ids: Optional[list] = None,
    output_size: Tuple[int, int] = (512, 512),
    apply_clahe_flag: bool = True,
    apply_local_avg: bool = True,
    verbose: bool = True
) -> dict:
    """
    Batch preprocess multiple images
    
    Args:
        input_dir: Directory containing raw images
        output_dir: Directory to save processed images
        image_ids: List of image IDs to process (without extension). If None, process all
        output_size: Target dimensions
        apply_clahe_flag: Whether to apply CLAHE
        apply_local_avg: Whether to apply local average subtraction
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with processing statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of images to process
    if image_ids is None:
        image_files = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg'))
        image_ids = [f.stem for f in image_files]
    
    stats = {
        'total': len(image_ids),
        'successful': 0,
        'failed': 0,
        'failed_ids': []
    }
    
    # Process with progress bar
    iterator = tqdm(image_ids, desc="Preprocessing") if verbose else image_ids
    
    for img_id in iterator:
        try:
            # Find input file (try both .png and .jpg)
            input_path = input_dir / f"{img_id}.png"
            if not input_path.exists():
                input_path = input_dir / f"{img_id}.jpg"
            
            if not input_path.exists():
                if verbose:
                    print(f"Warning: Could not find {img_id}")
                stats['failed'] += 1
                stats['failed_ids'].append(img_id)
                continue
            
            # Output path
            output_path = output_dir / f"{img_id}.png"
            
            # Skip if already processed
            if output_path.exists():
                stats['successful'] += 1
                continue
            
            # Preprocess
            ben_graham_preprocessing(
                image_path=input_path,
                output_size=output_size,
                apply_clahe_flag=apply_clahe_flag,
                apply_local_avg=apply_local_avg,
                save_path=output_path
            )
            
            stats['successful'] += 1
            
        except Exception as e:
            if verbose:
                print(f"Error processing {img_id}: {e}")
            stats['failed'] += 1
            stats['failed_ids'].append(img_id)
    
    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total images:      {stats['total']}")
        print(f"Successfully processed: {stats['successful']}")
        print(f"Failed:            {stats['failed']}")
        if stats['failed'] > 0:
            print(f"Failed IDs: {stats['failed_ids'][:10]}...")
        print("=" * 60)
    
    return stats


def compare_preprocessing(
    image_path: Union[str, Path],
    output_size: Tuple[int, int] = (512, 512)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare different preprocessing stages for visualization
    Useful for understanding what each step does
    
    Args:
        image_path: Path to input image
        output_size: Target dimensions
        
    Returns:
        Tuple of (original, cropped, normalized, final_enhanced)
    """
    # Original
    original = cv2.imread(str(image_path))
    original_resized = cv2.resize(original, output_size)
    
    # After circle crop
    cropped = circle_crop(original.copy(), sigmaX=10)
    cropped_resized = cv2.resize(cropped, output_size)
    
    # After local average subtraction
    normalized = subtract_local_average(cropped_resized.copy(), kernel_size=51)
    
    # Final (with CLAHE)
    final = apply_clahe(normalized.copy(), clip_limit=2.0, tile_grid_size=(8, 8))
    
    return original_resized, cropped_resized, normalized, final


def load_preprocessed_image(
    image_path: Union[str, Path],
    normalize: bool = True
) -> np.ndarray:
    """
    Load a preprocessed image and optionally normalize to [0, 1]
    Use this when loading images for model training
    
    Args:
        image_path: Path to preprocessed image
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Image array, optionally normalized
    """
    img = cv2.imread(str(image_path))
    
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB (Keras/TensorFlow expects RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if normalize:
        img = img.astype(np.float32) / 255.0
    
    return img


def main():
    """
    Demo of preprocessing functions
    Run: python src/preprocess.py
    """
    print("\n" + "=" * 60)
    print("Ben Graham's Preprocessing - Demo")
    print("=" * 60)
    
    # Example: preprocess a single image
    print("\n📸 Example: Preprocessing a single image")
    print("-" * 60)
    
    # Note: Update this path to an actual image from your dataset
    sample_image = "data/raw/train_images/000c1434d8d7.png"
    
    if Path(sample_image).exists():
        print(f"Input: {sample_image}")
        
        processed = ben_graham_preprocessing(
            image_path=sample_image,
            output_size=(512, 512),
            save_path="data/processed/sample_processed.png"
        )
        
        print(f"✓ Processed image saved to: data/processed/sample_processed.png")
        print(f"  Shape: {processed.shape}")
        print(f"  Data type: {processed.dtype}")
        print(f"  Value range: [{processed.min()}, {processed.max()}]")
    else:
        print(f"⚠️  Sample image not found: {sample_image}")
        print("   Update the path to test preprocessing")
    
    print("\n" + "=" * 60)
    print("✅ Preprocessing module ready!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Test on sample images in notebooks/02_preprocess.ipynb")
    print("  2. Run batch processing on full dataset")
    print("  3. Use processed images for model training")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()