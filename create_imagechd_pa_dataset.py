import os
import glob
import shutil
import numpy as np
import nibabel as nib
from tqdm import tqdm
import argparse

def create_pa_only_dataset(source_path, dest_path, pa_label=7, only_pa_cases=True):
    """
    Create a new dataset with only pulmonary artery segmentations from ImageCHD dataset.
    
    Parameters:
    -----------
    source_path : str
        Path to the original ImageCHD dataset
    dest_path : str
        Path where the new dataset will be saved
    pa_label : int
        Label value for pulmonary artery (default: 7)
    only_pa_cases : bool
        If True, only include cases that have PA segmentations
        If False, include all cases but zero out non-PA segmentations
    """
    # Create destination directory if it doesn't exist
    os.makedirs(dest_path, exist_ok=True)
    
    # Find all label files in the source dataset (excluding hidden files)
    label_files = sorted([f for f in glob.glob(os.path.join(source_path, "*label.nii.gz")) 
                      if not os.path.basename(f).startswith('._')])
    
    if not label_files:
        raise FileNotFoundError(f"No label files found in {source_path}")
    
    print(f"Found {len(label_files)} label files in the original dataset.")
    
    # Initialize counters
    processed_cases = 0
    cases_with_pa = 0
    
    # Process each case
    for label_file in tqdm(label_files, desc="Processing cases"):
        try:
            # Get corresponding image file
            case_id = os.path.basename(label_file).split('_')[1]
            image_file = os.path.join(source_path, f"ct_{case_id}_image.nii.gz")
            
            # Check if image file exists
            if not os.path.exists(image_file):
                print(f"Warning: Image file for case {case_id} not found, skipping.")
                continue
            
            # Load the label file
            nii_label = nib.load(label_file)
            label_data = nii_label.get_fdata()
            
            # Fix for floating-point precision: use np.isclose() for comparison
            # Consider values close to PA label as the PA label
            has_pa = np.any(np.isclose(label_data, pa_label, rtol=1e-5))
            
            # Skip case if it doesn't have PA and we're only keeping PA cases
            if only_pa_cases and not has_pa:
                continue
            
            # Create PA-only label data - make sure to convert to integer type
            pa_only_data = np.zeros_like(label_data, dtype=np.int16)
            if has_pa:
                # Use isclose instead of exact equality for PA label detection
                pa_mask = np.isclose(label_data, pa_label, rtol=1e-5)
                pa_only_data[pa_mask] = pa_label
                cases_with_pa += 1
            
            # Save the modified label file
            pa_label_nii = nib.Nifti1Image(pa_only_data, nii_label.affine, nii_label.header)
            pa_label_file = os.path.join(dest_path, f"ct_{case_id}_label.nii.gz")
            nib.save(pa_label_nii, pa_label_file)
            
            # Copy the corresponding image file
            dest_image_file = os.path.join(dest_path, f"ct_{case_id}_image.nii.gz")
            shutil.copy2(image_file, dest_image_file)
            
            processed_cases += 1
            
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
    
    print("\n" + "="*50)
    print("ImageCHD PA-Only Dataset Creation Summary")
    print("="*50)
    print(f"Total cases in original dataset: {len(label_files)}")
    print(f"Cases with PA segmentations: {cases_with_pa}")
    print(f"Total cases in new dataset: {processed_cases}")
    print(f"New dataset saved to: {dest_path}")
    
    # Create a README file with dataset information
    with open(os.path.join(dest_path, "README.txt"), "w") as f:
        f.write("Pulmonary Artery Segmentation Dataset (from ImageCHD)\n")
        f.write("="*50 + "\n\n")
        f.write(f"This dataset is derived from the ImageCHD dataset.\n")
        f.write(f"Only Pulmonary Artery segmentations (label {pa_label}) are included.\n\n")
        f.write(f"Original dataset size: {len(label_files)} cases\n")
        f.write(f"Cases with PA segmentations: {cases_with_pa}\n")
        f.write(f"Cases included in this dataset: {processed_cases}\n\n")
        f.write(f"Dataset created on: {import_date()}\n")
        if only_pa_cases:
            f.write("Note: Only cases with PA segmentations are included in this dataset.\n")
        else:
            f.write("Note: All cases are included, but only PA segmentations are preserved.\n")
    
    return processed_cases, cases_with_pa

def verify_dataset(dest_path, pa_label=7):
    """
    Verify the newly created dataset to ensure all label files
    only contain the pulmonary artery label.
    
    Parameters:
    -----------
    dest_path : str
        Path to the new dataset
    pa_label : int
        Label value for pulmonary artery (default: 7)
    """
    print("\nVerifying the new dataset...")
    
    # Find all label files in the new dataset
    label_files = sorted([f for f in glob.glob(os.path.join(dest_path, "*label.nii.gz"))])
    
    if not label_files:
        print("No label files found in the new dataset.")
        return
    
    valid_files = 0
    issues_found = 0
    
    for label_file in tqdm(label_files, desc="Verifying files"):
        try:
            nii_label = nib.load(label_file)
            label_data = nii_label.get_fdata()
            
            # Check for unique labels - with int16 conversion
            label_data_int = np.round(label_data).astype(np.int16)
            unique_labels = np.unique(label_data_int)
            
            # Verify that only 0 (background) and PA label exist
            valid = len(unique_labels) <= 2 and all(label in [0, pa_label] for label in unique_labels)
            
            if valid:
                valid_files += 1
            else:
                issues_found += 1
                print(f"Issue found in {os.path.basename(label_file)}: Contains labels {unique_labels}")
                
        except Exception as e:
            print(f"Error verifying {label_file}: {e}")
            issues_found += 1
    
    print(f"\nVerification complete:")
    print(f"- Valid files: {valid_files}")
    print(f"- Issues found: {issues_found}")
    
    if issues_found == 0:
        print("✅ All files verified successfully! The dataset contains only PA segmentations.")
    else:
        print("⚠️ Some issues were found. Please check the output above.")

def calculate_dataset_statistics(dest_path, pa_label=7):
    """
    Calculate basic statistics for the new PA-only dataset.
    
    Parameters:
    -----------
    dest_path : str
        Path to the new dataset
    pa_label : int
        Label value for pulmonary artery (default: 7)
    """
    print("\nCalculating statistics for the new dataset...")
    
    # Find all label files in the new dataset
    label_files = sorted([f for f in glob.glob(os.path.join(dest_path, "*label.nii.gz"))])
    
    if not label_files:
        print("No label files found in the new dataset.")
        return
    
    total_slices = 0
    slices_with_pa = 0
    total_pa_voxels = 0
    case_stats = []
    
    for label_file in tqdm(label_files, desc="Calculating statistics"):
        try:
            nii_label = nib.load(label_file)
            label_data = nii_label.get_fdata()
            case_id = os.path.basename(label_file).split('_')[1]
            
            # Convert to int16 to eliminate floating-point issues
            label_data_int = np.round(label_data).astype(np.int16)
            
            # Get voxel dimensions for volume calculation
            voxel_dims = nii_label.header.get_zooms()
            voxel_volume = np.prod(voxel_dims)
            
            # Count PA voxels
            pa_voxels = np.sum(label_data_int == pa_label)
            pa_volume = pa_voxels * voxel_volume
            
            # Count slices with PA
            num_slices = label_data.shape[2]  # Assuming axial slices
            slices_with_pa_case = 0
            
            for slice_idx in range(num_slices):
                slice_data = label_data_int[:,:,slice_idx]
                if pa_label in np.unique(slice_data):
                    slices_with_pa_case += 1
            
            total_slices += num_slices
            slices_with_pa += slices_with_pa_case
            total_pa_voxels += pa_voxels
            
            case_stats.append({
                "case_id": case_id,
                "total_slices": num_slices,
                "slices_with_pa": slices_with_pa_case,
                "pa_voxels": pa_voxels,
                "pa_volume_mm3": pa_volume
            })
            
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
    
    # Calculate summary statistics
    avg_pa_slices = slices_with_pa / len(label_files) if len(label_files) > 0 else 0
    pa_slice_percentage = (slices_with_pa / total_slices) * 100 if total_slices > 0 else 0
    
    print("\n" + "="*50)
    print("ImageCHD PA-Only Dataset Statistics")
    print("="*50)
    print(f"Total cases: {len(label_files)}")
    print(f"Total slices: {total_slices}")
    print(f"Slices with PA: {slices_with_pa} ({pa_slice_percentage:.2f}%)")
    print(f"Average slices with PA per case: {avg_pa_slices:.2f}")
    print(f"Total PA voxels: {total_pa_voxels}")
    
    # Create a statistics file
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Save case statistics to CSV
        df_stats = pd.DataFrame(case_stats)
        df_stats.to_csv(os.path.join(dest_path, "pa_case_statistics.csv"), index=False)
        
        # Create visualizations only if there are PA voxels
        if total_pa_voxels > 0:
            sns.set(style="whitegrid")
            
            # 1. Histogram of PA volumes
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            sns.histplot(df_stats["pa_volume_mm3"], bins=20, kde=True)
            plt.title("Distribution of PA Volumes (mm³)")
            plt.xlabel("Volume (mm³)")
            
            # 2. Slices with PA per case
            plt.subplot(2, 2, 2)
            df_stats = df_stats.sort_values("slices_with_pa", ascending=False)
            top_cases = df_stats.head(10) if len(df_stats) > 10 else df_stats
            sns.barplot(x="case_id", y="slices_with_pa", data=top_cases)
            plt.title("Top Cases by Number of Slices with PA")
            plt.xticks(rotation=45)
            
            # 3. PA volume per case
            plt.subplot(2, 2, 3)
            df_stats = df_stats.sort_values("pa_volume_mm3", ascending=False)
            top_cases = df_stats.head(10) if len(df_stats) > 10 else df_stats
            sns.barplot(x="case_id", y="pa_volume_mm3", data=top_cases)
            plt.title("Top Cases by PA Volume")
            plt.xticks(rotation=45)
            
            # 4. Percentage of slices with PA per case
            if total_slices > 0:
                plt.subplot(2, 2, 4)
                df_stats["percent_slices_with_pa"] = (df_stats["slices_with_pa"] / df_stats["total_slices"]) * 100
                sns.histplot(df_stats["percent_slices_with_pa"], bins=20, kde=True)
                plt.title("Distribution of Percentage of Slices with PA")
                plt.xlabel("Percentage of Slices with PA")
            
            plt.tight_layout()
            plt.savefig(os.path.join(dest_path, "pa_dataset_statistics.png"))
        else:
            print("\nSkipping visualizations as no PA voxels were found.")
        
        print(f"\nDetailed statistics saved to {os.path.join(dest_path, 'pa_case_statistics.csv')}")
        if total_pa_voxels > 0:
            print(f"Visualizations saved to {os.path.join(dest_path, 'pa_dataset_statistics.png')}")
        
    except ImportError:
        print("\nStatistics saved, but visualizations require pandas, matplotlib, and seaborn.")

def import_date():
    """Get the current date for documentation purposes."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    parser = argparse.ArgumentParser(description="Create a Pulmonary Artery-only segmentation dataset from ImageCHD dataset")
    parser.add_argument("--source", type=str, default="DATA/ImageCHD_dataset/ImageCHD_dataset",
                        help="Path to the original ImageCHD dataset")
    parser.add_argument("--dest", type=str, default="DATA/ImageCHD_PA_only_dataset",
                        help="Path where the new dataset will be saved")
    parser.add_argument("--pa-label", type=int, default=7,
                        help="Label value for pulmonary artery (default: 7)")
    parser.add_argument("--all-cases", action="store_true",
                        help="Include all cases, not just those with PA segmentations")
    parser.add_argument("--skip-verification", action="store_true",
                        help="Skip dataset verification step")
    parser.add_argument("--skip-statistics", action="store_true",
                        help="Skip dataset statistics calculation")
    
    args = parser.parse_args()
    
    print(f"Creating PA-only dataset from: {args.source}")
    print(f"Saving to: {args.dest}")
    print(f"PA label: {args.pa_label}")
    print(f"Include all cases: {args.all_cases}")
    
    try:
        # Create the PA-only dataset
        processed_cases, cases_with_pa = create_pa_only_dataset(
            args.source, 
            args.dest, 
            pa_label=args.pa_label, 
            only_pa_cases=not args.all_cases
        )
        
        # Verify the dataset
        if not args.skip_verification:
            verify_dataset(args.dest, pa_label=args.pa_label)
        
        # Calculate dataset statistics
        if not args.skip_statistics:
            calculate_dataset_statistics(args.dest, pa_label=args.pa_label)
            
        print("\nProcess completed successfully.")
        
    except Exception as e:
        print(f"Error creating PA-only dataset: {e}")

if __name__ == "__main__":
    main()