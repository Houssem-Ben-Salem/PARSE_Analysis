import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from collections import Counter
from tqdm import tqdm  # For progress bar

def analyze_chd68_dataset(data_path, target_label=7, slice_axis=2):
    """
    Analyze the CHD68 dataset to get statistics about a specific label and overall dataset.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset directory
    target_label : int
        The label to analyze in detail (default is 7 for PA)
    slice_axis : int
        The axis representing slices (0=sagittal, 1=coronal, 2=axial/default)
    
    Returns:
    --------
    dict
        Dictionary containing various statistics about the dataset
    """
    # Find all label files in the dataset (excluding hidden files)
    label_files = sorted([f for f in glob.glob(os.path.join(data_path, "*label.nii.gz")) 
                      if not os.path.basename(f).startswith('._')])
    
    if not label_files:
        raise FileNotFoundError(f"No label files found in {data_path}")
    
    print(f"Found {len(label_files)} label files.")
    
    # Initialize statistics dictionary
    stats = {
        "total_cases": len(label_files),
        "cases_with_target_label": 0,
        "slices_with_target_label": 0,
        "total_slices": 0,
        "labels_found": set(),
        "label_frequencies": {},
        "label_volumes": {},
        "per_case_stats": []
    }
    
    # Analyze each case
    for label_file in tqdm(label_files, desc="Analyzing cases"):
        # Load the label file
        try:
            nii_label = nib.load(label_file)
            label_data = nii_label.get_fdata()
            
            # Get case ID from filename
            case_id = os.path.basename(label_file).split('_')[1]
            
            # Get voxel dimensions for volume calculation
            voxel_dims = nii_label.header.get_zooms()
            voxel_volume = np.prod(voxel_dims)
            
            # Basic case info
            num_slices = label_data.shape[slice_axis]  # Get slices based on specified axis
            stats["total_slices"] += num_slices
            
            # Find unique labels in this case
            unique_labels = np.unique(label_data).astype(int)
            stats["labels_found"].update(unique_labels)
            
            # Count label occurrences in this case
            case_label_counts = {int(label): np.sum(label_data == label) for label in unique_labels if label > 0}
            
            # Check if target label exists in this case
            has_target_label = target_label in unique_labels
            if has_target_label:
                stats["cases_with_target_label"] += 1
                
                # Count slices with target label
                slices_with_target = 0
                for slice_idx in range(num_slices):
                    # Get slice data based on specified axis
                    if slice_axis == 0:
                        slice_data = label_data[slice_idx,:,:]
                    elif slice_axis == 1:
                        slice_data = label_data[:,slice_idx,:]
                    else:  # slice_axis == 2
                        slice_data = label_data[:,:,slice_idx]
                        
                    if target_label in np.unique(slice_data):
                        slices_with_target += 1
                
                stats["slices_with_target_label"] += slices_with_target
            
            # Update label frequencies
            for label, count in case_label_counts.items():
                if label in stats["label_frequencies"]:
                    stats["label_frequencies"][label] += count
                else:
                    stats["label_frequencies"][label] = count
                    
                # Calculate volume in mm³
                if label in stats["label_volumes"]:
                    stats["label_volumes"][label] += count * voxel_volume
                else:
                    stats["label_volumes"][label] = count * voxel_volume
            
            # Store per-case statistics
            case_stats = {
                "case_id": case_id,
                "num_slices": num_slices,
                "unique_labels": list(map(int, unique_labels)),
                "has_target_label": has_target_label,
                "target_label_volume": case_label_counts.get(target_label, 0) * voxel_volume if has_target_label else 0,
                "slices_with_target": slices_with_target if has_target_label else 0
            }
            stats["per_case_stats"].append(case_stats)
            
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
    
    # Convert labels_found to a sorted list
    stats["labels_found"] = sorted(list(map(int, stats["labels_found"])))
    
    # Calculate percentages
    stats["percent_cases_with_target_label"] = (stats["cases_with_target_label"] / stats["total_cases"]) * 100 if stats["total_cases"] > 0 else 0
    stats["percent_slices_with_target_label"] = (stats["slices_with_target_label"] / stats["total_slices"]) * 100 if stats["total_slices"] > 0 else 0
    
    # Calculate label distribution
    total_labeled_voxels = sum(stats["label_frequencies"].values())
    stats["label_distribution"] = {label: (count / total_labeled_voxels) * 100 
                                  for label, count in stats["label_frequencies"].items()}
    
    return stats

def print_statistics(stats, target_label=7):
    """
    Print the statistics in a readable format.
    
    Parameters:
    -----------
    stats : dict
        Dictionary containing the statistics
    target_label : int
        The target label (default is 7 for PA)
    """
    print("\n" + "="*50)
    print(f"CHD68 Dataset Statistics (Target Label: {target_label})")
    print("="*50)
    
    print(f"\nOverall Dataset Statistics:")
    print(f"- Total number of cases: {stats['total_cases']}")
    print(f"- Total number of slices: {stats['total_slices']}")
    print(f"- Unique labels found: {stats['labels_found']}")
    
    print(f"\nTarget Label ({target_label}) Statistics:")
    print(f"- Number of cases with target label: {stats['cases_with_target_label']} ({stats['percent_cases_with_target_label']:.2f}%)")
    print(f"- Number of slices with target label: {stats['slices_with_target_label']} ({stats['percent_slices_with_target_label']:.2f}%)")
    
    if target_label in stats["label_volumes"]:
        print(f"- Total volume of target label: {stats['label_volumes'][target_label]:.2f} mm³")
    
    print(f"\nLabel Distribution (by voxel count):")
    for label, percentage in sorted(stats["label_distribution"].items()):
        volume = stats["label_volumes"].get(label, 0)
        print(f"- Label {label}: {percentage:.2f}% (Volume: {volume:.2f} mm³)")
    
    # Optional: print detailed per-case stats
    print("\nPer-case statistics available in the returned data structure.")

def visualize_dataset_statistics(stats, target_label=7):
    """
    Create visualizations of the dataset statistics.
    
    Parameters:
    -----------
    stats : dict
        Dictionary containing the statistics
    target_label : int
        The target label (default is 7 for PA)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set up the plotting style
        sns.set(style="whitegrid")
        plt.figure(figsize=(15, 10))
        
        # 1. Label distribution pie chart
        plt.subplot(2, 2, 1)
        labels = list(stats["label_distribution"].keys())
        sizes = list(stats["label_distribution"].values())
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Label Distribution (by voxel count)')
        
        # 2. Target label presence across cases
        plt.subplot(2, 2, 2)
        target_cases = stats["cases_with_target_label"]
        other_cases = stats["total_cases"] - target_cases
        plt.bar(['With PA', 'Without PA'], [target_cases, other_cases])
        plt.title(f'Cases With/Without Label {target_label}')
        plt.ylabel('Number of Cases')
        
        # 3. Volume distribution
        plt.subplot(2, 2, 3)
        volumes = [stats["label_volumes"].get(label, 0) for label in sorted(stats["label_volumes"].keys())]
        plt.bar([str(label) for label in sorted(stats["label_volumes"].keys())], volumes)
        plt.title('Volume Distribution by Label')
        plt.xlabel('Label')
        plt.ylabel('Volume (mm³)')
        
        # 4. Per-case target label volume (top 10 cases)
        plt.subplot(2, 2, 4)
        df_cases = pd.DataFrame(stats["per_case_stats"])
        if not df_cases.empty and "target_label_volume" in df_cases.columns:
            top_cases = df_cases.sort_values("target_label_volume", ascending=False).head(10)
            plt.bar(top_cases["case_id"], top_cases["target_label_volume"])
            plt.title(f'Top 10 Cases by Label {target_label} Volume')
            plt.xlabel('Case ID')
            plt.ylabel('Volume (mm³)')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('chd68_dataset_statistics.png')
        plt.close()
        
        print("\nDataset statistics visualization saved to chd68_dataset_statistics.png")
        
    except ImportError:
        print("\nVisualization skipped. Please install matplotlib and seaborn: pip install matplotlib seaborn")

def main():
    data_path = "DATA/CHD68_segmentation_dataset/CHD68_segmentation_dataset_miccai19"
    target_label = 7  # PA (Pulmonary Artery)
    
    print(f"Analyzing CHD68 dataset at: {data_path}")
    print(f"Target label: {target_label}")
    
    try:
        stats = analyze_chd68_dataset(data_path, target_label)
        print_statistics(stats, target_label)
        visualize_dataset_statistics(stats, target_label)
        
        # Save statistics to a CSV file
        df_cases = pd.DataFrame(stats["per_case_stats"])
        df_cases.to_csv("chd68_case_statistics.csv", index=False)
        
        print("\nDetailed case statistics saved to chd68_case_statistics.csv")
        
        return stats
    
    except Exception as e:
        print(f"Error analyzing dataset: {e}")

if __name__ == "__main__":
    main()