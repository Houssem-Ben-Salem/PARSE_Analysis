import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from skimage import measure
import argparse
from pathlib import Path

def analyze_pa_dataset(dataset_path, pa_label=7, output_dir=None):
    """
    Perform comprehensive statistical analysis on a PA-only dataset.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset containing PA segmentations
    pa_label : int
        Label value for pulmonary artery (default: 7)
    output_dir : str
        Directory to save output files (defaults to dataset_path/statistics)
    
    Returns:
    --------
    dict
        Dictionary containing all statistical information
    """
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(dataset_path, "statistics")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all label files
    label_files = sorted([f for f in glob.glob(os.path.join(dataset_path, "*label.nii.gz")) 
                      if not os.path.basename(f).startswith('._')])
    
    if not label_files:
        raise FileNotFoundError(f"No label files found in {dataset_path}")
    
    print(f"Found {len(label_files)} label files in the dataset.")
    
    # Initialize data structures for statistics
    cases_data = []
    slice_data = []
    total_voxels = 0
    total_pa_voxels = 0
    total_slices = 0
    slices_with_pa = 0
    
    # Process each case
    for label_file in tqdm(label_files, desc="Analyzing cases"):
        try:
            # Get case ID
            case_id = os.path.basename(label_file).split('_')[1]
            
            # Load label file
            nii_label = nib.load(label_file)
            label_data = np.round(nii_label.get_fdata()).astype(np.int16)
            
            # Get corresponding image file
            image_file = os.path.join(dataset_path, f"ct_{case_id}_image.nii.gz")
            image_data = None
            if os.path.exists(image_file):
                nii_image = nib.load(image_file)
                image_data = nii_image.get_fdata()
            
            # Get voxel dimensions for volume calculation
            voxel_dims = nii_label.header.get_zooms()
            voxel_volume = np.prod(voxel_dims)
            
            # Count PA voxels
            pa_mask = (label_data == pa_label)
            case_pa_voxels = np.sum(pa_mask)
            case_total_voxels = np.prod(label_data.shape)
            
            # Calculate PA volume in mm³
            pa_volume = case_pa_voxels * voxel_volume
            
            # Count slices with PA
            num_slices = label_data.shape[2]  # Assuming axial orientation
            case_slices_with_pa = 0
            slices_info = []
            
            for slice_idx in range(num_slices):
                slice_data_mask = label_data[:,:,slice_idx]
                has_pa = np.any(slice_data_mask == pa_label)
                
                if has_pa:
                    case_slices_with_pa += 1
                    slice_pa_voxels = np.sum(slice_data_mask == pa_label)
                    
                    # Calculate center of mass for this slice's PA
                    if slice_pa_voxels > 0:
                        slice_pa_mask = (slice_data_mask == pa_label)
                        try:
                            com_y, com_x = ndimage.center_of_mass(slice_pa_mask)
                            pa_area_mm2 = slice_pa_voxels * (voxel_dims[0] * voxel_dims[1])
                            
                            # Get image statistics if available
                            intensity_stats = None
                            if image_data is not None:
                                slice_img = image_data[:,:,slice_idx]
                                pa_intensities = slice_img[slice_pa_mask]
                                intensity_stats = {
                                    'mean': np.mean(pa_intensities),
                                    'std': np.std(pa_intensities),
                                    'min': np.min(pa_intensities),
                                    'max': np.max(pa_intensities)
                                }
                            
                            slices_info.append({
                                'case_id': case_id,
                                'slice_idx': slice_idx,
                                'slice_position': slice_idx / num_slices,  # Normalized position
                                'pa_voxels': slice_pa_voxels,
                                'pa_area_mm2': pa_area_mm2,
                                'center_x': com_x,
                                'center_y': com_y,
                                'intensity_stats': intensity_stats
                            })
                        except Exception as e:
                            print(f"Error processing center of mass for case {case_id}, slice {slice_idx}: {e}")
            
            # Calculate 3D shape metrics if there are PA voxels
            shape_metrics = {}
            if case_pa_voxels > 0:
                try:
                    # Get connected components
                    labeled_array, num_features = ndimage.label(pa_mask)
                    
                    # Calculate properties for each component
                    regions = measure.regionprops(labeled_array)
                    
                    # Take the largest component
                    if regions:
                        largest_region = max(regions, key=lambda r: r.area)
                        
                        # Calculate 3D shape metrics
                        shape_metrics = {
                            'volume': largest_region.area * voxel_volume,
                            'num_components': num_features,
                        }
                        
                        # Calculate elongation if possible
                        if hasattr(largest_region, 'axis_major_length') and hasattr(largest_region, 'axis_minor_length'):
                            if largest_region.axis_minor_length > 0:
                                shape_metrics['elongation'] = largest_region.axis_major_length / largest_region.axis_minor_length
                        
                        # Calculate 3D surface area (approximation)
                        # Use a structural element to detect boundary voxels
                        struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
                        eroded = ndimage.binary_erosion(pa_mask, struct)
                        boundary = pa_mask & ~eroded
                        surface_voxels = np.sum(boundary)
                        
                        # Estimate surface area in mm^2
                        # We use an approximation based on the voxel dimensions
                        surface_area = surface_voxels * (voxel_dims[0] * voxel_dims[1])
                        shape_metrics['surface_area_mm2'] = surface_area
                        
                        # Calculate 3D sphericity using volume and surface area
                        # Sphericity = (36π * V²)^(1/3) / S
                        if surface_area > 0:
                            volume = largest_region.area * voxel_volume
                            sphericity = ((36 * np.pi * volume**2)**(1/3)) / surface_area
                            shape_metrics['sphericity'] = sphericity
                except Exception as e:
                    print(f"Error calculating shape metrics for case {case_id}: {e}")
            
            # Add case statistics
            case_data = {
                'case_id': case_id,
                'total_slices': num_slices,
                'slices_with_pa': case_slices_with_pa,
                'percent_slices_with_pa': (case_slices_with_pa / num_slices) * 100 if num_slices > 0 else 0,
                'pa_voxels': case_pa_voxels,
                'total_voxels': case_total_voxels,
                'pa_volume_mm3': pa_volume,
                'pa_percentage': (case_pa_voxels / case_total_voxels) * 100 if case_total_voxels > 0 else 0,
                'voxel_dims': voxel_dims,
                'shape_metrics': shape_metrics
            }
            
            cases_data.append(case_data)
            slice_data.extend(slices_info)
            
            # Update global counters
            total_voxels += case_total_voxels
            total_pa_voxels += case_pa_voxels
            total_slices += num_slices
            slices_with_pa += case_slices_with_pa
            
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
    
    # Calculate global statistics
    global_stats = {
        'total_cases': len(label_files),
        'cases_with_pa': sum(1 for case in cases_data if case['pa_voxels'] > 0),
        'total_slices': total_slices,
        'slices_with_pa': slices_with_pa,
        'percent_slices_with_pa': (slices_with_pa / total_slices) * 100 if total_slices > 0 else 0,
        'total_voxels': total_voxels,
        'total_pa_voxels': total_pa_voxels,
        'pa_percentage': (total_pa_voxels / total_voxels) * 100 if total_voxels > 0 else 0,
        'total_pa_volume_mm3': sum(case['pa_volume_mm3'] for case in cases_data)
    }
    
    # Calculate distribution statistics
    if cases_data:
        pa_volumes = [case['pa_volume_mm3'] for case in cases_data]
        pa_percentages = [case['pa_percentage'] for case in cases_data]
        slices_counts = [case['total_slices'] for case in cases_data]
        
        distribution_stats = {
            'pa_volume_mean': np.mean(pa_volumes),
            'pa_volume_median': np.median(pa_volumes),
            'pa_volume_std': np.std(pa_volumes),
            'pa_volume_min': np.min(pa_volumes),
            'pa_volume_max': np.max(pa_volumes),
            'pa_volume_quartiles': np.percentile(pa_volumes, [25, 50, 75]),
            'pa_percentage_mean': np.mean(pa_percentages),
            'pa_percentage_median': np.median(pa_percentages),
            'slices_per_case_mean': np.mean(slices_counts),
            'slices_per_case_median': np.median(slices_counts),
            'slices_per_case_std': np.std(slices_counts)
        }
        
        # Categorize cases by PA volume
        volume_quartiles = np.percentile(pa_volumes, [25, 50, 75])
        case_categories = {
            'small_pa': sum(1 for vol in pa_volumes if vol <= volume_quartiles[0]),
            'medium_pa': sum(1 for vol in pa_volumes if volume_quartiles[0] < vol <= volume_quartiles[2]),
            'large_pa': sum(1 for vol in pa_volumes if vol > volume_quartiles[2])
        }
        distribution_stats.update(case_categories)
    else:
        distribution_stats = {}
    
    # Create summary statistics
    stats = {
        'global': global_stats,
        'distribution': distribution_stats,
        'cases': cases_data,
        'slices': slice_data
    }
    
    # Generate visualizations
    generate_visualizations(stats, output_dir)
    
    # Save statistics to files
    save_statistics(stats, output_dir)
    
    return stats

def generate_visualizations(stats, output_dir):
    """Generate visualizations from the statistics."""
    print("\nGenerating visualizations...")
    
    # Set up plot style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Create a dataframe for easier plotting
    cases_df = pd.DataFrame([{k: v for k, v in case.items() if not isinstance(v, dict)} 
                              for case in stats['cases']])
    
    # 1. PA Volume Distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.histplot(cases_df['pa_volume_mm3'], kde=True)
    plt.title('PA Volume Distribution (mm³)')
    plt.xlabel('Volume (mm³)')
    plt.ylabel('Count')
    
    # 2. Percentage of Slices with PA per Case
    plt.subplot(2, 2, 2)
    sns.histplot(cases_df['percent_slices_with_pa'], kde=True)
    plt.title('Percentage of Slices with PA per Case')
    plt.xlabel('Percentage of Slices (%)')
    plt.ylabel('Count')
    
    # 3. PA Volume by Case (Top 20)
    plt.subplot(2, 2, 3)
    if len(cases_df) > 0:
        top_cases = cases_df.sort_values('pa_volume_mm3', ascending=False).head(20)
        sns.barplot(x='case_id', y='pa_volume_mm3', data=top_cases)
        plt.title('Top 20 Cases by PA Volume')
        plt.xlabel('Case ID')
        plt.ylabel('PA Volume (mm³)')
        plt.xticks(rotation=90)
    
    # 4. Slices per Case Distribution
    plt.subplot(2, 2, 4)
    sns.histplot(cases_df['total_slices'], kde=True)
    plt.title('Slices per Case Distribution')
    plt.xlabel('Number of Slices')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pa_volume_statistics.png'), dpi=300)
    plt.close()
    
    # Create a second set of visualizations
    if len(stats['slices']) > 0:
        # Create slice dataframe
        slices_df = pd.DataFrame([{k: v for k, v in slice_data.items() if not isinstance(v, dict)} 
                                 for slice_data in stats['slices']])
        
        plt.figure(figsize=(12, 8))
        
        # 1. PA Area Distribution per Slice
        plt.subplot(2, 2, 1)
        sns.histplot(slices_df['pa_area_mm2'], kde=True)
        plt.title('PA Area Distribution per Slice (mm²)')
        plt.xlabel('Area (mm²)')
        plt.ylabel('Count')
        
        # 2. PA Voxels per Slice
        plt.subplot(2, 2, 2)
        sns.histplot(slices_df['pa_voxels'], kde=True)
        plt.title('PA Voxels per Slice')
        plt.xlabel('Number of PA Voxels')
        plt.ylabel('Count')
        
        # 3. PA Centroid Positions
        plt.subplot(2, 2, 3)
        scatter = plt.scatter(slices_df['center_x'], slices_df['center_y'], 
                   c=slices_df['pa_area_mm2'], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='PA Area (mm²)')
        plt.title('PA Centroid Positions Across All Slices')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        # 4. PA Area vs. Slice Position
        plt.subplot(2, 2, 4)
        sns.scatterplot(x='slice_position', y='pa_area_mm2', data=slices_df, alpha=0.3)
        plt.title('PA Area vs. Normalized Slice Position')
        plt.xlabel('Normalized Slice Position (0=top, 1=bottom)')
        plt.ylabel('PA Area (mm²)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pa_slice_statistics.png'), dpi=300)
        plt.close()
        
        # Create heatmap of PA distribution
        try:
            # Group by case and calculate PA percentage per slice position
            grouped_slices = slices_df.copy()
            grouped_slices['slice_position_bin'] = (grouped_slices['slice_position'] * 10).astype(int)
            heatmap_data = grouped_slices.groupby(['case_id', 'slice_position_bin'])['pa_area_mm2'].mean().unstack()
            
            plt.figure(figsize=(16, 10))
            sns.heatmap(heatmap_data, cmap="YlGnBu", robust=True)
            plt.title('PA Distribution Across Cases and Slice Positions')
            plt.xlabel('Normalized Slice Position (0=top, 1=bottom)')
            plt.ylabel('Case ID')
            plt.savefig(os.path.join(output_dir, 'pa_distribution_heatmap.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create PA distribution heatmap: {e}")
    
    # Create a plot showing class imbalance
    plt.figure(figsize=(10, 6))
    labels = ['Background', 'PA']
    sizes = [stats['global']['total_voxels'] - stats['global']['total_pa_voxels'], 
             stats['global']['total_pa_voxels']]
    plt.pie(sizes, labels=labels, autopct='%1.2f%%', startangle=90, colors=['#1f77b4', '#ff7f0e'])
    plt.title('Class Distribution (PA vs. Background)')
    plt.savefig(os.path.join(output_dir, 'class_imbalance.png'), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def save_statistics(stats, output_dir):
    """Save statistics to CSV and text files."""
    # Save case-level statistics
    cases_df = pd.DataFrame([{k: v for k, v in case.items() if not isinstance(v, dict)} 
                              for case in stats['cases']])
    cases_df.to_csv(os.path.join(output_dir, 'case_statistics.csv'), index=False)
    
    # Save slice-level statistics
    if stats['slices']:
        slices_df = pd.DataFrame([{k: v for k, v in slice_data.items() if not isinstance(v, dict)} 
                                  for slice_data in stats['slices']])
        slices_df.to_csv(os.path.join(output_dir, 'slice_statistics.csv'), index=False)
    
    # Create a summary text file
    with open(os.path.join(output_dir, 'summary_statistics.txt'), 'w') as f:
        f.write("# Pulmonary Artery Dataset Summary Statistics\n\n")
        
        f.write("## Global Statistics\n")
        for key, value in stats['global'].items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("\n## Distribution Statistics\n")
        for key, value in stats['distribution'].items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f}\n")
            elif isinstance(value, np.ndarray):
                f.write(f"{key}: {', '.join(f'{v:.2f}' for v in value)}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    # Create a detailed report
    with open(os.path.join(output_dir, 'detailed_report.md'), 'w') as f:
        f.write("# Pulmonary Artery Dataset Detailed Report\n\n")
        
        f.write("## Dataset Overview\n\n")
        f.write(f"- Total cases: {stats['global']['total_cases']}\n")
        f.write(f"- Cases with PA segmentations: {stats['global']['cases_with_pa']}\n")
        f.write(f"- Total slices: {stats['global']['total_slices']}\n")
        f.write(f"- Slices with PA: {stats['global']['slices_with_pa']} ({stats['global']['percent_slices_with_pa']:.2f}%)\n")
        f.write(f"- Total PA volume: {stats['global']['total_pa_volume_mm3']:.2f} mm³\n\n")
        
        f.write("## Class Balance\n\n")
        f.write(f"- Total voxels: {stats['global']['total_voxels']}\n")
        f.write(f"- PA voxels: {stats['global']['total_pa_voxels']} ({stats['global']['pa_percentage']:.4f}%)\n")
        f.write(f"- Background voxels: {stats['global']['total_voxels'] - stats['global']['total_pa_voxels']} ({100 - stats['global']['pa_percentage']:.4f}%)\n\n")
        
        f.write("## PA Volume Distribution\n\n")
        f.write(f"- Mean PA volume: {stats['distribution']['pa_volume_mean']:.2f} mm³\n")
        f.write(f"- Median PA volume: {stats['distribution']['pa_volume_median']:.2f} mm³\n")
        f.write(f"- Std. deviation: {stats['distribution']['pa_volume_std']:.2f} mm³\n")
        f.write(f"- Min PA volume: {stats['distribution']['pa_volume_min']:.2f} mm³\n")
        f.write(f"- Max PA volume: {stats['distribution']['pa_volume_max']:.2f} mm³\n")
        f.write(f"- 25th percentile: {stats['distribution']['pa_volume_quartiles'][0]:.2f} mm³\n")
        f.write(f"- 75th percentile: {stats['distribution']['pa_volume_quartiles'][2]:.2f} mm³\n\n")
        
        f.write("## Case Categories\n\n")
        f.write(f"- Small PA cases: {stats['distribution']['small_pa']} ({(stats['distribution']['small_pa']/stats['global']['total_cases'])*100:.2f}%)\n")
        f.write(f"- Medium PA cases: {stats['distribution']['medium_pa']} ({(stats['distribution']['medium_pa']/stats['global']['total_cases'])*100:.2f}%)\n")
        f.write(f"- Large PA cases: {stats['distribution']['large_pa']} ({(stats['distribution']['large_pa']/stats['global']['total_cases'])*100:.2f}%)\n\n")
        
        f.write("## Slice Distribution\n\n")
        f.write(f"- Mean slices per case: {stats['distribution']['slices_per_case_mean']:.2f}\n")
        f.write(f"- Median slices per case: {stats['distribution']['slices_per_case_median']:.2f}\n")
        f.write(f"- Std. deviation: {stats['distribution']['slices_per_case_std']:.2f}\n\n")
        
        f.write("## Segmentation Challenges\n\n")
        f.write("Based on the statistics, the following challenges are anticipated for PA segmentation:\n\n")
        
        # Calculate class imbalance ratio
        imbalance_ratio = (stats['global']['total_voxels'] - stats['global']['total_pa_voxels']) / stats['global']['total_pa_voxels']
        
        f.write(f"1. **Class Imbalance**: The background-to-PA ratio is {imbalance_ratio:.2f}:1, ")
        if imbalance_ratio > 1000:
            f.write("indicating extreme class imbalance. This will require special handling such as weighted loss functions, focal loss, or sampling techniques.\n\n")
        elif imbalance_ratio > 100:
            f.write("indicating severe class imbalance. Consider using weighted loss functions or focal loss.\n\n")
        else:
            f.write("which is typical for small anatomical structures. Standard techniques like weighted loss functions should be sufficient.\n\n")
        
        f.write(f"2. **Size Variability**: PA volume varies from {stats['distribution']['pa_volume_min']:.2f} mm³ to {stats['distribution']['pa_volume_max']:.2f} mm³, ")
        volume_coefficient_of_variation = stats['distribution']['pa_volume_std'] / stats['distribution']['pa_volume_mean']
        if volume_coefficient_of_variation > 1.0:
            f.write("with very high variability (coefficient of variation > 1.0). Data augmentation and ensuring representative cases in training/validation splits will be crucial.\n\n")
        elif volume_coefficient_of_variation > 0.5:
            f.write("with moderate variability. Ensure training data covers the full range of volumes.\n\n")
        else:
            f.write("with relatively consistent sizes. Standard training approaches should work well.\n\n")
        
        percent_slices_with_pa = stats['global']['percent_slices_with_pa']
        f.write(f"3. **Sparse Segmentation**: Only {percent_slices_with_pa:.2f}% of slices contain PA segmentations. ")
        if percent_slices_with_pa < 10:
            f.write("This extreme sparsity may require specialized handling like 3D context or curriculum learning approaches.\n\n")
        elif percent_slices_with_pa < 25:
            f.write("This sparsity suggests that 3D context or slice selection strategies may be beneficial.\n\n")
        else:
            f.write("This is manageable with standard 2D or 3D segmentation approaches.\n\n")
    
    print(f"Statistics saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze a PA-only dataset")
    parser.add_argument("--dataset", type=str, default="DATA/ImageCHD_PA_only_dataset",
                        help="Path to the PA-only dataset")
    parser.add_argument("--pa-label", type=int, default=7,
                        help="Label value for pulmonary artery (default: 7)")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save output files")
    
    args = parser.parse_args()
    
    print(f"Analyzing PA-only dataset at: {args.dataset}")
    
    try:
        stats = analyze_pa_dataset(args.dataset, args.pa_label, args.output)
        
        # Print summary statistics
        print("\n" + "="*50)
        print("PA Dataset Analysis Summary")
        print("="*50)
        print(f"Total cases: {stats['global']['total_cases']}")
        print(f"Cases with PA: {stats['global']['cases_with_pa']}")
        print(f"Total slices: {stats['global']['total_slices']}")
        print(f"Slices with PA: {stats['global']['slices_with_pa']} ({stats['global']['percent_slices_with_pa']:.2f}%)")
        print(f"Total PA volume: {stats['global']['total_pa_volume_mm3']:.2f} mm³")
        print(f"Mean PA volume per case: {stats['distribution']['pa_volume_mean']:.2f} mm³")
        print(f"PA to background ratio: 1:{(stats['global']['total_voxels'] - stats['global']['total_pa_voxels']) / stats['global']['total_pa_voxels']:.2f}")
        
        if args.output:
            output_dir = args.output
        else:
            output_dir = os.path.join(args.dataset, "statistics")
        
        print(f"\nDetailed report saved to: {os.path.join(output_dir, 'detailed_report.md')}")
        print(f"Visualizations saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")

if __name__ == "__main__":
    main()