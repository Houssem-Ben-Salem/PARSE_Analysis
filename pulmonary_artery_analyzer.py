import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from pathlib import Path
import argparse


class PulmonaryArteryDatasetAnalyzer:
    def __init__(self, data_dir):
        """
        Initialize the analyzer with the path to the dataset.
        
        Parameters:
        -----------
        data_dir : str
            Path to the Parse2022 dataset directory
        """
        self.data_dir = Path(data_dir)
        self.patient_ids = self._get_patient_ids()
        self.stats = {}
        
    def _get_patient_ids(self):
        """Get all patient IDs from the dataset directory."""
        return [f for f in os.listdir(self.data_dir) if f.startswith('PA')]
        
    def analyze_dataset(self):
        """Analyze the entire dataset and collect statistics."""
        print(f"Found {len(self.patient_ids)} patients in dataset")
        
        # Initialize data collection dictionaries
        image_shapes = []
        image_spacings = []
        label_volumes = []
        label_percentages = []
        intensity_stats = []
        
        # Analyze each patient
        for patient_id in tqdm(self.patient_ids, desc="Analyzing patients"):
            # Load image and label
            img_path = self.data_dir / patient_id / 'image' / f"{patient_id}.nii.gz"
            label_path = self.data_dir / patient_id / 'label' / f"{patient_id}.nii.gz"
            
            if not img_path.exists() or not label_path.exists():
                print(f"Warning: Missing files for patient {patient_id}")
                continue
                
            # Load image data
            nii_img = nib.load(str(img_path))
            img_data = nii_img.get_fdata()
            
            # Load label data
            nii_label = nib.load(str(label_path))
            label_data = nii_label.get_fdata()
            
            # Get image shape and spacing
            shape = img_data.shape
            spacing = nii_img.header.get_zooms()
            
            # Calculate volume of segmentation mask
            label_volume = np.sum(label_data > 0)
            # Calculate percentage of voxels that are labeled
            label_percentage = (label_volume / label_data.size) * 100
            
            # Calculate intensity statistics within the segmented regions
            if label_volume > 0:
                vessel_intensities = img_data[label_data > 0]
                background_intensities = img_data[label_data == 0]
                
                vessel_stats = {
                    'patient_id': patient_id,
                    'vessel_min': np.min(vessel_intensities),
                    'vessel_max': np.max(vessel_intensities),
                    'vessel_mean': np.mean(vessel_intensities),
                    'vessel_std': np.std(vessel_intensities),
                    'bg_min': np.min(background_intensities),
                    'bg_max': np.max(background_intensities),
                    'bg_mean': np.mean(background_intensities),
                    'bg_std': np.std(background_intensities),
                    'contrast_ratio': np.mean(vessel_intensities) / np.mean(background_intensities)
                }
                intensity_stats.append(vessel_stats)
            
            # Store the data
            image_shapes.append((patient_id, shape))
            image_spacings.append((patient_id, spacing))
            label_volumes.append((patient_id, label_volume))
            label_percentages.append((patient_id, label_percentage))
            
        # Convert to DataFrame for easier analysis
        self.stats['shapes'] = pd.DataFrame(image_shapes, columns=['patient_id', 'shape'])
        self.stats['spacings'] = pd.DataFrame(image_spacings, columns=['patient_id', 'spacing'])
        self.stats['volumes'] = pd.DataFrame(label_volumes, columns=['patient_id', 'volume'])
        self.stats['percentages'] = pd.DataFrame(label_percentages, columns=['patient_id', 'percentage'])
        self.stats['intensities'] = pd.DataFrame(intensity_stats)
        
        return self.stats
    
    def visualize_sample(self, patient_id, output_dir=None):
        """
        Visualize a sample patient with image and segmentation overlay.
        
        Parameters:
        -----------
        patient_id : str
            Patient ID to visualize
        output_dir : str, optional
            Directory to save visualizations
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        img_path = self.data_dir / patient_id / 'image' / f"{patient_id}.nii.gz"
        label_path = self.data_dir / patient_id / 'label' / f"{patient_id}.nii.gz"
        
        # Load image data
        nii_img = nib.load(str(img_path))
        img_data = nii_img.get_fdata()
        
        # Load label data
        nii_label = nib.load(str(label_path))
        label_data = nii_label.get_fdata()
        
        # Get the middle slice for each dimension
        slices = [img_data.shape[i] // 2 for i in range(3)]
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        
        for i, (axis, slice_idx) in enumerate(zip(['Sagittal', 'Coronal', 'Axial'], slices)):
            # Extract the slice
            if i == 0:  # Sagittal
                img_slice = img_data[slice_idx, :, :]
                label_slice = label_data[slice_idx, :, :]
            elif i == 1:  # Coronal
                img_slice = img_data[:, slice_idx, :]
                label_slice = label_data[:, slice_idx, :]
            else:  # Axial
                img_slice = img_data[:, :, slice_idx]
                label_slice = label_data[:, :, slice_idx]
                
            # Plot original image
            axes[i, 0].imshow(img_slice.T, cmap='gray')
            axes[i, 0].set_title(f"{axis} Plane - Image")
            axes[i, 0].axis('off')
            
            # Plot image with segmentation overlay
            axes[i, 1].imshow(img_slice.T, cmap='gray')
            mask = label_slice.T > 0
            overlay = np.zeros((*img_slice.T.shape, 4))
            overlay[mask, :] = [1, 0, 0, 0.5]  # Red with 50% opacity
            axes[i, 1].imshow(overlay)
            axes[i, 1].set_title(f"{axis} Plane - Segmentation Overlay")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{patient_id}_visualization.png"))
            plt.close()
        else:
            plt.show()
            
    def generate_report(self, output_dir=None):
        """
        Generate a comprehensive report of the dataset.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save the report
        """
        if not self.stats:
            self.analyze_dataset()
            
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Create a report DataFrame
        report = {}
        
        # Image shapes
        shapes = self.stats['shapes']['shape'].tolist()
        unique_shapes = set(str(s) for s in shapes)
        report['unique_shapes'] = len(unique_shapes)
        report['shape_count'] = {str(s): shapes.count(s) for s in unique_shapes}
        
        # Image spacings
        spacings = self.stats['spacings']['spacing'].tolist()
        unique_spacings = set(str(s) for s in spacings)
        report['unique_spacings'] = len(unique_spacings)
        report['spacing_count'] = {str(s): spacings.count(s) for s in unique_spacings}
        
        # Label volumes
        volumes = self.stats['volumes']['volume'].tolist()
        report['min_volume'] = np.min(volumes)
        report['max_volume'] = np.max(volumes)
        report['mean_volume'] = np.mean(volumes)
        report['std_volume'] = np.std(volumes)
        
        # Label percentages
        percentages = self.stats['percentages']['percentage'].tolist()
        report['min_percentage'] = np.min(percentages)
        report['max_percentage'] = np.max(percentages)
        report['mean_percentage'] = np.mean(percentages)
        report['std_percentage'] = np.std(percentages)
        
        # Intensity statistics
        intensity_df = self.stats['intensities']
        report['mean_contrast_ratio'] = intensity_df['contrast_ratio'].mean()
        report['min_contrast_ratio'] = intensity_df['contrast_ratio'].min()
        report['max_contrast_ratio'] = intensity_df['contrast_ratio'].max()
        
        # Generate some plots
        plt.figure(figsize=(10, 6))
        sns.histplot(percentages, kde=True)
        plt.title('Distribution of Segmentation Percentages')
        plt.xlabel('Percentage of Voxels Segmented (%)')
        plt.ylabel('Count')
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'segmentation_percentage_dist.png'))
            plt.close()
            
        plt.figure(figsize=(10, 6))
        sns.histplot(intensity_df['contrast_ratio'], kde=True)
        plt.title('Distribution of Contrast Ratios')
        plt.xlabel('Contrast Ratio (Vessel/Background)')
        plt.ylabel('Count')
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'contrast_ratio_dist.png'))
            plt.close()
            
        # Create a comprehensive text report
        report_text = "Pulmonary Artery Segmentation Dataset Analysis\n"
        report_text += "="*50 + "\n\n"
        
        report_text += f"Total number of patients: {len(self.patient_ids)}\n\n"
        
        report_text += "Image Dimensions:\n"
        report_text += f"  - Number of unique shapes: {report['unique_shapes']}\n"
        for shape, count in report['shape_count'].items():
            report_text += f"    - {shape}: {count} images\n"
        report_text += "\n"
        
        report_text += "Image Spacings:\n"
        report_text += f"  - Number of unique spacings: {report['unique_spacings']}\n"
        for spacing, count in report['spacing_count'].items():
            report_text += f"    - {spacing}: {count} images\n"
        report_text += "\n"
        
        report_text += "Segmentation Statistics:\n"
        report_text += f"  - Min volume: {report['min_volume']:.2f} voxels\n"
        report_text += f"  - Max volume: {report['max_volume']:.2f} voxels\n"
        report_text += f"  - Mean volume: {report['mean_volume']:.2f} voxels\n"
        report_text += f"  - Std volume: {report['std_volume']:.2f} voxels\n\n"
        
        report_text += f"  - Min percentage: {report['min_percentage']:.4f}%\n"
        report_text += f"  - Max percentage: {report['max_percentage']:.4f}%\n"
        report_text += f"  - Mean percentage: {report['mean_percentage']:.4f}%\n"
        report_text += f"  - Std percentage: {report['std_percentage']:.4f}%\n\n"
        
        report_text += "Intensity Statistics:\n"
        report_text += f"  - Mean contrast ratio: {report['mean_contrast_ratio']:.4f}\n"
        report_text += f"  - Min contrast ratio: {report['min_contrast_ratio']:.4f}\n"
        report_text += f"  - Max contrast ratio: {report['max_contrast_ratio']:.4f}\n\n"
        
        report_text += "Model Selection Recommendations:\n"
        
        # Add recommendations based on findings
        if report['mean_percentage'] < 1.0:
            report_text += "  - The dataset has a significant class imbalance (small percentage of voxels labeled as vessels).\n"
            report_text += "  - Consider using models that handle class imbalance well, such as U-Net with weighted loss functions.\n"
            report_text += "  - Dice loss or Focal loss might be more appropriate than binary cross-entropy.\n"
        
        if report['unique_shapes'] > 1:
            report_text += "  - Dataset contains images of different dimensions.\n"
            report_text += "  - Consider using patch-based approaches or resizing to a common dimension.\n"
            report_text += "  - 3D U-Net with patch sampling might be appropriate.\n"
        
        if report['unique_spacings'] > 1:
            report_text += "  - Dataset contains images with different spacings.\n"
            report_text += "  - Consider resampling all images to a common spacing before training.\n"
        
        if report['mean_contrast_ratio'] < 1.5:
            report_text += "  - Low contrast between vessels and background.\n"
            report_text += "  - Consider preprocessing steps like histogram equalization or contrast enhancement.\n"
            report_text += "  - Models with attention mechanisms might perform better.\n"
        
        # General recommendations
        report_text += "\nRecommended Models:\n"
        report_text += "  1. 3D U-Net: Good baseline for volumetric medical image segmentation\n"
        report_text += "  2. V-Net: Specifically designed for volumetric medical image segmentation with dice loss\n"
        report_text += "  3. Attention U-Net: Helpful if vessels are small and contrast is variable\n"
        report_text += "  4. nnU-Net: State-of-the-art self-configuring method for medical image segmentation\n"
        report_text += "  5. DeepMedic: Multi-scale 3D CNN architecture good for capturing details at different scales\n\n"
        
        report_text += "Preprocessing Recommendations:\n"
        report_text += "  1. Intensity normalization (e.g., z-score normalization within foreground)\n"
        report_text += "  2. Resampling to isotropic resolution if voxel spacings are anisotropic\n"
        report_text += "  3. Data augmentation (rotation, scaling, elastic deformation, etc.)\n"
        report_text += "  4. Connected component analysis to ensure only pulmonary arteries are labeled\n\n"
        
        report_text += "Evaluation Metrics Recommendations:\n"
        report_text += "  1. Dice coefficient: Standard for medical image segmentation\n"
        report_text += "  2. Hausdorff distance: Important for boundary accuracy\n"
        report_text += "  3. Average surface distance: More robust to outliers than Hausdorff\n"
        report_text += "  4. Precision and recall: Important due to class imbalance\n"
        
        if output_dir:
            with open(os.path.join(output_dir, 'dataset_report.txt'), 'w') as f:
                f.write(report_text)
            
            # Also save as CSV
            pd.DataFrame({
                'Metric': list(report.keys()),
                'Value': list(report.values())
            }).to_csv(os.path.join(output_dir, 'dataset_metrics.csv'), index=False)
            
        return report_text


def main():
    parser = argparse.ArgumentParser(description='Analyze pulmonary artery segmentation dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Parse2022 dataset directory')
    parser.add_argument('--output_dir', type=str, default='dataset_analysis', help='Path to save analysis results')
    parser.add_argument('--visualize', type=str, default=None, help='Patient ID to visualize (optional)')
    args = parser.parse_args()
    
    analyzer = PulmonaryArteryDatasetAnalyzer(args.data_dir)
    
    # Generate and save the report
    print("Analyzing dataset...")
    report = analyzer.generate_report(args.output_dir)
    print(f"Report saved to {args.output_dir}/dataset_report.txt")
    
    # Visualize a sample if requested
    if args.visualize:
        print(f"Visualizing patient {args.visualize}...")
        analyzer.visualize_sample(args.visualize, args.output_dir)
        print(f"Visualization saved to {args.output_dir}/{args.visualize}_visualization.png")
    else:
        # Visualize first patient as example
        if len(analyzer.patient_ids) > 0:
            sample_id = analyzer.patient_ids[0]
            print(f"Visualizing sample patient {sample_id}...")
            analyzer.visualize_sample(sample_id, args.output_dir)
            print(f"Visualization saved to {args.output_dir}/{sample_id}_visualization.png")


if __name__ == "__main__":
    main()