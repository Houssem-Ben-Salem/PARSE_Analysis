import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
import hashlib
import pandas as pd

def compute_file_hash(filepath):
    """
    Compute a hash of the file contents to identify identical files.
    """
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def compare_nifti_data(file1, file2):
    """
    Compare the actual data in two NIFTI files.
    """
    try:
        nii1 = nib.load(file1)
        nii2 = nib.load(file2)
        
        # Compare shapes first (quick check)
        if nii1.shape != nii2.shape:
            return False, None
        
        # For segmentation masks, convert to integers to avoid floating point issues
        data1 = np.round(nii1.get_fdata()).astype(np.int16)
        data2 = np.round(nii2.get_fdata()).astype(np.int16)
        
        # Calculate similarity - exact match first
        if np.array_equal(data1, data2):
            return True, 1.0
        
        # If not exact match, calculate similarity percentage
        total_voxels = np.prod(data1.shape)
        matching_voxels = np.sum(data1 == data2)
        similarity = matching_voxels / total_voxels
        
        # Return True if similarity is above threshold (e.g. 99%)
        is_duplicate = similarity > 0.99
        return is_duplicate, similarity
        
    except Exception as e:
        print(f"Error comparing {file1} and {file2}: {e}")
        return False, None

def create_dataset_index(dataset_path, pattern="*label.nii.gz"):
    """
    Create a searchable index of all files in a dataset with their hashes.
    """
    print(f"Creating index for {dataset_path}...")
    files = sorted([f for f in glob.glob(os.path.join(dataset_path, pattern)) 
               if not os.path.basename(f).startswith('._')])
    
    index = {}
    for file in tqdm(files, desc="Computing hashes"):
        file_hash = compute_file_hash(file)
        case_id = os.path.basename(file).split('_')[1]
        index[file] = {
            'hash': file_hash,
            'case_id': case_id
        }
    
    return index

def find_all_duplicates(dataset1_path, dataset2_path):
    """
    Find all duplicate masks in dataset2 that match any mask in dataset1.
    """
    print(f"Checking all files in:")
    print(f"Dataset 1: {dataset1_path}")
    print(f"Dataset 2: {dataset2_path}")
    
    # Create indices for both datasets
    dataset1_index = create_dataset_index(dataset1_path)
    dataset2_index = create_dataset_index(dataset2_path)
    
    print(f"Found {len(dataset1_index)} label files in Dataset 1")
    print(f"Found {len(dataset2_index)} label files in Dataset 2")
    
    # Create reverse lookup by hash for dataset1
    dataset1_by_hash = {}
    for file, info in dataset1_index.items():
        dataset1_by_hash[info['hash']] = file
    
    # Check each file in dataset2 against dataset1
    duplicates = []
    near_duplicates = []
    
    print("\nChecking each file in Dataset 2 against Dataset 1...")
    for file2, info2 in tqdm(dataset2_index.items(), desc="Checking files"):
        # Quick check by hash first
        if info2['hash'] in dataset1_by_hash:
            file1 = dataset1_by_hash[info2['hash']]
            # Confirm with direct comparison
            is_duplicate, similarity = compare_nifti_data(file1, file2)
            if is_duplicate:
                duplicates.append({
                    'dataset1_file': file1,
                    'dataset1_case_id': dataset1_index[file1]['case_id'],
                    'dataset2_file': file2,
                    'dataset2_case_id': info2['case_id'],
                    'similarity': similarity,
                    'matched_by': 'hash'
                })
                continue
        
        # If no hash match, do a more comprehensive check (this is slower)
        # We'll only do this for files that didn't match by hash
        for file1, info1 in dataset1_index.items():
            # Skip if we already checked this by hash
            if info1['hash'] == info2['hash']:
                continue
                
            # Compare data directly
            is_duplicate, similarity = compare_nifti_data(file1, file2)
            if is_duplicate:
                duplicates.append({
                    'dataset1_file': file1,
                    'dataset1_case_id': info1['case_id'],
                    'dataset2_file': file2,
                    'dataset2_case_id': info2['case_id'],
                    'similarity': similarity,
                    'matched_by': 'content'
                })
                break
            elif similarity is not None and similarity > 0.95:
                near_duplicates.append({
                    'dataset1_file': file1,
                    'dataset1_case_id': info1['case_id'],
                    'dataset2_file': file2,
                    'dataset2_case_id': info2['case_id'],
                    'similarity': similarity
                })
    
    # Calculate statistics
    duplicate_cases = set([d['dataset2_case_id'] for d in duplicates])
    percent_duplicates = (len(duplicate_cases) / len(dataset2_index)) * 100 if dataset2_index else 0
    
    return {
        'dataset1_count': len(dataset1_index),
        'dataset2_count': len(dataset2_index),
        'duplicates': duplicates,
        'duplicates_count': len(duplicates),
        'duplicate_cases': sorted(list(duplicate_cases)),
        'duplicate_cases_count': len(duplicate_cases),
        'percent_duplicates': percent_duplicates,
        'near_duplicates': near_duplicates,
        'near_duplicates_count': len(near_duplicates)
    }

def generate_duplicate_report(results, output_path=None):
    """
    Generate a detailed report of the duplication analysis.
    """
    report = ["# Dataset Duplication Analysis Report", ""]
    
    report.append("## Summary")
    report.append(f"- Dataset 1 contains {results['dataset1_count']} label files")
    report.append(f"- Dataset 2 contains {results['dataset2_count']} label files")
    report.append(f"- Found {results['duplicates_count']} duplicated files in Dataset 2")
    report.append(f"- Affecting {results['duplicate_cases_count']} cases ({results['percent_duplicates']:.2f}% of Dataset 2)")
    report.append(f"- Found {results['near_duplicates_count']} near-duplicate files (>95% similar)")
    report.append("")
    
    if results['duplicates_count'] > 0:
        report.append("## Exact Duplicates")
        report.append("| Dataset 2 Case | Dataset 1 Case | Similarity | Matched By |")
        report.append("|---------------|---------------|------------|------------|")
        for dup in results['duplicates']:
            report.append(f"| {dup['dataset2_case_id']} | {dup['dataset1_case_id']} | {dup['similarity']:.4f} | {dup['matched_by']} |")
        report.append("")
    
    if results['near_duplicates_count'] > 0:
        report.append("## Near Duplicates (>95% Similar)")
        report.append("| Dataset 2 Case | Dataset 1 Case | Similarity |")
        report.append("|---------------|---------------|------------|")
        for dup in sorted(results['near_duplicates'], key=lambda x: x['similarity'], reverse=True):
            report.append(f"| {dup['dataset2_case_id']} | {dup['dataset1_case_id']} | {dup['similarity']:.4f} |")
        report.append("")
    
    report.append("## List of Duplicate Cases in Dataset 2")
    report.append("These cases from Dataset 2 should be excluded when merging:")
    report.append("```")
    report.append(", ".join(results['duplicate_cases']))
    report.append("```")
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(report_text)
        print(f"Detailed report saved to {output_path}")
    
    return report_text

def create_merged_dataset(dataset1_path, dataset2_path, output_path, duplicate_results):
    """
    Create a merged dataset without duplicates.
    """
    import shutil
    
    print(f"\nCreating merged dataset at: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # Copy all files from dataset 1
    dataset1_files = glob.glob(os.path.join(dataset1_path, "*.nii.gz"))
    print(f"Copying {len(dataset1_files)} files from Dataset 1...")
    for file in tqdm(dataset1_files):
        shutil.copy2(file, output_path)
    
    # Get duplicate case IDs to exclude
    duplicate_case_ids = set(duplicate_results['duplicate_cases'])
    
    # Copy non-duplicate files from dataset 2
    dataset2_files = glob.glob(os.path.join(dataset2_path, "*.nii.gz"))
    files_to_copy = []
    
    for file in dataset2_files:
        case_id = os.path.basename(file).split('_')[1]
        if case_id not in duplicate_case_ids:
            files_to_copy.append(file)
    
    print(f"Copying {len(files_to_copy)} non-duplicate files from Dataset 2...")
    for file in tqdm(files_to_copy):
        shutil.copy2(file, output_path)
    
    # Calculate unique cases
    unique_cases1 = len(set([os.path.basename(f).split('_')[1] for f in dataset1_files if 'label' in f]))
    unique_cases2 = len(set([os.path.basename(f).split('_')[1] for f in files_to_copy if 'label' in f]))
    
    print(f"\nMerged dataset created successfully!")
    print(f"- Unique cases from Dataset 1: {unique_cases1}")
    print(f"- Unique cases from Dataset 2: {unique_cases2}")
    print(f"- Total unique cases: {unique_cases1 + unique_cases2}")
    
    # Create a README file with information about the merged dataset
    with open(os.path.join(output_path, "README.txt"), "w") as f:
        f.write("# Merged PA-Only Dataset\n\n")
        f.write(f"This dataset is created by merging unique cases from two datasets:\n")
        f.write(f"- Dataset 1: {dataset1_path}\n")
        f.write(f"- Dataset 2: {dataset2_path}\n\n")
        f.write(f"Total unique cases: {unique_cases1 + unique_cases2}\n")
        f.write(f"- From Dataset 1: {unique_cases1}\n")
        f.write(f"- From Dataset 2: {unique_cases2}\n\n")
        f.write(f"Duplicate cases excluded: {len(duplicate_case_ids)}\n")
        f.write(f"Created on: {import_date()}\n")

def import_date():
    """Get the current date for documentation purposes."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Check for duplicates between two datasets by content")
    parser.add_argument("--dataset1", type=str, default="DATA/ImageCHD_PA_only_dataset",
                        help="Path to the first dataset")
    parser.add_argument("--dataset2", type=str, default="DATA/CHD68_PA_only_dataset",
                        help="Path to the second dataset")
    parser.add_argument("--output", type=str, default="content_duplication_report.md",
                        help="Path to save the duplication report")
    parser.add_argument("--merge", type=str, default=None,
                        help="Path where to save the merged dataset (if specified)")
    
    args = parser.parse_args()
    
    # Find duplicates
    duplicate_results = find_all_duplicates(args.dataset1, args.dataset2)
    
    # Generate report
    report = generate_duplicate_report(duplicate_results, args.output)
    print("\n" + "="*80)
    print(report)
    print("="*80)
    
    # Create merged dataset if requested
    if args.merge:
        create_merged_dataset(args.dataset1, args.dataset2, args.merge, duplicate_results)

if __name__ == "__main__":
    main()