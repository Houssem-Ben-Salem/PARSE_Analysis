# Pulmonary Artery Segmentation Analysis Tools

A comprehensive toolkit for analyzing pulmonary artery (PA) segmentation datasets, supporting dataset preprocessing, statistics generation, duplicate detection, and visualization.

## Overview

This repository contains a collection of Python scripts for working with medical imaging datasets focused on pulmonary artery segmentation. These tools enable researchers to:

- Create PA-only datasets from larger multi-label segmentation datasets
- Analyze statistical properties of PA segmentations
- Identify and handle dataset duplicates
- Generate comprehensive reports and visualizations

The toolkit was originally designed for the PARSE (Pulmonary Artery Segmentation) dataset but can be adapted for other similar medical imaging segmentation datasets.

## Scripts

### Dataset Creation and Preprocessing

- **create_imagechd_pa_dataset.py**: Extract PA segmentations from the ImageCHD dataset
- **create_pa_dataset.py**: Extract PA segmentations from the CHD68 dataset

### Analysis Tools

- **analyze_pa_dataset.py**: Comprehensive statistical analysis of PA segmentations
- **stats.py**: Basic statistical analysis of CHD68 dataset with label distribution
- **pulmonary_artery_analyzer.py**: Specialized analyzer for pulmonary artery datasets

### Dataset Validation and Quality Control

- **check_pa_dataset_duplicates.py**: Identify duplicate cases between datasets

## Requirements

```
nibabel
numpy
pandas
matplotlib
seaborn
tqdm
scikit-image
scipy
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/pa-segmentation-analysis.git
   cd pa-segmentation-analysis
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Creating a PA-only Dataset

Extract pulmonary artery segmentations from the ImageCHD dataset:

```bash
python create_imagechd_pa_dataset.py --source /path/to/ImageCHD_dataset --dest /path/to/output --pa-label 7
```

Extract pulmonary artery segmentations from the CHD68 dataset:

```bash
python create_pa_dataset.py --source /path/to/CHD68_dataset --dest /path/to/output --pa-label 7
```

### Analyzing a PA Dataset

Generate comprehensive statistics for a PA dataset:

```bash
python analyze_pa_dataset.py --dataset /path/to/PA_dataset --pa-label 7 --output /path/to/stats_output
```

Basic CHD68 dataset analysis with label distribution:

```bash
python stats.py
```

Analyze PARSE dataset:

```bash
python pulmonary_artery_analyzer.py --data_dir /path/to/PARSE_dataset --output_dir /path/to/output
```

### Checking for Dataset Duplicates

Identify duplicate cases between two datasets:

```bash
python check_pa_dataset_duplicates.py --dataset1 /path/to/first_dataset --dataset2 /path/to/second_dataset --output duplication_report.md
```

## Dataset Statistics

The analysis tools generate various statistics including:

- Volume distribution of PA segmentations
- Percentage of slices containing PA segmentations
- Class imbalance metrics
- Shape and intensity features
- 3D morphological characteristics

Output includes both CSV files with raw data and visualizations in PNG format.

### Sample Output

The `analyze_pa_dataset.py` script generates comprehensive visual reports including:

- PA volume distribution histograms
- Slice-by-slice PA area heatmaps
- Class distribution pie charts
- PA centroid position visualizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use these tools in your research, please cite:

```
@misc{pa-segmentation-analysis,
  author = {Your Name},
  title = {Pulmonary Artery Segmentation Analysis Tools},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/pa-segmentation-analysis}
}
```

## Acknowledgments

- The PARSE dataset contributors
- ImageCHD and CHD68 dataset creators
