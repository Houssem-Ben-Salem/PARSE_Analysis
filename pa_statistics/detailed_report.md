# Pulmonary Artery Dataset Detailed Report

## Dataset Overview

- Total cases: 110
- Cases with PA segmentations: 110
- Total slices: 28882
- Slices with PA: 17440 (60.38%)
- Total PA volume: 47482128.00 mm³

## Class Balance

- Total voxels: 7513630882
- PA voxels: 47482128 (0.6319%)
- Background voxels: 7466148754 (99.3681%)

## PA Volume Distribution

- Mean PA volume: 431655.71 mm³
- Median PA volume: 344221.50 mm³
- Std. deviation: 382668.89 mm³
- Min PA volume: 34557.00 mm³
- Max PA volume: 2560758.00 mm³
- 25th percentile: 187712.50 mm³
- 75th percentile: 524723.25 mm³

## Case Categories

- Small PA cases: 28 (25.45%)
- Medium PA cases: 54 (49.09%)
- Large PA cases: 28 (25.45%)

## Slice Distribution

- Mean slices per case: 262.56
- Median slices per case: 275.00
- Std. deviation: 82.75

## Segmentation Challenges

Based on the statistics, the following challenges are anticipated for PA segmentation:

1. **Class Imbalance**: The background-to-PA ratio is 157.24:1, indicating severe class imbalance. Consider using weighted loss functions or focal loss.

2. **Size Variability**: PA volume varies from 34557.00 mm³ to 2560758.00 mm³, with moderate variability. Ensure training data covers the full range of volumes.

3. **Sparse Segmentation**: Only 60.38% of slices contain PA segmentations. This is manageable with standard 2D or 3D segmentation approaches.

