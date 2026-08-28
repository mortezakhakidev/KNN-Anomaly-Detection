# Synthetic Outlier Generation & Refinement using GMM and KNN

## Overview
This repository provides a robust Python pipeline for generating and refining synthetic outliers (anomalies) in tabular datasets. It is specifically designed to create two types of outliers:

- **Local Outliers** – Subtle anomalies that lie within the overall range of normal data but deviate from local density patterns.
- **Global Outliers** – Extreme points that fall far outside the normal data distribution.

These synthetic anomalies are then **iteratively refined** using a **K-Nearest Neighbors (KNN) classifier** to ensure they are challenging to detect—making them ideal for testing, benchmarking, or validating anomaly detection frameworks.

> **All source code is written in Python.**

---

## Key Features
- **Gaussian Mixture Model (GMM) for Local Outliers** – Automatically selects the best covariance type and number of components using the Bayesian Information Criterion (BIC), then scales the covariance to generate realistic local deviations.
- **Uniform Distribution for Global Outliers** – Generates extreme points outside the normal data range.
- **Iterative Refinement with KNN** – Employs a KNN classifier to filter out "easy" anomalies, retaining only those that are difficult to distinguish from normal data.
- **Flexible Input** – Works with any dataset; simply adjust the column names or file path in the script.
- **Reproducibility** – Fixed random seeds ensure consistent results across runs.

---

## How It Works

1. **Data Loading**  
   The script reads a CSV file (by default, `dataset/your_dataset.csv`) and extracts the specified columns of interest.  
   **Note:** You are not limited to this specific file. You can use **any CSV dataset** containing numerical features. Just update the `columns_of_interest` variable (e.g., `["temp", "humidity"]`) or the file path to match your data.

2. **Local Outlier Generation**  
   - The data is split into training and test sets.  
   - Multiple GMMs are trained with different covariance types (`spherical`, `tied`, `diag`, `full`) and component counts (1–2).  
   - The model with the lowest BIC score on the test set is selected.  
   - The covariance matrix of the best model is scaled by a factor (×5) to create subtle outliers.  
   - New synthetic points are sampled from this modified GMM.

3. **Global Outlier Generation**  
   - A uniform distribution is defined using 90% of the minimum and 110% of the maximum of each feature.  
   - Random points are sampled from this expanded range to create extreme global anomalies.

4. **Iterative Refinement**  
   - A KNN classifier is trained on a mix of normal data and candidate outliers.  
   - Outliers that are correctly classified as anomalies are retained.  
   - Misclassified outliers are replaced with new samples drawn from the remaining candidate pool.  
   - This process repeats until the number of rejected outliers falls below a threshold (default: 50), ensuring the final set contains only "hard" anomalies.

5. **Output**  
   The refined local and global outlier sets are saved as CSV files in the `dataset/` directory.

---

## How to Run

1. **Prepare your dataset**  
   Place your CSV file inside the `dataset/` folder and rename it to `your_dataset.csv` (or change the filename in the script).  
   **Important:** Make sure your dataset contains the columns you define in `columns_of_interest` (e.g., `temp` and `humidity`). If your data has different feature names, simply update this list in the script.

2. **Execute the script**  
   From the project root directory, run:
   ```bash
   python anomaly_detection_KNN.py

3. **Check the outputs**  
   The refined outlier datasets will be saved as:
   - `dataset/refined_local_outliers.csv`
   - `dataset/refined_global_outliers.csv`

---

## Customization Options

You can easily adjust the following parameters inside `anomaly_detection_KNN.py` to fit your specific dataset and use case:

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `columns_of_interest` | The feature columns to use from your CSV | `["temp", "humidity"]` |
| `count` | Number of outliers to generate per type | `500` |
| `factor` | Covariance scaling factor for local outliers (higher = more extreme) | `5` |
| `sample_size` | Initial number of outliers used in each refinement iteration | `500` |
| `max_removals` | Stopping criterion: minimum number of rejected outliers per iteration | `50` |

---

## License
This project is provided for academic and research purposes. Please cite this repository if you use the code in your work.

---

## References
- Scikit-learn: Machine Learning in Python (Pedregosa et al., 2011)
- Gaussian Mixture Models for density estimation and anomaly generation.
