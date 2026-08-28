########################################################################################################
#####
#####
##### Generate Global and Local outliers in order to be used for testing anomaly detection frameworks
##### Morteza Khaki - University of Tabriz
#####
#####
########################################################################################################

import os
import random
import numpy as np
import pandas as pd
import sklearn
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

# Set fixed seed for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

# Load and preprocess dataset
columns_of_interest = ["temp", "humidity"]
data_frame = pd.read_csv('dataset/h358data_modified.csv')
data_frame = data_frame[columns_of_interest].dropna().reset_index(drop=True)

# Function to generate localized synthetic anomalies
def generate_local_outliers(data, count=500):
    factor = 5
    records = data.values
    train_set, test_set = train_test_split(records, test_size=0.3, random_state=1)
    best_model, lowest_criterion = None, float('inf')
    bic_scores = []

    for covariance_type in ["spherical", "tied", "diag", "full"]:
        for components in range(1, 3):
            model = GaussianMixture(n_components=components, covariance_type=covariance_type, random_state=seed_value)
            model.fit(train_set)
            score = model.bic(test_set)
            bic_scores.append(score)
            if score < lowest_criterion:
                lowest_criterion, best_model = score, model

    optimized_model = sklearn.base.clone(best_model)
    optimized_model.weights_ = best_model.weights_
    optimized_model.means_ = best_model.means_
    optimized_model.covariances_ = factor * best_model.covariances_

    samples = optimized_model.sample(count)[0]
    outlier_df = pd.DataFrame(samples, columns=data.columns)
    outlier_df['label'] = 1
    return outlier_df

# Generate local outliers
localized_anomalies = generate_local_outliers(data_frame, count=500)

# Function to create global anomalies
def generate_global_outliers(data, count=500):
    stats = data.describe()
    min_vals = stats.loc['min'] * 0.9
    max_vals = stats.loc['max'] * 1.1
    range_vals = max_vals - min_vals

    dist = scipy.stats.uniform(loc=min_vals, scale=range_vals)
    samples = [dist.rvs(size=min_vals.shape) for _ in range(count)]

    global_outlier_df = pd.DataFrame(samples, columns=data.columns)
    global_outlier_df['label'] = 1
    return global_outlier_df

# Generate global outliers
globalized_anomalies = generate_global_outliers(data_frame, count=500)

# Function to iteratively refine outlier sets
def refine_outliers(base_data, anomaly_data, model='KNN', sample_size=500, max_removals=50):
    retained_count = float('inf')
    original_data = base_data.copy()
    original_data['label'] = 0
    remaining_anomalies = anomaly_data.copy()

    chosen_indices = random.sample(remaining_anomalies.index.tolist(), sample_size)
    current_outliers = remaining_anomalies.loc[chosen_indices]
    remaining_anomalies.drop(chosen_indices, inplace=True)

    if model == 'KNN':
        classifier = KNeighborsClassifier()

    while retained_count > max_removals:
        training_data = pd.concat((original_data, current_outliers), axis=0).reset_index(drop=True)
        X_train, y_train = training_data.iloc[:, :-1].values, training_data.iloc[:, -1].values
        classifier.fit(X_train, y_train)

        predictions = classifier.predict(current_outliers.iloc[:, :-1].values)
        correct_outliers = current_outliers.loc[predictions == 1]
        retained_count = np.sum(predictions == 0)

        print(f"Removed: {retained_count}")

        new_indices = random.sample(remaining_anomalies.index.tolist(), retained_count)
        newly_sampled = remaining_anomalies.loc[new_indices]
        remaining_anomalies.drop(new_indices, inplace=True)

        current_outliers = pd.concat((correct_outliers, newly_sampled), axis=0).reset_index(drop=True)

    return current_outliers

# Refine anomalies
filtered_local_anomalies = refine_outliers(data_frame, localized_anomalies, sample_size=500)
filtered_global_anomalies = refine_outliers(data_frame, globalized_anomalies, sample_size=500)

# Save results to files
filtered_local_anomalies.to_csv("dataset/refined_local_outliers.csv", index=False)
filtered_global_anomalies.to_csv("dataset/refined_global_outliers.csv", index=False)
