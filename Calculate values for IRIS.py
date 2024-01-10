#process the values for IRIS

import os
import pandas as pd
import numpy as np

def detect_outliers(values):
    """
    Detect outliers using IQR method.
    Returns a list of indices of the outliers in the input list.
    """
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = [i for i, value in enumerate(values) if value < lower_bound or value > upper_bound]
    return outliers

'''FOR GENERAL STATS'''
def average_and_save_outliers(layer_sizes, layer_counts, number_of_trainings):
    for size in layer_sizes:
        for count in layer_counts:
            data_dicts = []  # to store data from all files for this model size and count
            max_epochs = 0  # to know how many lines we need to iterate over

            # Load data from all 20 files and determine the maximum number of epochs
            for k in range(number_of_trainings):
                file_path = f"models/{k}/stats/stats_Size: {size}, Count: {count}.csv"
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    data_dicts.append(df.to_dict('list'))
                    max_epochs = max(max_epochs, len(df))

            # Average and detect outliers for each epoch
            averaged_data = []
            for epoch in range(max_epochs):
                epoch_data = {
                    'train_loss': [],
                    'test_loss': [],
                    'best_val_loss': [],
                    'num_epochs': [],
                    'time for training': []
                }
                for data_dict in data_dicts:
                    for key in epoch_data:
                        # Check if there's data for this epoch in the current file
                        if len(data_dict[key]) > epoch:
                            epoch_data[key].append(data_dict[key][epoch])

                averaged_epoch_data = {}
                outliers_detected = {}
                averaged_excluding_outliers = {}

                for key, values in epoch_data.items():
                    outliers_detected[key] = detect_outliers(values)
                    values_excluding_outliers = [v for i, v in enumerate(values) if i not in outliers_detected[key]]

                    averaged_epoch_data[key] = np.mean(values)
                    averaged_excluding_outliers[key] = np.mean(
                        values_excluding_outliers) if values_excluding_outliers else ''

                # For non-changing values, only write them for the first epoch
                if epoch > 0:
                    for key in ['best_val_loss', 'num_epochs', 'time for training']:
                        averaged_epoch_data[key] = ''
                        outliers_detected[key] = []
                        averaged_excluding_outliers[key] = ''

                averaged_data.append((averaged_epoch_data, averaged_excluding_outliers, outliers_detected))

            # Save to CSV
            columns = []
            data = []
            for item in averaged_data:
                row = {}
                for key in item[0]:
                    row[key] = item[0][key]
                    row[f"{key}_excl_outliers"] = item[1][key]
                    row[f"{key}_outliers"] = ', '.join(map(str, item[2][key]))
                data.append(row)

            averaged_df = pd.DataFrame(data)
            output_path = f"models/averaged_stats_Size: {size}, Count: {count}.csv"
            averaged_df.to_csv(output_path, index=False)

    return "Averaging and outlier detection and saving completed."



'''FOR RANKING:CSV'''
# Load the ranking.csv file into a DataFrame
ranking_df = pd.read_csv("models_useful_data/ranking.csv")


def calculate_avg_excluding_outliers_and_save(df, group_columns, metric_columns, output_folder):
    """
    Calculate average metrics for each group in group_columns, excluding outliers, and save the results to CSV files.
    """
    avg_data = []
    outliers_data = []

    for name, group in df.groupby(group_columns):
        row = dict(zip(group_columns, name))
        outliers_row = dict(zip(group_columns, name))
        for metric in metric_columns:
            values = group[metric].tolist()
            outliers = detect_outliers(values)
            values_excl_outliers = [v for i, v in enumerate(values) if i not in outliers]
            row[metric] = np.mean(values_excl_outliers) if values_excl_outliers else np.nan
            outliers_row[f"{metric}_outliers"] = outliers
        avg_data.append(row)
        outliers_data.append(outliers_row)

    avg_df = pd.DataFrame(avg_data)
    outliers_df = pd.DataFrame(outliers_data)

    # Save the DataFrames to the specified output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    avg_df.to_csv(os.path.join(output_folder, "averaged_ranking.csv"), index=False)
    outliers_df.to_csv(os.path.join(output_folder, "outliers_ranking.csv"), index=False)

    return "Averaging, outlier detection, and saving to CSV completed."

group_columns = ['size', 'count']
metric_columns = ['epochs', 'f1_score', 'training_time', 'time_for_epoch', 'evaluation_time']

output_folder = "models_useful_data"
calculate_avg_excluding_outliers_and_save(ranking_df, group_columns, metric_columns, output_folder)



layer_sizes = [4, 10, 20, 50, 100, 200, 500, 1000, 2000]
layer_counts = [2, 3, 4]  
number_of_trainings = 20

#average_and_save_outliers(layer_sizes, layer_counts, number_of_trainings)
