import os
import pandas as pd

# Path to where your MNIST_models are saved
base_path = "models_useful_data/MNIST dataset/MNIST_models"

# Layer sizes and counts as in your code
layer_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
layer_counts = [1, 2, 3, 4, 5]

results = []

# Loop over each size and count combination
for size in layer_sizes:
    for count in layer_counts:
        key = f"Size_{size}_Count_{count}"

        # Lists to store dataframes from each iteration
        dfs = []

        # Load data from each iteration and store in dfs list
        for k in range(3):  # since you have 3 folders: 0, 1, 2
            file_path = os.path.join(base_path, str(k), "stats", f"stats_{key}.csv")

            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                dfs.append(data)

        # Check if dataframes were loaded
        if dfs:
            # Concatenate data on axis 0 (rows) and group by index (which corresponds to epoch number),
            # then compute mean for each group (i.e., for each epoch)
            avg_df = pd.concat(dfs).groupby(level=0).mean()

            # As these columns remain same, take from the first dataframe
            avg_df['best_val_loss'] = dfs[0]['best_val_loss'].values
            avg_df['num_epochs'] = dfs[0]['num_epochs'].values
            avg_df['time for training'] = dfs[0]['time for training'].values

            # Save average data for current size and count combination to a CSV file
            avg_df.to_csv(os.path.join(base_path, f"averaged_results_{key}.csv"), index=False)
            print(f"Results for {key} saved to: {os.path.join(base_path, f'averaged_results_{key}.csv')}")

