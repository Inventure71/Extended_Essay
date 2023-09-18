import os
import torch
from torch import nn
import pandas as pd
import time
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader


def load_model(size, count):
    layers = [nn.Linear(4, size), nn.ReLU()]
    for _ in range(count - 1):
        layers.extend([nn.Linear(size, size), nn.ReLU()])
    layers.append(nn.Linear(size, 3))
    model = nn.Sequential(*layers).to(device)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def counting_parameters():
    layer_sizes = [4, 10, 20, 50, 100, 200, 500, 1000, 2000]
    layer_counts = [2, 3, 4]
    #layer_counts = [2]

    results = []

    number_of_trainings = 1
    train_loader, test_loader = setup_data()  # Assuming setup_data is available in this script

    for k in range(number_of_trainings):
        for size in layer_sizes:
            for count in layer_counts:
                # Load the model
                model_key = f"Size: {size}, Count: {count}"
                model_path = f"models/{k}/model_{model_key}.pth"
                file_path = f"models/stats/{k}/stats_{model_key}.pth"

                if not os.path.exists(model_path):  # Check if the model exists
                    continue

                model = load_model(size, count)

                total_parameters = count_parameters(model)
                print(f"The model has {total_parameters} parameters.")

                result = {
                    "Model Key": model_key,
                    "Parameters": total_parameters
                }

                results.append(result)

    # Save the results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv("model_parameters.csv", index=False)


# Given layer_sizes and layer_counts
layer_sizes = [4, 10, 20, 50, 100, 200, 500, 1000, 2000]
layer_counts = [2, 3, 4]
layer_counts = [4]

def loss_over_time ():
    # Update the file pattern and configurations based on the provided lists
    file_pattern = "averaged_stats_Size: {}, Count: {}.csv"
    configurations = [(size, count) for size in layer_sizes for count in layer_counts]
    directory = r"models_useful_data/IRIS dataset/Avaraged stats/"

    # Start plotting
    plt.figure(figsize=(18, 12))

    #set colors
    colormap = plt.cm.tab20  # This colormap provides a good set of distinct colors
    colors = [colormap(i % 20) for i in range(len(configurations))]

    #colors = [colormap(i) for i in range(len(configurations))]
    line_styles = ['-', '-', '-', '-']

    # Iterate through each file
    for idx, config in enumerate(configurations):
        file_path = os.path.join(directory, file_pattern.format(config[0], config[1]))

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Calculate time per epoch
            total_time = df["time for training_excl_outliers"].iloc[0]
            num_epochs = df["num_epochs"].iloc[0]
            time_per_epoch = total_time / num_epochs
            times = [time_per_epoch * (epoch + 1) for epoch in range(int(num_epochs))]

            # Plotting
            #plt.plot(times, df["train_loss"][:int(num_epochs)], label=f"Train Loss (Size: {config[0]}, Count: {config[1]})", color=colors[idx], linestyle=line_styles[idx % len(line_styles)])

            plt.plot(times, df["test_loss"][:int(num_epochs)],
                 label=f"Test (Size: {config[0]}, Count: {config[1]})",
                 color=colors[idx],
                 linestyle=line_styles[(idx + 1) % len(line_styles)])


    # Setting labels, title, legend, and displaying the plot
    plt.xlabel("Time (s)")
    plt.ylabel("Loss")
    plt.title("Test loss Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig()
    plt.show()


def plot_benchmark_data_double_columns(file_path):
    """
    Plots the benchmark data from the given file path using double columns for Accuracy and Time Taken.

    Parameters:
    - file_path: The path to the benchmark CSV file.
    """
    # Read the data
    df = pd.read_csv(file_path)

    # Set up the figure and axes
    fig, ax1 = plt.subplots(figsize=(18, 8))
    ax2 = ax1.twinx()

    # Sort the data by accuracy for better visualization
    df = df.sort_values(by='Time Taken (s)', ascending=True)

    # Define the width of the bars and the positions for each bar
    width = 0.35
    ind = range(len(df))

    # Plot accuracy bars
    p1 = ax1.bar(ind, df['Accuracy'], width, color='g', alpha=0.6, label='Accuracy')

    # Plot time taken bars
    p2 = ax2.bar([i + width for i in ind], df['Time Taken (s)'], width, color='b', alpha=0.6, label='Time Taken')

    # Set labels, title, and legends
    ax1.set_xlabel('Model Configuration')
    ax1.set_ylabel('Accuracy', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax2.set_ylabel('Time Taken (s)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax1.set_xticks([i + width / 2 for i in ind])
    ax1.set_xticklabels(df['Model Key'], rotation=45, ha="right")

    # Add colored grid
    ax1.yaxis.grid(True, color='g', linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.yaxis.grid(True, color='b', linestyle='--', linewidth=0.5, alpha=0.7)

    # Title and show
    plt.title("Benchmark Results: Accuracy and Time Taken")
    fig.tight_layout()
    plt.show()


def plot_benchmark_with_parameters(benchmark_file_path, parameters_file_path):
    """
    Plots the benchmark data using the number of parameters on the x-axis.

    Parameters:
    - benchmark_file_path: The path to the benchmark CSV file.
    - parameters_file_path: The path to the CSV file containing the number of parameters.
    """
    # Read the data
    benchmark_df = pd.read_csv(benchmark_file_path)
    parameters_df = pd.read_csv(parameters_file_path)

    # Merge the dataframes on the model key to get the corresponding number of parameters
    merged_df = benchmark_df.merge(parameters_df, on='Model Key')

    # Sort the merged dataframe by the number of parameters
    merged_df = merged_df.sort_values(by='Parameters')

    # Set up the figure and axes
    fig, ax1 = plt.subplots(figsize=(18, 8))
    ax2 = ax1.twinx()

    # Define the width of the bars and the positions for each bar
    width = 0.35
    ind = range(len(merged_df))

    # Plot accuracy bars
    p1 = ax1.bar(ind, merged_df['Accuracy'], width, color='b', alpha=0.6, label='Accuracy')

    # Plot time taken bars
    p2 = ax2.bar([i + width for i in ind], merged_df['Time Taken (s)'], width, color='r', alpha=0.6, label='Time Taken')

    # Set labels, title, and legends
    ax1.set_xlabel('Number of Parameters')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.set_ylabel('Time Taken (s)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.set_xticks([i + width / 2 for i in ind])
    ax1.set_xticklabels(merged_df['Parameters'], rotation=45, ha="right")

    # Add colored grid
    ax1.yaxis.grid(True, color='b', linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.yaxis.grid(True, color='r', linestyle='--', linewidth=0.5, alpha=0.7)

    # Draw lines from the top of each bar
    for bar in p1:
        height = bar.get_height()
        ax1.plot([bar.get_x() + bar.get_width() / 2, bar.get_x() + bar.get_width() / 2], [0, height], 'b-', alpha=0.6)

    for bar in p2:
        height = bar.get_height()
        ax2.plot([bar.get_x() + bar.get_width() / 2, bar.get_x() + bar.get_width() / 2], [0, height], 'r-', alpha=0.6)

    # Title and show
    plt.title("Benchmark Results: Accuracy and Time Taken with Number of Parameters")
    fig.tight_layout()
    plt.show()


# Testing the function with the provided files
#plot_benchmark_with_parameters("models_useful_data/IRIS dataset/model_average_benchmarks.csv", "model_parameters.csv")

# Testing the function with the provided file
plot_benchmark_data_double_columns("models_useful_data/IRIS dataset/model_average_benchmarks.csv")

#counting_parameters()

#loss_over_time()

