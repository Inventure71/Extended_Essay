import os
import time
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

def setup_data():
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Download and load the training data
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    return trainloader, testloader

def evaluate_model(model, dataloader, device):
    all_labels = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = sum(1 for x, y in zip(all_labels, all_predictions) if x == y) / len(all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')

    return accuracy, precision, recall, f1

def benchmark(models_path, test_loader, device):
    results = []
    for model_path in os.listdir(models_path):
        if model_path.endswith('.pth'):

            size, count = model_path.split('_')[2], model_path.split('_')[4].removesuffix(".pth")

            layers = [nn.Flatten(), nn.Linear(784, int(size)), nn.ReLU()]
            for _ in range(int(count) - 1):
                layers.extend([nn.Linear(int(size), int(size)), nn.ReLU()])
            layers.append(nn.Linear(int(size), 10))
            model = nn.Sequential(*layers).to(device)

            model.load_state_dict(torch.load(os.path.join(models_path, model_path)))

            start_time = time.time()
            accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device)


            _ = model(test_loader.dataset[0][0].unsqueeze(0).to(device))
            inference_time = time.time() - start_time

            results.append({
                'model': model_path,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'inference_time': inference_time
            })

            print(results)

    return results

'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_, test_loader = setup_data()
results = benchmark(r"models_useful_data/MNIST dataset/MNIST_models/0", test_loader, device)
results2 = benchmark(r"models_useful_data/MNIST dataset/MNIST_models/1", test_loader, device)
results3 = benchmark(r"models_useful_data/MNIST dataset/MNIST_models/2", test_loader, device)


# Convert results to a DataFrame for better visualization and saving
df_results = pd.DataFrame(results)
df_results1 = pd.DataFrame(results2)
df_results2 = pd.DataFrame(results3)

# Save the results to a CSV file
save_path = r"models_useful_data/MNIST dataset/benchmark_results_0.csv"
save_path1 = r"models_useful_data/MNIST dataset/benchmark_results_1.csv"
save_path2 = r"models_useful_data/MNIST dataset/benchmark_results_2.csv"

df_results.to_csv(save_path, index=False)
df_results1.to_csv(save_path1, index=False)
df_results2.to_csv(save_path2, index=False)

#print(df_results)
print(f"Results saved to: {save_path}")
'''

