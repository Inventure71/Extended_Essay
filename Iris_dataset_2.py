import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm

number_of_trainings = 20

def setup_dirs(k):
    if not os.path.exists(f"models/{k}"):
        os.makedirs(f"models/{k}")
    if not os.path.exists(f"models/{k}/stats"):
        os.makedirs(f"models/{k}/stats")


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


def loop(k, train_loader, test_loader):
    torch.manual_seed(42)

    layer_sizes = [4, 10, 20, 50, 100, 200, 500, 1000, 2000]
    layer_counts = [2, 3, 4]
    max_epochs = 10000000
    max_no_improvement = 1000
    min_delta = 0.000001

    stats = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for size in layer_sizes:
        for count in layer_counts:
            start_time = time.time()

            layers = [nn.Linear(4, size), nn.ReLU()]
            for _ in range(count - 1):
                layers.extend([nn.Linear(size, size), nn.ReLU()])
            layers.append(nn.Linear(size, 3))
            model = nn.Sequential(*layers).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

            key = f"Size: {size}, Count: {count}"
            stats[key] = {
                'train_loss': [],
                'test_loss': [],
                'best_val_loss': float('inf'),
                'num_epochs': 0,
                'time for training': 0
            }

            pbar = tqdm(range(max_epochs), desc=f"Model {key}")

            prev_val_loss = float('inf')
            num_epochs_no_improvement = 0

            for epoch in pbar:
                model.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                epoch_loss = running_loss / len(train_loader.dataset)

                model.eval()
                running_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        running_loss += loss.item() * inputs.size(0)
                val_loss = running_loss / len(test_loader.dataset)

                pbar.set_description(f"Model {key}, Train Loss: {epoch_loss:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)

                if val_loss < stats[key]['best_val_loss'] - min_delta:
                    stats[key]['best_val_loss'] = val_loss
                    torch.save(model.state_dict(), f"models/{k}/model_{key}.pth")
                    num_epochs_no_improvement = 0
                else:
                    num_epochs_no_improvement += 1

                stats[key]['train_loss'].append(epoch_loss)
                stats[key]['test_loss'].append(val_loss)
                stats[key]['num_epochs'] += 1

                if num_epochs_no_improvement >= max_no_improvement:
                    print(f"No improvement for {num_epochs_no_improvement} epochs. Early stopping.")
                    break

            stats[key]['time for training'] = time.time() - start_time
            data_table = pd.DataFrame(stats[key])
            data_table.to_csv(f"models/{k}/stats/stats_{key}.csv", index=False)

    print('Training Complete')


for k in range(0, number_of_trainings):
    setup_dirs(k)
    train_loader, test_loader = setup_data()
    loop(k, train_loader, test_loader)
