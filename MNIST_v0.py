import os
import time
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import pandas as pd
from tqdm import tqdm

number_of_trainings = 3

def setup_dirs(k):
    if not os.path.exists(f"MNIST_models/{k}"):
        os.makedirs(f"MNIST_models/{k}")
    if not os.path.exists(f"MNIST_models/{k}/stats"):
        os.makedirs(f"MNIST_models/{k}/stats")


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



def loop(k, train_loader, test_loader):
    torch.manual_seed(42)

    layer_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    layer_counts = [1, 2, 3, 4, 5]
    max_epochs = 10000000
    max_no_improvement = 50
    min_delta = 0.0001

    stats = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for size in layer_sizes:
        for count in layer_counts:
            start_time = time.time()

            layers = [nn.Flatten(), nn.Linear(784, size), nn.ReLU()]  #change 4 in 784 because new database has 28*28 pixels
            for _ in range(count - 1):
                layers.extend([nn.Linear(size, size), nn.ReLU()])
            layers.append(nn.Linear(size, 10)) # changed 3 to 10 because there are 10 classes in MNIST
            model = nn.Sequential(*layers).to(device)


            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

            key = f"Size_{size}_Count_{count}"
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
                    torch.save(model.state_dict(), f"MNIST_models/{k}/model_{key}.pth")
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
            data_table.to_csv(f"MNIST_models/{k}/stats/stats_{key}.csv", index=False)

    print('Training Complete')


for k in range(0, number_of_trainings):
    setup_dirs(k)
    train_loader, test_loader = setup_data()
    loop(k, train_loader, test_loader)
