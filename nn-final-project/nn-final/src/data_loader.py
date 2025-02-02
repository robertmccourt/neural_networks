import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import pickle

from LeNet import CNN
# from LeNet5 import LeNet5

def get_data_loaders(batch_size: int = 64, dataset: str = "mnist"):
    """
    Load MNIST or CIFAR-10 datasets and return DataLoaders with resized 32x32 inputs.
    :param batch_size: Batch size for DataLoader.
    :param dataset: Dataset to load ('mnist' or 'cifar').
    :return: Train and test DataLoaders for the selected dataset.
    """
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat grayscale 3 times
        ])

        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif dataset == "cifar":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    else:
        raise ValueError("Invalid dataset. Choose 'mnist' or 'cifar'.")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    batch_size = 32
    epochs = 10
    
    # dataset = "mnist"
    dataset = "cifar"

    train_loader, test_loader = get_data_loaders(batch_size=batch_size, dataset=dataset)
    
    batch = next(iter(train_loader))
    input_data, label = batch
    input_data = input_data.numpy()
    label = label.numpy()
    
    model = CNN(input_shape=(3, 32, 32), num_classes=10)

    # adam
    adam_runs = {}
    for i in range(3):
        model = CNN(input_shape=(3, 32, 32), num_classes=10)
        training_accuracies = model.train(input_data,
                                        label,
                                        epochs,
                                        learning_rate=0.01,
                                        optimizer="adam")
        
        test_accuracy = model.getAccuracy(test_loader)


        adam_runs[i] = (training_accuracies, test_accuracy)

    with open(f"{dataset}_adam_runs.pkl", "wb") as f:
        pickle.dump(adam_runs, f)

    # GD
    gd_runs = {}
    for i in range(3):
        model = CNN(input_shape=(3, 32, 32), num_classes=10)
        training_accuracies = model.train(input_data,
                                        label,
                                        epochs,
                                        learning_rate=0.001,
                                        optimizer="gd")
        
        test_accuracy = model.getAccuracy(test_loader)


        gd_runs[i] = (training_accuracies, test_accuracy)

    with open(f"{dataset}_gd_runs.pkl", "wb") as f:
        pickle.dump(gd_runs, f)  

    
