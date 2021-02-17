from torchvision import datasets
import torch

def Cifar100_data_load(transform):
    train_dataset = datasets.CIFAR100(
        root='./.data',
        train=True,
        transform=transform,
        download=True
    )
    test_dataset = datasets.CIFAR100(
        root='./.data',
        train=False,
        transform=transform,
        download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=True
    )
    return train_loader, test_loader