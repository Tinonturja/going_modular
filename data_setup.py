"""
Contains functionality for creating PyTorch DataLoader's for image classification data.
"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform:transforms.Compose,
        batch_size: int,
):
    """
    Creates training and testing DataLoaders.

    Takes in a training directory and a testing directory, and creates training and testing dataloaders, which track the training and testing file batchwise

    Args:
        train_dir: Path to training directory
        test_dir: Path to testing directory
        transform: torchvision transforms to perform on training and testing data
        batch_size: How many images the computer will show in one epoch


    Returns:
        A tuple of (train_dataloader,test_dataloader, class_names).
        Where class_names is a list of the target classes.

        Example usage:



    """

    train_data = datasets.ImageFolder(root=train_dir,
                                    transform= transform)
    test_data = datasets.ImageFolder(root= test_dir,
                                     transform=transform)

    # Get Class names
    class_names = train_data.classes

    # Turning Image into DataLoaders
    batch_size = 32
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True)

    test_dataloader = DataLoader(dataset = test_data,
                                 batch_size=batch_size,
                                 shuffle= False,
                                 pin_memory=True)
    

    return train_dataloader,test_dataloader
