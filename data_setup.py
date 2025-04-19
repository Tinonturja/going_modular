"""Takes the data directory and converts it into dataloader(images stored by batch) and converts
the image into tensors and in proper formats so that deep learning can be done
"""
# create dataset
import os
import torchvision
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int=NUM_WORKERS
):
    """Creates training and testing dataloaders

    Takes in a training directory and testing directory path and turns them into PyTorch Datasets
    and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory
        test_dir: Path to testing directory
        transform: Torchvision transforms to perform on training and testing dataset; convert them into tensors
        batch_size: int, Number of samples per batch in each of the DataLoader
        num_workers: int, An integer for number of workers per dataloader


    Returns:
        A tuple of (train_dataloader,test_dataloader, class_names).
        where class_names is a list of the target classes.
    """

    train_data = datasets.ImageFolder(train_dir,transform=transform)
    test_data = datasets.ImageFolder(test_dir,transform=transform)

    # Get Class names
    class_names = train_data.classes

    # Turn images into dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader,test_dataloader,class_names