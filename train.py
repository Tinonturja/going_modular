import torch.utils.data


def train_loop(dataset: torch.utils.data.DataLoader,
               model: torch.nn.Module,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device):


    """Takes the train dataloader, perform on them, and calcualte loss and accuracy

    Args:
        train_data: torch.utils.data.DataLoader
        model: torch.nn.Module

    """
    model.to(device)

    model.train()

    for batch,(X,y) in enumerate(dataset):

        # Pass the model to the target device
        X,y = X.to(device),y.to(device)

        # Forward pass
        y_logit = model(X)

