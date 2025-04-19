import os
import torch
# import data_setup,engine,model_builder,distutils
from torchvision import transforms


def train_loop(dataset: torch.utils.data.DataLoader,
               model: torch.nn.Module,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device):


    """
    Running a PyTorch train.py script on the command line with various hyperparameter settings.
    __model,__batch_size, __lr, and __num_epochs are known as argument flags

    Takes the train dataloader, perform on them, and calcualte loss and accuracy

    Args:
        train_data: torch.utils.data.DataLoader
        model: torch.nn.Module

    """
    model.to(device)

    model.train()

    #initialize train loss and train accuracy
    train_loss,train_acc = 0,0
    for batch,(X,y) in enumerate(dataset):

        # Pass the model to the target device
        X,y = X.to(device),y.to(device)

        # Forward pass
        y_logit = model(X)
        Y_pred_prob = torch.argmax(torch.softmax(y_logit,dim=1),dim=1)

        # Calculate loss
        loss = loss_fn(y_logit,y)
        train_loss += loss.item()

        # Set the gradient zero
        optimizer.zero_grad()

        # Backward Propagation
        loss.backward()

        #optimizer step
        optimizer.step()

    # Calculate average loss of each batch
    train_loss = train_loss/len(dataset)
    train_acc = train_acc/len(dataset)
