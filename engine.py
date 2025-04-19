import os
import torch
from tqdm.auto import tqdm
from typing import Dict,List,Tuple
def train_loop(dataset: torch.utils.data.DataLoader,
               model: torch.nn.Module,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device):


    """
    Running a PyTorch engine.py script on the command line with various hyperparameter settings.
    __model,__batch_size, __lr, and __num_epochs are known as argument flags

    Takes the train dataloader, perform on them, and calcualte loss and accuracy

    Args:
        train_data: torch.utils.data.DataLoader
        model: torch.nn.Module

    """
    model.to(device)

    model.train()

    #initialize train loss and train accuracynm
    train_loss,train_acc = 0,0
    for batch,(X,y) in enumerate(dataset):

        # Pass the model to the target device
        X,y = X.to(device),y.to(device)

        # Forward pass
        y_logit = model(X)
        y_pred_prob = torch.argmax(torch.softmax(y_logit,dim=1),dim=1)

        # Calculate loss
        loss = loss_fn(y_logit,y)
        train_loss += loss.item()
        acc = (y_pred_prob==y).sum().item()
        train_acc +=acc
        # Set the gradient zero
        optimizer.zero_grad()

        # Backward Propagation
        loss.backward()

        #optimizer step
        optimizer.step()

    # Calculate average loss of each batch
    train_loss = train_loss/len(dataset)
    train_acc = train_acc/len(dataset)

    return train_loss,train_acc


def test_loop(model:torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device)->Tuple[float,float]:
    """Tests a PyTorch model for a single epoch.

      Turns a target PyTorch model to "eval" mode and then performs
      a forward pass on a testing dataset.

      Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

      Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy)"""

    # Put the model in eval mode
    model.eval()


    # Set up loss and accuracy values
    test_loss, test_acc = 0,0

    with torch.inference_mode():
        for batch, (X,y) in enumerate(dataloader):

            # Pass the model to the target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_logit = model(X)
            y_pred_prob = torch.argmax(torch.softmax(y_logit, dim=1), dim=1)

            # Calculate loss
            loss = loss_fn(y_logit, y)
            test_loss += loss.item()
            acc = (y_pred_prob == y).sum().item()
            test_acc += acc


    # average loss of the total dataset
    test_loss = test_loss/len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss,test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device):
    """Trains and tests a PyTorch model.

      Passes a target PyTorch models through train_step() and test_step()
      functions for a number of epochs, training and testing the model
      in the same epoch loop.

      Calculates, prints and stores evaluation metrics throughout.

      Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

      Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for
        each epoch.
        In the form: {train_loss: [...],
                      train_acc: [...],
                      test_loss: [...],
                      test_acc: [...]}
        For example if training for epochs=2:
                     {train_loss: [2.0616, 1.0537],
                      train_acc: [0.3945, 0.3945],
                      test_loss: [1.2641, 1.5706],
                      test_acc: [0.3400, 0.2973]}
      """

    # Create empty result dictionary

    results = {"train_loss":[],
               "train_acc":[],
               "test_loss":[],
               "test_acc":[]}

    # Loop through training ang testing loop

    for epoch in tqdm(range(epochs)):
        train_loss,train_acc = train_loop(dataset=train_dataloader,
                                          model = model,
                                          loss_fn = loss_fn,
                                          optimizer = optimizer,
                                          device = device)

        test_loss,test_acc = test_loop(model = model,
                                       dataloader=test_dataloader,
                                       loss_fn=loss_fn,
                                       device=device)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc} | "


        )

        # update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # Return the filled results at the end of the epochs
    return results