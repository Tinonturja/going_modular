import os
import torch
import data_setup, engine, model,utils

from torchvision import transforms

# set up hyperparameters

NUM_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_UNITS = 20
LEARNING_RATE = 0.001

# setup dictionaries
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# setup target device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Creae transforms
train_transform = transforms.Compose([
    transforms.Resize(size = (64,64)),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Resize(size = (64,64)),
    transforms.ToTensor()
])


if __name__ == "__main__":
    # Create DataLoaders:
    train_dataloaders,test_dataloaders,class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                   test_dir=test_dir,
                                                                                   train_transform=train_transform,
                                                                                   test_transform=test_transform,
                                                                                   batch_size=BATCH_SIZE,
                                                                                   )

    # Create the model
    model_0 = model.TinyVGGModelV0(input_shape=3,
                                   hidden_units=10,
                                   output_shape=len(class_names))


    # Create Loss function and Optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_0.parameters(),
                                 lr = LEARNING_RATE)

    engine.train(model=model_0,
                 train_dataloader=train_dataloaders,
                 test_dataloader=test_dataloaders,
                 optimizer=optimizer,
                 loss_fn=loss_fn,
                 epochs=NUM_EPOCHS,
                 device=device)

    utils.save_model(model=model_0,
                     target_dir="models",
                     model_name="going_modular_script_mode_tinyvgg_model.pth")
