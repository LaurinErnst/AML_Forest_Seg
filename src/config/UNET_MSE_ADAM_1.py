import torch
from torch.optim import Adam


NAME = "UNET_MSE_ADAM_1"
# defines the batchsize for the model
BATCH_SIZE = 20

# defines the number of epochs for the model
NUM_EPOCHS = 10

# defines the loss function
LOSS_FUNC = torch.nn.MSELoss

# defines Parameters for the loss
LOSS_PARAMS = {"reduction": "mean"}

# defiens the model optimizer
OPT = Adam


OPT_PARAMS = {"lr": 0.1, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-3}
