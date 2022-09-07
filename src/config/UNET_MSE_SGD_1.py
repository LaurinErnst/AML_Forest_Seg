import torch
from torch.optim import SGD


NAME = "UNET_MSE_SGD_1"
# defines the batchsize for the model
BATCH_SIZE = 250

# defines the number of epochs for the model
NUM_EPOCHS = 100

# defines the loss function
LOSS_FUNC = torch.nn.MSELoss

# defines Parameters for the loss
LOSS_PARAMS = {"reduction": "mean"}

# defiens the model optimizer
OPT = SGD


OPT_PARAMS = {"lr": 0.001, "weight_decay": 1e-3}
