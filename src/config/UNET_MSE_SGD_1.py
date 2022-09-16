import torch
from torch.optim import SGD


NAME = "UNET_MSE_SGD_1"
# defines the batchsize for the model
BATCH_SIZE = 1

# defines the number of epochs for the model
NUM_EPOCHS = 5

# defines the loss function
LOSS_FUNC = torch.nn.MSELoss

# defines Parameters for the loss
LOSS_PARAMS = {"reduction": "mean"}

# defiens the model optimizer
OPT = SGD


OPT_PARAMS = {"lr": 1e-4, "weight_decay": 0}
