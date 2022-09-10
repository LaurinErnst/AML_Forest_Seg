import torch
from torch.optim import SGD


NAME = "UNET_BCE_SGD_1"
# defines the batchsize for the model
BATCH_SIZE = 5

# defines the number of epochs for the model
NUM_EPOCHS = 5

# defines the loss function
LOSS_FUNC = torch.nn.BCEWithLogitsLoss

# defines Parameters for the loss
LOSS_PARAMS = {"reduction": "mean", "pos_weight": None, "weight": None}

# defiens the model optimizer
OPT = SGD


OPT_PARAMS = {"lr": 0.0001, "weight_decay": 0}
