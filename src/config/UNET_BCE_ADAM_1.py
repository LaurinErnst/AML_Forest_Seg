import torch
from torch.optim import Adam


NAME = "UNET_BCE_ADAM_1"
# defines the batchsize for the model
BATCH_SIZE = 1

# defines the number of epochs for the model
NUM_EPOCHS = 5

# defines the loss function
LOSS_FUNC = torch.nn.BCEWithLogitsLoss

# defines Parameters for the loss
LOSS_PARAMS = {"reduction": "mean", "pos_weight": None, "weight": None}

# defiens the model optimizer
OPT = Adam


OPT_PARAMS = {"lr": 0.0001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0}
