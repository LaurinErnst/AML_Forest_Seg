import torch
from torch.optim import Adam

# defines the batchsize for the model
BATCH_SIZE = 20

# defines the number of epochs for the model
NUM_EPOCHS = 10

# defines the loss function
LOSS_FUNC = torch.nn.MSELoss()

# defiens the model optimizer
OPT = Adam
