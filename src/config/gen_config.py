# import the necessary packages
import torch


# define
SET_SIZE = 300
# define the trainings set size
TRAIN_SIZE = 250

# setting test batch size for dataloader
TEST_BATCH_SiZE = 50


# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device: " + DEVICE)
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False
