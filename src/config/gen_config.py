# import the necessary packages
import torch


# define
SET_SIZE = 5108
# define the trainings set size
TRAIN_SIZE = 10

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device: " + DEVICE)
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False
