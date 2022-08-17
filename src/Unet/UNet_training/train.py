from data_handling import batchloader as loader
from Unet import UNET as un
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt

Model = un.UNet(retain_dim=True)

lossFunc = torch.nn.MSELoss()
opt = Adam(Model.parameters())

print("Setup fertig")

train_steps = 10
test_steps = 0

# initialize a dictionary to store training historys
H = {"train_loss": [], "test_loss": []}

for e in range(10):
    # set the model in training mode
    Model.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalTestLoss = 0

    # loop over the training set
    for i in range(train_steps):

        x, y = loader.batchloader(10)

        # perform a forward pass and calculate the training loss
        pred = Model.forward(x)
        loss = lossFunc(pred, y)
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far
        totalTrainLoss += loss

    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / train_steps
    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, 10))
    print("Train loss: {:.6f}".format(
    avgTrainLoss))

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.show()

