import time
from data_handling import dataloader
from Unet import UNET as un
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import tqdm

Model = un.Model(retain_dim=True)

lossFunc = torch.nn.MSELoss()
opt = Adam(Model.parameters())

# loop over epochs
print("[INFO] training the network...")
startTime =  time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
	# set the model in training mode
	# initialize the total training and validation loss
    totalTrainLoss = 0
    totalTestLoss = 0

	# loop over the training set
    for i in range(train_steps):
        x, y = loader.batchloader(10)
		# perform a forward pass and calculate the training loss
        pred = Model(x)
        loss = lossFunc(pred, y)
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
		# add the loss to the total training loss so far
        totalTrainLoss += loss
	# switch off autograd
    with torch.no_grad():
		# set the model in evaluation mode
        Model.eval()
		# loop over the validation set
        for (x, y) in testLoader:
			# send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
			# make the predictions and calculate the validation loss
            pred = Model(x)
            totalTestLoss += lossFunc(pred, y)
	# calculate the average training and validation loss    
    avgTrainLoss = totalTrainLoss / trainSteps
    avgTestLoss = totalTestLoss / testSteps
	# update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
	# print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
    print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

