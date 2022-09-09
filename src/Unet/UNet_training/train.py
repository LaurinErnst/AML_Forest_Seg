import time
import torch
from tqdm import tqdm
from data_handling import dataloader
from Unet import UNET as un
from config import gen_config
from config import UNET_MSE_SGD_1 as model_con
from data_handling import net_saver
from data_handling import graph_saver
import gc

Model = un.UNet(retain_dim=True)

Model = Model.to(gen_config.DEVICE)
lossFunc = model_con.LOSS_FUNC(**model_con.LOSS_PARAMS)

opt = model_con.OPT(Model.parameters(), **model_con.OPT_PARAMS)
print(sum(p.numel() for p in Model.parameters() if p.requires_grad))
# calculate steps per epoch for training and test set
trainSteps = gen_config.TRAIN_SIZE // model_con.BATCH_SIZE
testSteps = (gen_config.SET_SIZE - gen_config.TRAIN_SIZE) // model_con.BATCH_SIZE
# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

# ___________________loader setup_______________


loader = dataloader.dataloader(
    gen_config.SET_SIZE,
    gen_config.TRAIN_SIZE,
    model_con.BATCH_SIZE,
    gen_config.TEST_BATCH_SiZE,
)

# ___________________________________________

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(model_con.NUM_EPOCHS)):
    Model.train()
    gc.collect()
    loader.start_new_epoch()
    # set the model in training mode
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalTestLoss = 0

    # loop over the training set
    while not loader.epoch_finished():
        x, y = loader.trainloader()
        x, y = x.to(gen_config.DEVICE), y.to(gen_config.DEVICE)
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
        print("[INFO] Loss of current batch: {}".format(loss.item()))

        x = x.cpu()
        y = y.cpu()

        gc.collect()

    # switch off autograd
    with torch.no_grad():
        # set the model in evaluation mode
        Model.eval()
        # loop over the validation set
        while not loader.testdata_loaded():
            x, y = loader.testloader()
            # send the input to the device
            x, y = x.to(gen_config.DEVICE), y.to(gen_config.DEVICE)
            # make the predictions and calculate the validation loss
            pred = Model.forward(x)
            totalTestLoss += lossFunc(pred, y)
            print(lossFunc(pred, y))

            x = x.cpu()
            y = y.cpu()

            gc.collect()
    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgTestLoss = totalTestLoss / testSteps
    # update our training history
    # H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    # H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, model_con.NUM_EPOCHS))
    print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

net_saver.save_model(Model, model_con.NAME)

graph_saver.graph_saver(
    H["train_loss"], model_con.NAME, title="Training Loss per Epoch"
)

graph_saver.graph_saver(H["test_loss"], model_con.NAME, title="Test Loss per Epoch")
