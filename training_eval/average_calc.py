import numpy as np
import csv
import pandas as pd
import os

directory = "results_run_2/results"


train_Unet_BCE_SGD = np.zeros(15)
train_Unet_MSE_SGD = np.zeros(15)
train_Unet_BCE_ADAM = np.zeros(15)
train_Unet_MSE_ADAM = np.zeros(15)
test_Unet_BCE_SGD = np.zeros(15)
test_Unet_MSE_SGD = np.zeros(15)
test_Unet_BCE_ADAM = np.zeros(15)
test_Unet_MSE_ADAM = np.zeros(15)

for i in range(6):
    print(i)
    dir = directory + str(i)
    with open(dir + "/csv_data/train_UNET_BCE_SGD_1_data.csv", mode="r") as infile:
        dat = []
        for row in csv.reader(infile):
            if row[0] != "0":

                dat.append(float(row[0]))
    dat = np.array(dat)
    train_Unet_BCE_SGD += dat

    with open(dir + "/csv_data/test_UNET_BCE_SGD_1_data.csv", mode="r") as infile:
        dat = []
        for row in csv.reader(infile):
            if row[0] != "0":

                dat.append(float(row[0]))
    dat = np.array(dat)
    test_Unet_BCE_SGD += dat

    with open(dir + "/csv_data/train_UNET_BCE_ADAM_1_data.csv", mode="r") as infile:
        dat = []
        for row in csv.reader(infile):
            if row[0] != "0":

                dat.append(float(row[0]))
        dat = np.array(dat)
        train_Unet_BCE_ADAM += dat

    with open(dir + "/csv_data/test_UNET_BCE_ADAM_1_data.csv", mode="r") as infile:
        dat = []
        for row in csv.reader(infile):
            if row[0] != "0":

                dat.append(float(row[0]))
        dat = np.array(dat)
        test_Unet_BCE_ADAM += dat

    with open(dir + "/csv_data/train_UNET_MSE_ADAM_1_data.csv", mode="r") as infile:
        dat = []
        for row in csv.reader(infile):
            if row[0] != "0":

                dat.append(float(row[0]))
        dat = np.array(dat)
        train_Unet_MSE_ADAM += dat

    with open(dir + "/csv_data/test_UNET_MSE_ADAM_1_data.csv", mode="r") as infile:
        dat = []
        for row in csv.reader(infile):
            if row[0] != "0":

                dat.append(float(row[0]))
        dat = np.array(dat)
        test_Unet_MSE_ADAM += dat

    with open(dir + "/csv_data/train_UNET_MSE_SGD_1_data.csv", mode="r") as infile:
        dat = []
        for row in csv.reader(infile):
            if row[0] != "0":

                dat.append(float(row[0]))
        dat = np.array(dat)
        train_Unet_MSE_SGD += dat

    with open(dir + "/csv_data/test_UNET_MSE_SGD_1_data.csv", mode="r") as infile:
        dat = []
        for row in csv.reader(infile):
            if row[0] != "0":

                dat.append(float(row[0]))
        dat = np.array(dat)
        test_Unet_MSE_SGD += dat

train_Unet_BCE_SGD = train_Unet_BCE_SGD / 6
train_Unet_MSE_SGD = train_Unet_MSE_SGD / 6
train_Unet_BCE_ADAM = train_Unet_BCE_ADAM / 6
train_Unet_MSE_ADAM = train_Unet_MSE_ADAM / 6
test_Unet_BCE_SGD = test_Unet_BCE_SGD / 6
test_Unet_MSE_SGD = test_Unet_MSE_SGD / 6
test_Unet_BCE_ADAM = test_Unet_BCE_ADAM / 6
test_Unet_MSE_ADAM = test_Unet_MSE_ADAM / 6


o = os.path.join("plots/train_UNET_BCE_SGD_1_data.csv")

# saving data in csv
data_np = np.array(train_Unet_BCE_SGD)  # convert to Numpy array
df = pd.DataFrame(data_np)  # convert to a dataframe
df.to_csv(o, index=False)  # save to file

o = os.path.join("plots/test_UNET_BCE_SGD_1_data.csv")

# saving data in csv
data_np = np.array(test_Unet_BCE_SGD)  # convert to Numpy array
df = pd.DataFrame(data_np)  # convert to a dataframe
df.to_csv(o, index=False)  # save to file

o = os.path.join("plots/train_UNET_BCE_ADAM_1_data.csv")

# saving data in csv
data_np = np.array(train_Unet_BCE_ADAM)  # convert to Numpy array
df = pd.DataFrame(data_np)  # convert to a dataframe
df.to_csv(o, index=False)  # save to file

o = os.path.join("plots/test_UNET_BCE_ADAM_1_data.csv")

# saving data in csv
data_np = np.array(test_Unet_BCE_ADAM)  # convert to Numpy array
df = pd.DataFrame(data_np)  # convert to a dataframe
df.to_csv(o, index=False)  # save to file

o = os.path.join("plots/train_UNET_MSE_ADAM_1_data.csv")

# saving data in csv
data_np = np.array(train_Unet_MSE_ADAM)  # convert to Numpy array
df = pd.DataFrame(data_np)  # convert to a dataframe
df.to_csv(o, index=False)  # save to file

o = os.path.join("plots/test_UNET_MSE_ADAM_1_data.csv")

# saving data in csv
data_np = np.array(test_Unet_MSE_ADAM)  # convert to Numpy array
df = pd.DataFrame(data_np)  # convert to a dataframe
df.to_csv(o, index=False)  # save to file

o = os.path.join("plots/train_UNET_MSE_SGD_1_data.csv")

# saving data in csv
data_np = np.array(train_Unet_MSE_SGD)  # convert to Numpy array
df = pd.DataFrame(data_np)  # convert to a dataframe
df.to_csv(o, index=False)  # save to file

o = os.path.join("plots/test_UNET_MSE_SGD_1_data.csv")

# saving data in csv
data_np = np.array(test_Unet_MSE_SGD)  # convert to Numpy array
df = pd.DataFrame(data_np)  # convert to a dataframe
df.to_csv(o, index=False)  # save to file
