from inspect import formatannotationrelativeto
import matplotlib.pyplot as plt
import pickle
import os
import torch
import numpy as np
import pandas as pd
import csv

# assign directo
directory = input("Directory containing results: ")

convert = ""
while convert != "y" and convert != "n":
    convert = input("convert pickles to csv?(needed if not done already) (y/n): ")

# convert pickles to csv
if convert == "y":

    # iterate over files in
    # that directory
    for filename in os.listdir(directory + "/loss_graphs"):
        f = os.path.join(directory + "/loss_graphs", filename)
        # checking if it is a file
        if os.path.isfile(f):
            with open(f, "rb") as file:
                data = pickle.load(file)
            if not os.path.exists(directory + "/csv_data"):
                os.mkdir(directory + "/csv_data")
            o = os.path.join(directory + "/csv_data", filename + "_data.csv")

            # saving data in csv
            data_np = np.array(data)  # convert to Numpy array
            df = pd.DataFrame(data_np)  # convert to a dataframe
            df.to_csv(o, index=False)  # save to file

if not os.path.exists(directory + "/graphs"):
    os.mkdir(directory + "/graphs")

file_names = pd.DataFrame
file_names_file = input("path/to/file_name_file: ")

with open("training_eval/" + file_names_file, mode="r") as infile:
    file_names = csv.reader(infile)
    i = 0
    data = np.zeros([2, 15])
    for file_name in file_names:
        print("Name of model: {}".format(file_name))

        with open(directory + "/csv_data/" + file_name[0], mode="r") as infile:
            dat = []
            for row in csv.reader(infile):
                if row[0] != "0":

                    dat.append(float(row[0]))
            data[i] = dat
        if i == 1:
            plot = input("Plot data? (y/n): ")
            if plot == "y":
                # Initialise the figure and axes.
                fig, ax = plt.subplots(1, figsize=(8, 6))

                # Set the title for the figure
                fig.suptitle(input("title of plot: "), fontsize=15)
                plt.xlabel(input("x_label: "))
                plt.ylabel(input("y_label: "))
                np.flip(data, axis=1)
                # Draw all the lines in the same plot, assigning a label for each one to be
                # shown in the legend.
                ax.plot(range(1, 16), data[0], label="SGD")
                ax.plot(range(1, 16), data[1], label="ADAM")
                ylim = float(input("y upper limit: "))
                plt.ylim(min([min(data[0]), min(data[1])]), ylim)
                # Add a legend, and position it on the lower right (with no box)
                plt.legend(loc="upper right", frameon=False)
                name = input("name of file: ")
                fig.savefig(directory + "/graphs/" + name + ".png")
                plt.show(block=False)

                # save = ""

                # while save != "y" and save != "n":
                #     save = input("save plot? (y/n) ")

                # if save == "y":
                #     name = input("name of file: ")

            i = 0
        elif i == 0:
            i += 1
