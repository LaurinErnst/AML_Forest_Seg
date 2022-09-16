import matplotlib.pyplot as plt
import pickle


def graph_saver(data, name, title="Loss", xlabel="Epochs", ylabel="Loss"):
    # plt.figure()
    # plt.plot(data)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.title(title)
    # plt.savefig("results/loss_graphs/" + name + ".png")
    with open("results/loss_graphs/" + name, "wb") as file:
        pickle.dump(data, file)
