import matplotlib.pyplot as plt


def graph_saver(data, name, title="Loss", xlabel="Epochs", ylabel="Loss"):
    plt.plot(data, name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig("results/loss_graphs/" + name + "png")
