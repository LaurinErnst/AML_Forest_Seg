import matplotlib.pyplot as plt
import numpy as np

with open("sat_bce_adam.csv",) as file:
	bce_adam = np.loadtxt(file)

with open("sat_mse_adam.csv",) as file:
	mse_adam = np.loadtxt(file)

with open("sat_mse_sgd.csv",) as file:
	mse_sgd = np.loadtxt(file)

fig, ax = plt.subplots(1, 3)

plt.xlabel("epochs")
plt.ylabel("loss")

ax[0].plot(bce_adam)
ax[0].set_title("BCE ADAM")
ax[1].plot(mse_sgd)
ax[1].set_title("MSE SGD")
ax[2].plot(mse_adam)
ax[2].set_title("MSE ADAM")

plt.tight_layout()
plt.savefig("fig.png")
