import numpy as np


def percentage_loss(x, y):
	x = np.flatten(x)
	y = np.flatten(y)

	x = np.round(x)
	y = np.round(y)

	return np.average(x == y)
