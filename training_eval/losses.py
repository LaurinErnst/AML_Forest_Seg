import numpy as np
from data_handling.dataloader import  *


def percentage_right(x, y):
	x = np.flatten(x)
	y = np.flatten(y)

	x = np.round(x)
	y = np.round(y)

	return np.average(x == y)


def test_model(model, n=5):
	dl = dataloader.dataloader(0, 0, 0)
	X, Y = dl.load_all()

	losses = []

	for i, x in enumerate(X):
		y = Y[i]
		y_hat = model(x)

		losses.append(percentage_right(y, y_hat))

	sorted_indices = np.argsort(losses)
	worst = sorted_indices[:n]
	best = np.flip(sorted_indices[-n:])
	return losses, np.average(losses), worst, best

