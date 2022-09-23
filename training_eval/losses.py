import gc

import numpy as np
from data_handling.dataloader import load_one
import pickle
import torch
import io


def percentage_right(x, y):
	x = x.flatten()
	y = y.flatten()

	x = torch.round(x)
	y = torch.round(y)

	print(x)
	print(y)
	print(np.average(x == y))
	return np.average(x == y)


def jaccard_index(mask_true, mask_pred):
	mask_true = torch.round(mask_true[0][0])
	mask_pred = torch.round(mask_pred[0][0])

	intersection = mask_pred * mask_true
	union = mask_true + mask_pred > 0

	return torch.sum(intersection) / torch.sum(union)


def test_model(model, n=5, jaccard=False):
	losses = []

	for i in range(5108):
		x, y = load_one(i)
		y = y / 255

		gc.collect()

		if i % 100 == 0:
			print(i / 5108)
		with torch.no_grad():
			y_hat = torch.sigmoid(model(x))

		if jaccard:
			losses.append(jaccard_index(y, y_hat))
		else:
			losses.append(percentage_right(y, y_hat))

	sorted_indices = np.argsort(losses)
	worst = sorted_indices[:n]
	best = np.flip(sorted_indices[-n:])
	return losses, np.average(losses), worst, best


class NetRecovery(pickle.Unpickler):
	def find_class(self, module, name):
		if module == 'torch.storage' and name == '_load_from_bytes':
			return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
		else:
			return super().find_class(module, name)


with open("testmodel", "rb") as file:
	m = NetRecovery(file).load()

print(test_model(m, jaccard=True))

