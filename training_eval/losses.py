import gc

import numpy as np
from data_handling.dataloader import load_one
import pickle
import torch
import io


class NetRecovery(pickle.Unpickler):
	def find_class(self, module, name):
		if module == 'torch.storage' and name == '_load_from_bytes':
			return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
		else:
			return super().find_class(module, name)


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
	okay = sorted_indices[(int(len(sorted_indices) / 2) - int(n / 2)):(int(len(sorted_indices) / 2) + int(n / 2))]
	return losses, np.average(losses), worst, okay, best


netnames = ["UNET_BCE_ADAM_1", "UNET_BCE_SGD_1", "UNET_MSE_ADAM_1", "UNET_MSE_SGD_1"]
for net in netnames:
	with open("results_run_2/results5/trained_models/" + net, "rb") as file:
		m = NetRecovery(file).load()

	data = test_model(m, jaccard=True)

	print(net)
	print(data[1:])

	with open("training_eval/data/" + net + ".pt", "wb") as file:
		pickle.dump(data, file)

with open("satresult\results1\results\trained_models" + "UNET_MSE_ADAM_1", "rb") as file:
	m = NetRecovery(file).load()

	data = test_model(m, jaccard=True)

	print(data[1:])