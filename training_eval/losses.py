import numpy as np
from data_handling.dataloader import dataloader
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


def test_model(model, n=5):
    dl = dataloader(0, 0, 0)

    model.eval()

    losses = []

    for i in range(5108):
        x, y = dl.load_all()
        y = y / 255

        if i % 100 == 0:
            print(i/5108)
        y_hat = model(x)

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
print(test_model(m))
