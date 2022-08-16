import torch

def mean_square_loss(y_gen, y_real):
    diff = torch.square(torch.sub(y_gen, y_real))

    