import torch


# VERALTET WIR NUTZEN ZUR ZEIT PYTORCH LOSS NOCH DA FALL WIR CUSTOM LOSS WOLLEN
def mean_square_loss(y_gen, y_real):
    diff = torch.square(torch.sub(y_gen, y_real))
