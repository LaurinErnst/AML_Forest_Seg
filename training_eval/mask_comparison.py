import pickle
import torch
import io

import torchvision.transforms as t
from data_handling.dataloader import load_one


class NetRecovery(pickle.Unpickler):
	def find_class(self, module, name):
		if module == 'torch.storage' and name == '_load_from_bytes':
			return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
		else:
			return super().find_class(module, name)


def convert_mask_to_transparent(mask):
	pixels = mask.load()
	for i in range(mask.size[0]):
		for j in range(mask.size[1]):
			if pixels[i, j][0] == 0:
				pixels[i, j] = (0, 0, 0, 0)
			else:
				pixels[i, j] = (212, 49, 49, 80)
	return mask


def create_overlap(background, overlap):
	transform = t.ToPILImage()

	background = transform(background / 255).convert("RGBA")
	overlap = transform(overlap / 255).convert("RGBA")
	overlap = convert_mask_to_transparent(overlap)

	background.paste(overlap, (0, 0), overlap)
	background.show()

	return background


def create_masks(model, indices):
	images = []

	for i in indices:
		img, mask = load_one(i)
		y = model(img)

		real_overlap = create_overlap(img[0], mask[0])
		calc_overlap = create_overlap(img[0], y[0])

		images.append((i, real_overlap, calc_overlap))

	return images


with open("testmodel", "rb") as file:
	m = NetRecovery(file).load()
create_masks(m, [1, 2, 3, 4])


