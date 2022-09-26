import pickle
import torch
import io

from PIL import ImageFilter

import torchvision.transforms as t
from NaiveSegmentation.naivesegmentation import naive_seg
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
				pixels[i, j] = (212, 49, 49, 65)
	return mask


def overlap_without_model(i):
	img, mask = load_one(i)
	return create_overlap(img[0], mask[0])


def create_overlap(background, overlap):
	transform = t.ToPILImage()

	background = transform(background / 255).convert("RGBA")
	overlap = transform(overlap / 255).convert("RGBA")
	overlap = convert_mask_to_transparent(overlap)

	background.paste(overlap, (0, 0), overlap)

	return background


def create_masks(model, indices):
	images = []

	for i in indices:
		img, mask = load_one(i)
		y = model(img)

		real_overlap = create_overlap(img[0], mask[0])
		true_vs_calc = create_overlap(mask[0], y[0])
		calc_overlap = create_overlap(img[0], y[0])

		images.append((i, real_overlap, calc_overlap, true_vs_calc))

	return images


with open("training_eval/testmodel", "rb") as file:
	m = NetRecovery(file).load()

# overlap_without_model(10).save("training_eval/masks/" + str(10) + "_overlap.png")
# """
images = create_masks(m, [2828, 2912, 5099,  # worst
                          505, 2366, 3490,  # okay
                          7])  # best

for i, image in enumerate(images):
	image[1].save("training_eval/masks/" + str(i) + "_1.png")
	image[2].save("training_eval/masks/" + str(i) + "_2.png")
	image[3].save("training_eval/masks/" + str(i) + "_3.png")
# """
