import numpy as np
import os
from PIL import Image
import numpy as np
import torch


class dataloader:
    def __init__(self, set_size, train_size, batch_size):
        self.set = np.arange(set_size)
        self.trainingset = []
        self.testset = []
        self.set_size = set_size

        self.batch_size = batch_size

        self.train_size = train_size
        self.trainingset = np.random.randint(set_size, size=train_size)
        self.testset = [s not in self.trainingset for s in self.set]

        self.batch_counter = 0
        self.test_counter = 0

    def start_new_epoch(self):
        self.batch_counter = 0
        np.random.shuffle(self.trainingset)

    def epoch_finished(self):
        return self.batch_counter >= int(self.train_size / self.batch_size)

    def trainloader(self):
        batch = self.trainingset[
            (self.batch_counter * self.batch_size) : (
                (self.batch_counter + 1) * self.batch_size
            )
        ]
        self.batch_counter += 1
        return self.batchloader(batch=batch)

    def testdata_loaded(self):
        if self.test_counter >= int(self.test_size / 10):
            self.test_counter = 0
            return True

        return False

    def testloader(self):
        batch = self.testset[(self.test_counter * 10) : ((self.test_counter + 1) * 10)]
        self.test_counter += 1
        return self.batchloader(batch=batch)

    def batchloader(self, batchsize=None, batch=None):

        im_directory = "data/images/"
        mask_directory = "data/masks/"
        # if batch is given iterate throught batch indices
        if batchsize == None and any(batch) != None:
            j = 0
            for i in batch:
                # open image
                f_im = im_directory + str(i) + ".jpg"
                f_mask = mask_directory + str(i) + ".jpg"
                im = Image.open(f_im)
                mask = Image.open(f_mask)
                mask = mask.convert("L")

                # extract data into numpy array
                im_data = np.array(im.getdata()).T
                mask_data = np.array(mask.getdata()).T

                # get data into right shape and concat image data to final data array
                if j == 0:

                    data_im = np.array([im_data.reshape(3, 256, 256)])
                    data_mask = np.array([mask_data.reshape(1, 256, 256)])
                    j += 1

                else:
                    im_data = im_data.reshape(1, 3, 256, 256)
                    data_im = np.concatenate((data_im, im_data))

                    mask_data = mask_data.reshape(1, 1, 256, 256)
                    data_mask = np.concatenate((data_mask, mask_data))

            return torch.tensor(data_im).float(), torch.tensor(data_mask).float()

        if batchsize != None and any(batch) == None:

            # if batch is not given generate random batch of size batchsize
            batch = np.random.randint(5108, size=batchsize)
            j = 0
            for i in batch:
                # open image
                f_im = os.path.join(im_directory, str(i) + ".jpg")
                f_mask = os.path.join(mask_directory, str(i) + ".jpg")
                im = Image.open(f_im)
                mask = Image.open(f_mask)
                mask = mask.convert("L")

                # extract data into numpy array
                im_data = np.array(im.getdata()).T
                mask_data = np.array(mask.getdata()).T

                # get data into right shape and concat image data to final data array
                if j == 0:

                    data_im = np.array([im_data.reshape(3, 256, 256)])
                    data_mask = np.array([mask_data.reshape(1, 256, 256)])
                    j += 1

                else:
                    im_data = im_data.reshape(1, 3, 256, 256)
                    data_im = np.concatenate((data_im, im_data))

                    mask_data = mask_data.reshape(1, 1, 256, 256)
                    data_mask = np.concatenate((data_mask, mask_data))

            return torch.tensor(data_im).float(), torch.tensor(data_mask).float()

        if batchsize == None and batch == None:
            # if batchsize and batch are None raise Exception
            raise Exception("Weder batchsize noch batch deklariert!")
