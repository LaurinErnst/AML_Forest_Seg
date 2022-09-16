import torch
import torch.nn as nn


class SatBlock(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, stride):
		super().__init__()
		self.conv = nn.Conv2d(out_channels=out_channel, in_channels=in_channel,
		                      kernel_size=kernel_size, stride=stride)
		self.batchnorm = nn.BatchNorm2d(out_channel)
		self.relu = nn.ReLU()

	def forward(self, x):
		return self.relu(self.batchnorm(self.conv(x)))


class AdditionBlock(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, stride):
		super().__init__()
		self.block = SatBlock(out_channel=out_channel, in_channel=in_channel,
		                   kernel_size=kernel_size, stride=stride)
		self.conv2 = nn.Conv2d(out_channels=out_channel, in_channels=out_channel,
		                       kernel_size=kernel_size, stride=stride)
		self.batchnorm2 = nn.BatchNorm2d(out_channel)
		self.relu2 = nn.ReLU()

	def forward(self, x):
		return self.relu2(self.batchnorm2(self.conv2(self.block(x))) + x)


class DoubleAdditionBlock(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size1, kernel_size2,
	             kernel_size3, stride):
		super().__init__()
		self.block1 = SatBlock(in_channel, out_channel, kernel_size1, stride)
		self.block2 = SatBlock(in_channel, out_channel, kernel_size2, stride)
		self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size3, 1)
		self.batchnorm3 = nn.BatchNorm2d(out_channel)
		self.relu3 = nn.ReLU()

	def forward(self, x):
		forward1 = self.block1(x)
		forward2 = self.batchnorm3(self.conv3(self.block2(x)))
		return self.relu3(forward1 + forward2)


class SatEncoder(nn.Module):
	def __init__(self, p_dropout=0.5):
		super().__init__()
		blocks = []

		blocks.append(SatBlock(3, 32, 7, 2))

		blocks.append(AdditionBlock(32, 32, 3, 1))
		blocks.append(AdditionBlock(32, 32, 3, 1))
		blocks.append(AdditionBlock(32, 32, 3, 1))

		blocks.append(DoubleAdditionBlock(32, 64, 1, 7, 3, 2))

		blocks.append(AdditionBlock(32, 32, 3, 1))
		blocks.append(AdditionBlock(32, 32, 3, 1))

		blocks.append(nn.Dropout(p_dropout))

		self.blocks = nn.ModuleList(blocks)

	def forward(self, x):
		for block in self.blocks:
			x = block(x)
		return x


class SatDecoder(nn.Module):
	def __init__(self):
		super().__init__()

		blocks = []

		blocks.append(nn.ConvTranspose2d(64, 16, 16, stride=2))
		blocks.append(nn.BatchNorm2d(16))
		blocks.append(nn.ReLU())

		blocks.append(nn.ConvTranspose2d(16, 1, 16, stride=2))
		blocks.append(nn.Sigmoid())

		self.blocks = nn.ModuleList(blocks)

	def forward(self, x):
		for block in self.blocks:
			x = block(x)
		return x


class SatNet(nn.Module):
	def __int__(self, p_drop=0.5):
		super().__init__()
		self.encoder = SatEncoder(p_drop)
		self.decoder = SatDecoder()

	def forward(self, x):
		return self.decoder(self.encoder(x))

