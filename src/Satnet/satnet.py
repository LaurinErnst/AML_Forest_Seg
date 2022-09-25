import torch
import torch.nn as nn


class SatBlock(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, stride, padding='valid'):
		super().__init__()
		self.conv = nn.Conv2d(out_channels=out_channel, in_channels=in_channel,
		                      kernel_size=kernel_size, stride=stride, padding=padding)
		self.batchnorm = nn.BatchNorm2d(out_channel)
		self.relu = nn.ReLU()

	def forward(self, x):
		return self.relu(self.batchnorm(self.conv(x)))


class AdditionBlock(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size):
		super().__init__()
		self.block = SatBlock(out_channel=out_channel, in_channel=in_channel,
		                      kernel_size=kernel_size, stride=1, padding='same')
		self.conv2 = nn.Conv2d(out_channels=out_channel, in_channels=out_channel,
		                       kernel_size=kernel_size, stride=1, padding='same')
		self.batchnorm2 = nn.BatchNorm2d(out_channel)
		self.relu2 = nn.ReLU()

	def forward(self, x):
		return self.relu2(self.batchnorm2(self.conv2(self.block(x))) + x)


class DoubleAdditionBlock(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size1, kernel_size2,
	             kernel_size3, stride):
		super().__init__()
		self.block1 = SatBlock(in_channel, out_channel, kernel_size1, stride)
		self.block2 = SatBlock(in_channel, out_channel, kernel_size2,
		                       stride, padding=(3, 3))
		self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size3, 1, padding='same')
		self.batchnorm3 = nn.BatchNorm2d(out_channel)
		self.relu3 = nn.ReLU()

	def forward(self, x):
		forward1 = self.block1(x)
		forward2 = self.batchnorm3(self.conv3(self.block2(x)))
		return self.relu3(forward1 + forward2)


class SatEncoder(nn.Module):
	def __init__(self, p_dropout=0.5):
		super().__init__()
		self.blocks = nn.ModuleList(
			[SatBlock(3, 32, 7, 2), AdditionBlock(32, 32, 3), AdditionBlock(32, 32, 3), AdditionBlock(32, 32, 3),
			 DoubleAdditionBlock(32, 64, 1, 7, 3, 2), AdditionBlock(64, 64, 3), AdditionBlock(64, 64, 3),
			 AdditionBlock(64, 64, 3), nn.Dropout(p_dropout)]
		)

	def forward(self, x):
		for block in self.blocks:
			x = block(x)
		print(x.shape)
		return x


class SatDecoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.blocks = nn.ModuleList(
			[nn.ConvTranspose2d(64, 16, 16, stride=2, padding=(7, 7)), nn.BatchNorm2d(16), nn.ReLU(),
			 nn.ConvTranspose2d(16, 1, 16, stride=2, padding=(5, 5)), nn.Sigmoid()]
		)

	def forward(self, x):
		for block in self.blocks:
			x = block(x)
		return x


class SatNet(nn.Module):
	def __init__(self, p_drop=0.5):
		super().__init__()
		self.encoder = SatEncoder(p_drop)
		self.decoder = SatDecoder()

	def forward(self, x):
		return self.decoder(self.encoder(x))
