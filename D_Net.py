import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
add_path('./pytorch-faster-rcnn/lib')
from model.roi_pooling.modules.roi_pool import _RoIPooling

# class DNet(nn.Module):
# 	def __init__(self):
# 		super(DNet, self).__init__()
# 		use_bias = False

# 		# assume input resized to 32x32
# 		conv1 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)
# 		norm1 = nn.BatchNorm2d(1024)
# 		relu1 = nn.LeakyReLU(0.2, True)

# 		#16x16
# 		conv2 = nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=use_bias)
# 		norm2 = nn.BatchNorm2d(2048)
# 		relu2 = nn.LeakyReLU(0.2, True)

# 		#8x8
# 		conv3 = nn.Conv2d(2048, 4096, kernel_size=4, stride=2, padding=1, bias=use_bias)
# 		norm3 = nn.BatchNorm2d(4096)
# 		relu3 = nn.LeakyReLU(0.2, True)

# 		#4x4
# 		conv4 = nn.Conv2d(4096, 1, kernel_size=4, stride=1, padding=0, bias=use_bias)
# 		sig = nn.Sigmoid()
# 		self.discriminator = nn.Sequential(conv1, norm1, relu1, conv2, norm2, relu2,
# 											 conv3, norm3, relu3, conv4, sig)
		
# 	def forward(self, x):
# 		return self.discriminator(x)
		
class DNet_pooling(nn.Module):
	def __init__(self):
		super(DNet_pooling, self).__init__()
		use_bias = False
		POOLING_SIZE = 32
		
		# resize input to 32x32 using roipooling
		self.rois = torch.from_numpy(np.array([0.0, 0.0, 0.0, 32.0, 32.0], np.float32))
		self.RCNN_roi_pool = _RoIPooling(POOLING_SIZE, POOLING_SIZE, 1.0)
		
		# assume input resized to 32x32
		conv1 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)
		norm1 = nn.BatchNorm2d(1024)
		relu1 = nn.LeakyReLU(0.2, True)

		#16x16
		conv2 = nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=use_bias)
		norm2 = nn.BatchNorm2d(2048)
		relu2 = nn.LeakyReLU(0.2, True)

		#8x8
		conv3 = nn.Conv2d(2048, 4096, kernel_size=4, stride=2, padding=1, bias=use_bias)
		norm3 = nn.BatchNorm2d(4096)
		relu3 = nn.LeakyReLU(0.2, True)

		#4x4
		conv4 = nn.Conv2d(4096, 1, kernel_size=4, stride=1, padding=0, bias=use_bias)
		sig = nn.Sigmoid()
		self.discriminator = nn.Sequential(conv1, norm1, relu1, conv2, norm2, relu2,
											 conv3, norm3, relu3, conv4, sig)
		
	def forward(self, x):
		
		rois = Variable(self.rois).cuda()
		pooled_feat = self.RCNN_roi_pool(x, rois.view(-1, 5))
		
		x = self.discriminator(pooled_feat)
		return x

class DNet_pooling_shallow(nn.Module):
	def __init__(self):
		super(DNet_pooling_shallow, self).__init__()
		use_bias = False
		POOLING_SIZE = 16
		
		self.rois = torch.from_numpy(np.array([0.0, 0.0, 0.0, POOLING_SIZE, POOLING_SIZE], np.float32))
		self.RCNN_roi_pool = _RoIPooling(POOLING_SIZE, POOLING_SIZE, 1.0)
		
		# assume input resized to 32x32
		conv1 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)
		norm1 = nn.BatchNorm2d(1024)
		relu1 = nn.LeakyReLU(0.2, True)

		# #16x16
		# conv2 = nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=use_bias)
		# norm2 = nn.BatchNorm2d(2048)
		# relu2 = nn.LeakyReLU(0.2, True)

		# #8x8
		# conv3 = nn.Conv2d(2048, 4096, kernel_size=4, stride=2, padding=1, bias=use_bias)
		# norm3 = nn.BatchNorm2d(4096)
		# relu3 = nn.LeakyReLU(0.2, True)

		#4x4
		conv2 = nn.Conv2d(1024, 1, kernel_size=8, stride=1, padding=0, bias=use_bias)
		sig = nn.Sigmoid()
		self.discriminator = nn.Sequential(conv1, norm1, relu1, conv2, sig)
		
	def forward(self, x):
		
		rois = Variable(self.rois).cuda()
		pooled_feat = self.RCNN_roi_pool(x, rois.view(-1, 5))
		
		x = self.discriminator(pooled_feat)
		return x

def build_DNet(is_shallow=False):
	if is_shallow:
		model = DNet_pooling_shallow()
	else:
		model = DNet_pooling()
	return model