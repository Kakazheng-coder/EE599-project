import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import torchvision.models as models



class LocalNetwork(nn.Module):
    def __init__(self,x1,x2,x3):
        super(GlobalNetwork, self).__init__(x1,x2,x3)
        # Setting up the Sequential of CNN Layers
        self.x1 = x1
        self.x2 = x2
		self.x3 = x3
        
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
        
		
		def forward(self):
			S1 = ST(self.x1)
			I1 = self.model(self.x1*S1)
			l1 = torch.mean(I1,1,True)
			M1 = F.MaxPool2d(l1,3,stride=2)
			
			S2 = ST(self.x2)
			I2 = self.model(self.x2*S2)
			l2 = torch.mean(I2,1,True)
			M2 = F.MaxPool2d(l2,2,stride=2)
			
			S3 = ST(self.x3)
			I3 = self.model(self.x3*S3)
			l3 = torch.mean(I3,1,True)
			M3 = F.MaxPool2d(l3,2,stride=2)
			
			M1 = M1.view(-1,60*60*15)
			M2 = M2.view(-1,30*30*20)
			M3 = M3.view(-1,15*15*25)
			
			lcl = torch.cat((M1,M2,M3),1)
			return lcl
			
class ST(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        	
