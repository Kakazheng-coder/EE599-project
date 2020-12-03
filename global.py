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

class GlobalNetwork(nn.Module):
    def __init__(self):
        super(GlobalNetwork, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(nn.Conv2d(3, 5, kernel_size=11,stride=2), nn.ReLU(inplace=True), nn.BatchNorm2D(5), nn.Conv2d(5, 10, kernel_size=5,stride=1,padding=2), nn.ReLU(inplace=True), nn.BathNorm2D(10) nn.Conv2d(10,15 , kernel_size=5,stride=1,padding=1), nn.ReLU(inplace=True), nn.BathNorm2D(15))
        self.se1 = SE_Block(15,16)
        
        self.cnn2 = nn.Sequential(nn.Conv2d(15, 20, kernel_size=11,stride=2), nn.ReLU(inplace=True), nn.BatchNorm2D(20))
        self.se2 = SE_Block(20,16)
        
        self.cnn3 = nn.Sequential(nn.Conv2d(20, 25, kernel_size=10,stride=2,padding=1), nn.ReLU(inplace=True), nn.BatchNorm2D(25))
        self.se3 = SE_Block(25,16)
		
		self.fc = nn.sequential(nn.Linear(121*121*15*56*56*20*25*25*25,1024), nn.ReLU(inplace = True), nn.Dropout2d(p=0.5), nn.Linear(1024,101))
"""
nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2), nn.Dropout2d(p=0.3), nn.Conv2d(256, 384, kernel_size=3,stride=1,padding=1), nn.ReLU(inplace=True),nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2), nn.MaxPool2d(3, stride = 2), nn.Dropout2d(p=0.3)) 
"""
          # Defining the fully connected layers
        #self.fc1 = nn.Sequential(nn.Linear(25*25*25, 1024), nn.ReLU(inplace=True), nn.Dropout2d(p=0.5), nn.Linear(1024, 128), nn.ReLU(inplace=True), nn.Linear(128,2))

	def layer1(self,x):
		
		output1 = self.cnn1(x)
        g1 = torch.mean(output1,1,True)
        c1 = self.se1(output1)
        A1 = F.Conv2D(g1,g1.size()[1],c1.size()[1],kernel_size = c1.size()[2],stride=1)
		return g1,A1,output1

	def layer2(self,A1):
  		output2 = self.cnn2(A1)
        g2 = torch.mean(output2,1,True)
        c2 = self.se2(output2)
        A2 = F.Conv2D(g2,g2.size()[1],c2.size()[1],kernel_size = c2.size()[2],stride=1)
		return g2,A2, output2

	def layer3(self,A2):
		output3 = self.cnn3(x)
        g3 = torch.mean(output3,1,True)
        c3 = self.se3(output3)
        A3 = F.Conv2D(g3,g3.size()[1],c3.size()[1],kernel_size = c3.size()[2],stride=1)
		return g3,A3, output3
	
	def forward(self, x):

        g1, A1,_ = self.layer1(x)
		g2, A2,_ = self.layer2(A1)
        g3, A3,_ = self.layer3(A2)
        
        g1 = g1.view(-1,121*121*15)
        g2 = g2.view(-1,56*56*20)
        g3 = g3.view(-1,25*25*25)
        
        concat = torch.cat((g1,g2,g3),axis=1)
        #The other parameter is num_classes. I am taking it as 100
        gbl = self.fc(concat)
        
	return gbl
        

class SE_Block(nn.Module):
	
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)



