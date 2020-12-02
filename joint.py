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
from global import *
from local import *
import torch.nn.functional as F

class joint(nn.Module):
	def __init__(self):
		super(joint,self).__init__()
	
	def forward(self,x):
		glbl = GlobalNetwork()

		g1, A1,output1 = glbl.layer1(x)
		g2, A2,output2 = glbl.layer2(A1)
        g3, A3,output3 = glbl.layer3(A2)

		lcll = LocalNetwork(output1,output2,output3)

		gbl = glbl.forward(x)
		lcl = lcll.forward()

		j = torch.cat((gbl,lcl),axis=1)

		return j

		

		
