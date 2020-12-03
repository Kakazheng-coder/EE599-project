import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class GlobalNetwork(nn.Module):
    def __init__(self):
        super(GlobalNetwork, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(nn.Conv2d(3, 5, kernel_size=11,stride=2), nn.ReLU(inplace=True), 
                                  nn.BatchNorm2d(5), nn.Conv2d(5, 10, kernel_size=7,stride=1,padding=2), 
                                  nn.ReLU(inplace=True), nn.BatchNorm2d(10), 
                                  nn.Conv2d(10,16 , kernel_size=11,stride=1,padding=1), nn.ReLU(inplace=True), 
                                  nn.BatchNorm2d(16))
        self.se1 = SE_Block(16,16)

        self.cnn2 = nn.Sequential(nn.Conv2d(16, 20, kernel_size=3,stride=2), nn.ReLU(inplace=True), 
                                  nn.BatchNorm2d(20))
        self.se2 = SE_Block(20,16)

        self.cnn3 = nn.Sequential(nn.Conv2d(20, 25, kernel_size=4,stride=2), nn.ReLU(inplace=True), 
                                  nn.BatchNorm2d(25))
        self.se3 = SE_Block(25,16)

        self.fc = nn.Sequential(nn.Linear(113*113+56*56+27*27,1024), nn.ReLU(inplace = True), 
                                nn.Dropout2d(p=0.5), nn.Linear(1024,101))
        
        self.c1 = nn.Conv2d(1,16,kernel_size=1,stride=1)
        self.c2 = nn.Conv2d(1,20,kernel_size=1,stride=1)
        self.c3 = nn.Conv2d(1,25,kernel_size=1,stride=1)
        
    """
    """
          # Defining the fully connected layers
        #self.fc1 = nn.Sequential(nn.Linear(25*25*25, 1024), nn.ReLU(inplace=True), nn.Dropout2d(p=0.5), nn.Linear(1024, 128), nn.ReLU(inplace=True), nn.Linear(128,2))
    def layer1(self,x):
        output1 = self.cnn1(x)
        g1 = torch.mean(output1,1,True)
        c1 = self.se1(output1)
        A1 = self.c1(g1)
        return g1,A1,output1

    def layer2(self,A1):
        output2 = self.cnn2(A1)
        g2 = torch.mean(output2,1,True)
        c2 = self.se2(output2)
        A2 = self.c2(g2)
        return g2,A2, output2

    def layer3(self,A2):
        output3 = self.cnn3(A2)
        g3 = torch.mean(output3,1,True)
        c3 = self.se3(output3)
        A3 = self.c3(g3)
        return g3,A3, output3

    def forward(self, x):

        g1, A1,_ = self.layer1(x)
        g2, A2,_ = self.layer2(A1)
        g3, A3,_ = self.layer3(A2)

        g1 = g1.view(-1,113*113*1)
        print(g1.size())
        g2 = g2.view(-1,56*56*1)
        g3 = g3.view(-1,27*27*1)

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



