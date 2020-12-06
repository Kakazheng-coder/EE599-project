
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from SENET import senet154
             
class Spatial(nn.Module):
  def __init__(self):
    super(Spatial, self).__init__()
    self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
    self.bn1 = nn.BatchNorm2d(1)
    self.conv2 = nn.Conv2d(1, 1, kernel_size=1)
    self.bn2 = nn.BatchNorm2d(1)
  def forward(self, x):
    # global cross-channel averaging
    x = torch.mean(x,1, keepdim=True) # 由hwc 变为 hw1
    # 3-by-3 conv
    h = x.size(2)
    x = F.relu(self.bn1(self.conv1(x)))
    # bilinear resizing
    x = F.upsample(x, (h,h), mode='bilinear', align_corners=True)
    # scaling conv
    x = F.relu(self.bn2(self.conv2(x)))
    return x
						 
class Channel(nn.Module):
    
  def __init__(self, c, r=16):
    super(Channel, self).__init__()
    self.conv1 = nn.Conv2d(c, c // r, 1)
    self.bn1 = nn.BatchNorm2d(c // r)
    self.conv2 = nn.Conv2d(c // r, c, 1)
    self.bn2 = nn.BatchNorm2d(c)


  def forward(self, x):
    # squeeze operation (global average pooling)
    x = F.avg_pool2d(x, x.size()[2:]) #输出是1*1*c
    # excitation operation (2 conv layers)
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    return x 

class SCA(nn.Module):
  def __init__(self,c):
    super(SCA,self).__init__()
    self.S = Spatial()
    self.C = Channel(c,16)
    self.conv = nn.Conv2d(c,c, kernel_size=1)
    self.bn1 = nn.BatchNorm2d(c)
  def forward(self,x):
    S = self.S(x)
    C = self.C(x)
    SC = S*C

    return torch.sigmoid(F.relu(self.bn1(self.conv(SC))))

						 
class GlobalNetwork(nn.Module):

  def __init__(self):
    super(GlobalNetwork, self).__init__()
    # Setting up the Sequential of CNN Layers
    
    self.senet154_ = senet154(num_classes=#num_classes, pretrained=None)

    self.layer0 = self.senet154_.layer0
    #global backbone
    self.layer1 = self.senet154_.layer1
    self.layer2 = self.senet154_.layer2
    self.layer3 = self.senet154_.layer3
    self.layer4 = self.senet154_.layer4

    self.fc_out = nn.Sequential(
                                nn.Linear(2048+512+1024, 2048)
                                nn.BatchNorm1d(2048),
                                nn.ReLU(True),
                                nn.Linear(2048,#num_classes)
                                )

    self.SC1 = SCA(512)
    self.SC2 = SCA(1024)
    self.SC3 = SCA(2048)

  def forward(self, x):
    batch_size = x.size()[0]  # obtain the batch size
    x0 = self.layer0(x)
    x1 = self.layer1(x0)
    x2 = self.layer2(x1)
    A2 = self.SC1(x2)
    x2_out = x2*A2

    GAP1 = F.adaptive_avg_pool2d(x2_out, (1, 1)).view(x2_out.size(0), -1)
    
    x3 = self.layer3(x2_out)
    A3 = self.SC2(x3)
    x3_out = x3*A3

    GAP2 = F.adaptive_avg_pool2d(x3_out, (1, 1)).view(x3_out.size(0), -1)

    x4 = self.layer4(x3_out)
    A4 = self.SC3(x4)
    x4_out = x4*A4

    x4_avg = F.avg_pool2d(x4_out, x4_out.size()[2:]).view(x4_out.size(0),  -1)

    concat = torch.cat([GAP1,GAP2,x4_avg],1)

    return self.fc_out(concat)
