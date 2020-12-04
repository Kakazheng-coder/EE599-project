# -*- coding: utf-8 -*-
"""Untitled6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14-AiBFjp6OzSm3up6XfMyWLunKQBnaWP
"""

from glob import glob
from google.colab import drive
import os
import sys
import numpy as np
import shutil

drive.mount('/content/drive')

os.chdir('/content/drive/MyDrive/Colab Notebooks/')

!ls

import matplotlib.pyplot as plt
import matplotlib.image as img
from os import listdir
from os.path import isfile, join
import numpy as np
import collections
from torchvision import transforms

image_root = './images/'
meta_root = './images/class/class.txt'

def label_with_index(meta_root): #build dictionary between label name and it's number by one-hot encode
    class_to_ix = {} #{label_name: index}
    ix_to_class = {} #{index: label_name}
    with open(meta_root, 'r') as txt:
        classes = [l.strip() for l in txt.readlines()]
        class_to_ix = dict(zip(classes, range(len(classes))))
        ix_to_class = dict(zip(range(len(classes)), classes))
        class_to_ix = {v: k for k, v in ix_to_class.items()}
    sorted_class_to_ix = collections.OrderedDict(sorted(class_to_ix.items()))
    return class_to_ix,ix_to_class,sorted_class_to_ix


def clean_label_images(class_to_ix, root): #Resize all images and label it
    all_imgs = []
    all_classes = []
    resize_count = 0
    invalid_count = 0
    
    transform_ReS = transforms.Resize((255,255))
    transform_PIL = transforms.ToPILImage()
    transform_ToTensor = transforms.ToTensor()

    for i, subdir in enumerate(listdir(root)): #i is current loop number, subdir is folder name
        imgs = listdir(join(root, subdir)) #get all the images name
        class_ix = class_to_ix[subdir]   # find the index matched with folder name
        #print(i, class_ix, subdir)   
        for img_name in imgs:     #do clean and label in this folder
            img_arr = img.imread(join(root, subdir, img_name)) #read image
            img_arr_rs = img_arr
            # print(img_arr_rs)
            try:
                img_arr_rs = transform_PIL(img_arr)
                img_arr_rs = transform_ReS(img_arr)
                img_arr_rs = transform_ToTensor(img_arr)
                
                all_imgs.append(img_arr_rs)  #collect images
                all_classes.append(class_ix) #collect labels
            except:
                invalid_count += 1

    return np.array(all_imgs), np.array(all_classes) #return images data information and labels
    

def match_label_image(X,y): # Match every data information with the lable
    matchset = []     
    for i in range(len(y)):
        matchset.append((X[i],y[i]))
        
    return matchset ##[(image_information[0],labels[0]),(image_information[1],labels[1])]

batch_size = 20
num_workers = 10

class_to_ix,ix_to_class,sorted_class_to_ix = label_with_index(meta_root) ##labels_name and its one-hot encode
classes = len(class_to_ix) # total labels number
X, y = clean_label_images(class_to_ix, image_root) #clean and label

class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

d = Train_data[:,:-1]
target = labels[:,-1]
dataset = MyDataset(X, y)
 
split_ratio=0.8 #80-20 train/validation split 
length_train = int(split_ratio*len(d)) #number of training samples 
length_valid = len(d) - length_train #number of training samples 

trainset, valset = torch.utils.data.random_split(dataset, [length_train, length_valid])

# Declaring the train validation and test loaders
#shuffle enabled as True for the train dataloader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

#shuffle enabled as False for the valid dataloader
valloader = torch.utils.data.DataLoader(valset, batch_size=16, shuffle=False) 

# inspecting the length of datasets (test and train datasets)
num_train_samples=len(trainset)
print(num_train_samples)  
num_val_samples=len(valset)
print(num_val_samples)  

inputs, labels = next(iter(trainloader)) #generate a single batch from trainloader 
print(len(trainloader)) #number of batches 

print(inputs.size())

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

#importing all the standard packages (numpy, torch, torchvision)
import matplotlib.pyplot as plt
import numpy as np

#torch,torchvision, torchvision transforms
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F #useful library for operations like relu 
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter 
from tqdm.notebook import tqdm


# Testing function
def eval_model(model,loader,criterion,device):
    """model: instance of model class 
       loader: test dataloader
       criterion: loss function
       device: CPU/GPU
    """
    model.eval() #needed to run the model in eval mode to freeze all the layers
    correct=0
    total=0
    total_loss=0
    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
    with torch.no_grad():
        total=0
        correct=0
        for idx,(inputs,labels) in enumerate(loader):
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            outputs=F.softmax(outputs,dim=1)
            val_loss=criterion(outputs,labels)
            total_loss=total_loss+val_loss
            preds=torch.max(outputs,dim=1)[1]
            
            # Append batch prediction results
            predlist=torch.cat([predlist,preds.view(-1).cpu()])
            lbllist=torch.cat([lbllist,labels.view(-1).cpu()])
            correct=correct+(preds==labels).cpu().sum().numpy() 
            total=total+len(labels)
    Accuracy=100*(correct/total)
    fin_loss=total_loss/(len(loader))
    
    return(Accuracy,  fin_loss,  predlist, lbllist)

#defining loss and optimizer 
criterion = nn.CrossEntropyLoss()   # includes softmax for this criterion
initial_learning_rate = 0.0001
optimizer = optim.Adam(net.parameters(), lr=initial_learning_rate, weight_decay = 0.01) # weight decay adds L2 optimizer
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1);  # learning rate schedular

num_epochs=3
iter_count = 0
net=net.to(device)
train_loss_list=[]
train_acc_list=[]
val_loss_list=[]
val_acc_list=[]
best_val_acc=0

#TRAINING LOOP
for i in np.arange(num_epochs): #outer loop 
    train_loss=0.0
    correct=0
    for idx,(inputs,labels) in enumerate(tqdm(trainloader, desc=f'Epoch {i+1:02d}')):
        iter_count += 1
        
        #sending inputs and labels to device 
        inputs=inputs.to(device)
        labels=labels.to(device)
        
        #zero out the gradients to avoid any accumulation during backprop
        optimizer.zero_grad()
        
        #forward pass through the network
        outputs = net(inputs) #batch_size x 10
        
        #compute the loss between ground truth labels and outputs
        loss = criterion(outputs, labels)
        
        loss.backward() #computes derivative of loss for every variable (gradients)
        optimizer.step() #optimizer updates based on gradients 
        
        preds=torch.max(outputs,dim=1)[1] # obtaining the predicted class (dimension of outputs is batch_size x number of classes)
        correct=correct+(preds==labels).cpu().sum().numpy() #.cpu() transfers tensors from GPU to CPU
        train_loss=train_loss+loss.item()    
        
    train_loss=train_loss/len(trainloader) #computing the total loss for the entire training set
    train_accuracy=100*(correct/len(trainloader.dataset)) #train accuracy for the dataset
    val_accuracy,val_loss,  predlist, lbllist =eval_model(net,valloader,criterion,device) #validation accuracy, validation loss for the entire validation set  

    
    train_loss_list.append(train_loss)
    train_acc_list.append(train_accuracy)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_accuracy)
    net.train(True)
    print('Epoch:%d,Train Loss:%f,Training Accuracy:%f,Validation Accuracy:%f'%(i+1,train_loss,train_accuracy,val_accuracy))
    if(val_accuracy > best_val_acc):
        print('Saving the best model')
        best_val_acc=val_accuracy
        torch.save({
            'epoch': i+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss}, 'best_model.pth') #saving all the required information in .pth file (required for restarting models later)        

plt.figure()
plt.plot(np.arange(num_epochs), train_loss_list, label='Training loss')
plt.plot(np.arange(num_epochs), val_loss_list, label='Validation_loss')
plt.xlabel('epochs')
plt.ylabel('Multiclass Cross Entropy Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.arange(num_epochs), np.array(train_acc_list)/100, label='Training accuracy')
plt.plot(np.arange(num_epochs), np.array(val_acc_list)/100, label='Validation accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#saving the entire model 
torch.save(net,'model.pth')

### loading the saved entire model 
model_total=torch.load('model.pth')

### loading the saved best model with state dict 

best_model=net #declare the model class 
checkpoint=torch.load('best_model.pth')
best_model.load_state_dict(checkpoint['model_state_dict'])

##inference with the saved entire model 
model_total.eval()
test_accuracy_total, test_loss_total, predlist, lbllist = eval_model(model_total,testloader,criterion,device)
print('Test accuracy using the entire saved model:%f' %(test_accuracy_total))
print('Test loss using the entire saved model:%f' %(test_loss_total))

##inference with the best model loaded from state dict
best_model.eval()
test_accuracy_best, test_loss_best, predlist, lbllist = eval_model(best_model,testloader,criterion,device)
print('Test accuracy using the best saved model:%f' %(test_accuracy_best))
print('Test loss using the best saved model:%f' %(test_loss_best))
