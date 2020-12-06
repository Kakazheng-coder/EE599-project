import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import numpy as np
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image

from utils import Config


def get_data_transforms():  #Resize
    data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(224), 
                transforms.ToTensor(), 
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
            ]),
            'test': transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
    return data_transforms


def get_X_y_label(test_label_path,train_label_path,class_list_path):
    f_test = open(test_label_path)
    f_train = open(train_label_path)
    f_label = open(class_list_path)
    X_test = []
    y_test = []
    X_train = []
    y_train = []
    label_dict = {}

    next(f_test)
    for line in f_test:  #读取的是以每行字符串的格式
        tokens = [t for t in line[:-1].split(',')] #每行的两个节点组成的列表,每个列表里的单个节点号码为字符串
        X_test.append(tokens[0])
        y_test.append(tokens[1]) #得到所有节点的集合，#一共1005个节点

    next(f_train)
    for line in f_train:  #读取的是以每行字符串的格式
        tokens = [t for t in line[:-1].split(',')] #每行的两个节点组成的列表,每个列表里的单个节点号码为字符串
        X_train.append(tokens[0])
        y_train.append(tokens[1]) 
   
    for line in f_label:  #读取的是以每行字符串的格式
        tokens = [t for t in line[:-1].split(' ')] #每行的两个节点组成的列表,每个列表里的单个节点号码为字符串
        label_dict[tokens[0]] = tokens[1]
        classes = len(label_dict)

        
    return X_train,y_train,X_test,y_test,classes


# For category classification
class polyvore_train(Dataset): 
    def __init__(self, X_train, y_train, transform):
        self.X_train = X_train
        self.y_train = y_train
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], Config['train_set'],Config['train_set'])

    def __len__(self):
        return len(self.X_train) #train

    def __getitem__(self, item):  
        file_path = osp.join(self.image_dir, self.X_train[item])
        return self.transform(Image.open(file_path)),self.y_train[item]




class polyvore_test(Dataset):
    def __init__(self, X_test, y_test, transform):
        self.X_test = X_test
        self.y_test = y_test
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], Config['valaditon_set'], Config['valaditon_set'])


    def __len__(self):
        return len(self.X_test) ##test


    def __getitem__(self, item): 
        file_path = osp.join(self.image_dir, self.X_test[item])
        return self.transform(Image.open(file_path)), self.y_test[item]

def un_zip(file_name):
    """unzip zip file"""
    zip_file = zipfile.ZipFile(file_name)
    
    if os.path.isdir(file_name[:-4]):
        pass
    else:
        os.mkdir(file_name[:-4])
    for names in zip_file.namelist():
        zip_file.extract(names,file_name[:-4])
    zip_file.close()


def get_dataloader(debug, batch_size, num_workers):



    test_label_path = osp.join(Config['root_path'], 'val_labels.csv')
    train_label_path = osp.join(Config['root_path'], 'train_labels.csv')
    class_list_path = osp.join(Config['root_path'], 'class_list.txt')

    un_zip(osp.join(Config['root_path'], 'train_set.zip'))   
    un_zip(osp.join(Config['root_path'], 'val_set.zip'))    

    transforms = get_data_transforms()
    X_train,y_train,X_test,y_test,classes = get_X_y_label(test_label_path,train_label_path,class_list_path)

    if debug==True: 
        train_set = polyvore_train(X_train[:100], y_train[:100], transform=transforms['train'])
        test_set = polyvore_test(X_test[:100], y_test[:100], transform=transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}
    else:
        train_set = polyvore_train(X_train, y_train, transforms['train']) 
        test_set = polyvore_test(X_test, y_test, transforms['test']) 
        dataset_size = {'train': len(y_train), 'test': len(y_test)} 

    datasets = {'train': train_set, 'test': test_set} 
    dataloaders = {x: DataLoader(datasets[x], 
                                 shuffle=True if x=='train' else False, 
                                 batch_size=batch_size, 
                                 num_workers=num_workers) 
                                 for x in ['train', 'test']} 
    return dataloaders, classes, dataset_size 




########################################################################
# For Pairwise Compatibility Classification

