import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import time
import copy
from tqdm import tqdm
import os.path as osp

from utils import Config
from model import model
from data import get_dataloader



def train_model(dataloader, model, criterion, optimizer, device, num_epochs, dataset_size):
    model.to(device) #将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
    since = time.time() #记录当前时间
    best_model_wts = copy.deepcopy(model.state_dict()) #pth文件通过有序字典来保持模型参数，有序字典state_dict中每个元素都是Parameter参数，该参数是一种特殊的张量，包含data和requires_grad两个方法。其中data字段保存的是模型参数，requires_grad字段表示当前参数是否需要进行反向传播

    best_acc = 0.0

    for epoch in range(num_epochs): #几遍epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']: #分别记录看train和test跑出来的数据
            if phase=='train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'): #在使用的时候是设置一个上下文环境，也就是说只要设置了torch.set_grad_enabled(False)那么接下来所有的tensor运算产生的新的节点都是不可求导的
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred==labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase=='test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(best_model_wts, osp.join(Config['root_path'], Config['checkpoint_path'], 'model.pth'))
        print('Model saved at: {}'.format(osp.join(Config['root_path'], Config['checkpoint_path'], 'model.pth')))

    time_elapsed = time.time() - since#运行时间
    print('Time taken to complete training: {:0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best acc: {:.4f}'.format(best_acc))




if __name__=='__main__':

    dataloaders, classes, dataset_size = get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'], num_workers=Config['num_workers'])  ##返回快速读取文件的数据集，一共有几个分类，数据集大小{'train': len(y_train), 'test': len(y_test)} 编号的长度
    num_ftrs = model.fc.in_features 
    model.fc = nn.Linear(num_ftrs, classes) #用于设置网络中的全连接层的（in_features, out_features), classes这里代表最后的输出神经元

    criterion = nn.CrossEntropyLoss() ##交叉熵损失函数
    optimizer = optim.RMSprop(model.parameters(), lr=Config['learning_rate']) #设置优化器
    device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu') #创建一个GPU/CPU张量

    train_model(dataloaders, model, criterion, optimizer, device, num_epochs=Config['num_epochs'], dataset_size=dataset_size)