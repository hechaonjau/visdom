#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HC MSH on 2018/12/30

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from visdom import Visdom
import numpy as np
import time


viz = Visdom()

from model import CNN
#数据集地址
train_path='/home/hc/文档/dogs vs cats/cats_vs_dogs_data/data/train'
test_path='/home/hc/文档/dogs vs cats/cats_vs_dogs_data/data/test'


#超参数
num_epochs = 200
batch_size = 200
learning_rate = 0.001

#获得数据并转化为Tensor
train_dataset = dsets.ImageFolder(root=train_path,
                                  transform=transforms.Compose([
                                      transforms.Resize(128),
                                      transforms.CenterCrop(128),
                                      transforms.ToTensor(),   #归一化
                                  ])
                                  )#改变图片的size，标准化

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True     #随机排序
                                           )

test_dataset = dsets.ImageFolder(root=test_path,
                                  transform=transforms.Compose([
                                      transforms.Resize(128),
                                      transforms.CenterCrop(128),
                                      transforms.ToTensor(),
                                  ])
                                  )#改变图片的size，标准化

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size = batch_size,
                                           shuffle = False
                                           )

cnn = CNN()
print(cnn)
cnn = cnn.cuda()




line = viz.line(np.arange(10))
if __name__ == '__main__':
    #误差和优化
    loss_fun = nn.CrossEntropyLoss()
    loss_fun = loss_fun.cuda()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)    #采用SGD优化网络
    start_time = time.time()
    time_p, tr_acc,loss_p = [], [], []
    text = viz.text("<h1>convolution Nueral Network</h1>")
    #训练模型
    #cnn.train()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0   #训练集准确率
        correct = 0    #正确数
        total = 0     #总数
        sum_loss=0
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())           #使用cuda
            #前向传播 反向传播 优化
            optimizer.zero_grad()
            output = cnn.forward(images)
            loss = loss_fun.forward(output, labels)
            loss.backward()
            optimizer.step()
            _, predict = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predict == labels.cuda()).sum().item()
            if (i + 1) % 1 == 0:
                print('训练次数 [%d/%d], 步数 [%d/%d], Loss： %.4f,  Acc: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0], correct / total))
            sum_loss += loss.data[0] * len(labels)




            #可视化部分（使用visdom）
            #每25个batch可视化一次
            if i % 25 == 0:
                time_p.append(time.time()-start_time)
                tr_acc.append(correct/total)
                loss_p.append(sum_loss/total)
                viz.line(X=np.column_stack((np.array(time_p), np.array(time_p))),
                         Y=np.column_stack((np.array(loss_p), np.array(tr_acc))),
                        win=line,
                        opts=dict(legend=["Loss", "TRAIN_acc"]))
                # visdom text 支持html语句
                viz.text("<p style='color:red'>epoch:{}</p><br><p style='color:blue'>Loss:{:.4f}</p><br>"
                        "<p style='color:BlueViolet'>TRAIN_acc:{:.4f}</p><br><p style='color:orange'>"
                        "<p style='color:green'>Time:{:.2f}</p>".format(epoch, sum_loss/total,correct/total,
                                                                       time.time()-start_time),
                        win=text)
                sum_loss, correct, total = 0., 0., 0.




#存储模型
#保存整个网络
torch.save(cnn, '/home/hc/文档/dogs vs cats/codes/net/net3/net3.pkl')
#保存网络的参数
torch.save(cnn.state_dict(), '/home/hc/文档/dogs vs cats/codes/net/net3/net_params3.pkl')
