import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist
import torchvision.models as models

import h5py
import time
import os

import sys
import torchvision.models as models


#load training and testing datasets
train = torchvision.datasets.CIFAR100(root='./data',train=True,download=True,transform=transforms)
trainset = torch.utils.data.DataLoader(train,batch_size=128,shuffle=True)

test = torchvision.datasets.CIFAR100(root='./data',train=False,download=True,transform=transforms)
testset = torch.utils.data.DataLoader(test,batch_size=128,shuffle=False)

#upsampling for CIFAR100 - previously ImageNet dimensions
DIM = 224

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(DIM, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(DIM, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

#previously model has output size of 1000 (classifications for ImageNet)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512,100)

acc = 0  #Store best accuracy
epoch = 0 #store epochs

optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
costFunction = torch.nn.CrossEntropyLoss()

def train_model(epochs):
	model.train()

	for epoch in range(epochs):

		epoch_acc = 0
		epoch_loss = 0.0
		epoch_counter = 0

		for i,batch in enumerate(trainset,0):

			data , actual = batch
			#data = Variable(data)#.cuda()
			#actual = Variable(actual)#.cuda()

			with torch.no_grad():
				h = model.conv1(x)
				h = model.bn1(h)
				h = model.layer1(h)
				h = model.layer2(h)
				h = model.layer3(h)
			h = model.layer4(h)
			h = model.avgpool(h)
			h = h.view(h.size(0), -1)

			output = model.fc(h)
			prediction = output.data.max(1)[1]
			epoch_acc += float(prediction.eq(actual.data).sum())
			epoch_counter += 128

			loss = costFunction(prediction,actual)
			epoch_loss += loss.item() 

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		epoch_acc /= epoch_counter
		epoch_loss /= (epoch_counter/128)

		print(epoch, "Accuracy:" , "%.2f" % ((epoch_acc)*100.0), "Loss: ", "%.4f" % epoch_loss)


if __name__ == '__main__':
	train_model(5)


