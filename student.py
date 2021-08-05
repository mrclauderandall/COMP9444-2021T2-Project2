#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.

"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    stats = ((0.4299, 0.4299, 0.4299), (0.2116, 0.2116, 0.2116))
    if mode == 'train':     
        return transforms.Compose([
            transforms.RandomApply([transforms.RandomRotation(360)], p = 0.5),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5, fill=0),
            transforms.RandomApply([transforms.RandomCrop(48)], p = 0.4),
            
            transforms.Resize(64),
            
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2),
            transforms.Normalize(*stats,inplace=True)
        ])

    elif mode == 'test':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats,inplace=True)
        ])


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.activation = nn.ReLU(inplace = True)

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.b1 = nn.BatchNorm2d(out_channels)

        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.b2 = nn.BatchNorm2d(out_channels)

        self.d = nn.Dropout(p=0.2)

    def forward(self, x):

        x = self.activation(self.b1(self.c1(x)))
        x = self.activation(self.b2(self.c2(x)))
        x = self.d(x)

        return x


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        block = Block

        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)


        self.layer1 = nn.Sequential(block(64, 64, 1), block(64, 128, 1))
        self.layer2 = nn.Sequential(block(128, 256, 2), block(256, 512, 2))
        #self.layer3 = nn.Sequential(block(256, 256, 2), block(256, 512, 2))
        #self.layer4 = nn.Sequential(block(256, 512, 2), block(512, 512, 2))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.h1 = nn.Linear(512, 14)
        
        for module in self.modules():
            init_weights(module)



    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.h1(x)
        #x = self.output(x)

        return x



net = Network()
lossFunc = nn.CrossEntropyLoss()
############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 1
batch_size = 64
epochs = 100
optimiser = optim.Adam(net.parameters(), lr=0.001)