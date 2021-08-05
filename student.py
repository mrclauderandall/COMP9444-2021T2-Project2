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
            #transforms.RandomApply([transforms.ColorJitter(brightness=0, contrast=1, saturation=0, hue=0)], p = 0.3),
            
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2),
            transforms.Normalize(*stats,inplace=True)
        ])

    elif mode == 'test':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats,inplace=True)
        ])





############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)


        self.c1 = nn.Conv2d(3, 64, 7, stride = 1, padding = 3) # -> 64
        self.b1 = nn.BatchNorm2d(64)
        self.c2 = nn.Conv2d(64, 128, 3, stride = 1, padding = 1) # -> 32
        self.b2 = nn.BatchNorm2d(128)
        self.c3 = nn.Conv2d(128, 256, 3, stride = 1, padding = 1) # -> 32
        self.b3 = nn.BatchNorm2d(256)

        
        self.c4 = nn.Conv2d(256, 64, 3, stride = 1, padding = 1) # -> 32
        self.b4 = nn.BatchNorm2d(64)
        self.c5 = nn.Conv2d(64, 128, 3, stride = 1, padding = 1) # -> 32
        self.b5 = nn.BatchNorm2d(128)
        self.c6 = nn.Conv2d(128, 256, 3, stride = 1, padding = 1) # -> 32
        self.b6 = nn.BatchNorm2d(256)

        self.c7 = nn.Conv2d(256, 64, 3, stride = 1) # -> 32
        self.b7 = nn.BatchNorm2d(64)
        self.c8 = nn.Conv2d(64, 128, 3, stride = 1) # -> 32
        self.b8 = nn.BatchNorm2d(128)
        self.c9 = nn.Conv2d(128, 256, 3, stride = 1) # -> 32
        self.b9 = nn.BatchNorm2d(256)

        #pool
        self.c10 = nn.Conv2d(256, 512, 3, stride = 1, padding = 1) # -> 32
        self.b10 = nn.BatchNorm2d(512)

        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.h1 = nn.Linear(512, 14)
        
        
    def forward(self, x):
        
        x = self.c1(x)
        x = self.b1(x)
        x = self.relu(x)

        x = self.pool(x)

        
        x = self.c2(x)
        x = self.b2(x)
        x = self.relu(x)
        




        x = self.relu(self.b3(self.c3(x)))
        x = self.relu(self.b4(self.c4(x)))
        x = self.relu(self.b5(self.c5(x)))
        x = self.relu(self.b6(self.c6(x)))
        x = self.relu(self.b7(self.c7(x)))
        x = self.relu(self.b8(self.c8(x)))
        x = self.relu(self.b9(self.c9(x)))
        x = self.relu(self.b10(self.c10(x)))


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.h1(x)

        return x

        


net = Network()
lossFunc = nn.CrossEntropyLoss()
############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.9
batch_size = 64
epochs = 100
optimiser = optim.Adam(net.parameters(), lr=0.001)