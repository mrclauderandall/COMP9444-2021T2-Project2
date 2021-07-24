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
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        return transforms.ToTensor()
    elif mode == 'test':
        return transforms.ToTensor()

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1      = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2      = nn.Conv2d(32, 300, kernel_size=3)
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.fc         = nn.Linear(64, 9)
        
    def forward(self, t):
        x = t
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.avgpool(x)
        x = self.fc(x.view(-1, x.shape[1]))
        return x

'''
class loss(nn.Module):
    """
    Class for creating a custom loss function, if desired.
    If you instead specify a standard loss function,
    you can remove or comment out this class.
    """
    def __init__(self):
        super(loss, self).__init__()

    def forward(self, output, target):
        pass

'''
net = Network()
lossFunc = nn.MSELoss()
############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 256
epochs = 10
optimiser = optim.Adam(net.parameters(), lr=0.001)





class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 5)     # image 24 x 24
        self.pool = nn.MaxPool2d(2)         # image 12 x 12 
        self.conv2 = nn.Conv2d(6,50,5)      # image 8 x 8
        self.fc = nn.Linear(4*4*50, 300)    # image 4 x 4
        self.output = nn.Linear(300, 10)


    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc(x))
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x