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

import torchvision.models as models


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
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if mode == 'train':
        
        
        return transforms.Compose([
            transforms.RandomApply([transforms.RandomRotation(360)], p = 0.5),
            #transforms.RandomHorizontalFlip(p=0.2),
            #transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply([transforms.CenterCrop(48)], p = 0.3),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5, fill=0),
            #transforms.RandomApply([transforms.RandomCrop(48)], p = 0.4),
            
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
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(in_features = 8*8*256, out_features = 128)
        self.fcbn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features = 128, out_features = 14)       
        self.dropout_rate = 0.5
        
        
    def forward(self, t):
        s = t
        s = self.bn1(self.conv1(s))        # batch_size x 32 x 64 x 64
        s = F.relu(F.max_pool2d(s, 2))     # batch_size x 32 x 32 x 32
        s = self.bn2(self.conv2(s))        # batch_size x 64 x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))     # batch_size x 64 x 16 x 16
        s = self.bn3(self.conv3(s))        # batch_size x 128 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))  
        

        s = s.view(-1, 8*8*256)  # batch_size x 8*8*128

        #apply 2 fully connected layers with dropout
        s = F.relu(self.fcbn1(self.fc1(s)))    # batch_size x 128
        s = self.fc2(s)                                     # batch_size x 6

        return F.log_softmax(s, dim=1)
    
        
 
class ResNet(nn.Module):
    def __init__(self, resnet_block, layers, img_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet Layers
        self.layer1 = self._layers(resnet_block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._layers(resnet_block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._layers(resnet_block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._layers(resnet_block, layers[3], out_channels=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def _layers(self, resnet_block, no_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1,
                                                          stride=stride),
                                                nn.BatchNorm2d(out_channels*4))
        
        layers.append(resnet_block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4
        
        for i in range(no_residual_blocks - 1):
            layers.append(resnet_block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.log_softmax(self.fc(x))
        return x

class resnet_block(nn.Module):
    def __init__(self, in_channels, out_channels, idt_downsample=None, stride=1):
        super(resnet_block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = idt_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            
        x += identity
        x = self.relu(x)
        return x


def ResNet50(img_channels=3, num_classes=14):
    return ResNet(resnet_block, [3,4,6,3], img_channels, num_classes)





class myCustomLoss(nn.Module):
    def __init__(self):
        super(myCustomLoss, self).__init__()

    def forward(self, output, target):
        
        print(output)
        print(target)


        #specifying the batch size
        batch_size = output.size()[0] 
        #calculating the log of softmax values           
        output = F.log_softmax(output, dim=1)  
        #selecting the values that correspond to labels
        output = output[range(batch_size), target]
        print("after range:")
        print(output)
        #returning the results
        return -torch.sum(output)/14



class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()

    def forward(self, output, target):


        #specifying the batch size
        batch_size = output.size()[0]
        output = output[range(batch_size), target]
        #calculating the log of softmax values           
        output = Long(F.log_softmax(output, dim=0))
        return F.kl_div(output, target)

    


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


net = models.resnet18(pretrained=False, num_classes=14)
#net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
#net = ResNet50()
lossFunc = nn.CrossEntropyLoss()
############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.9
batch_size = 64
epochs = 100
optimiser = optim.Adam(net.parameters(), lr=0.001)
