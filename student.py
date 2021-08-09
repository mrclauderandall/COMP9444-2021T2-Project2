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
    - Data Transforms -
We decided to implement a number of data transformations to the input images.
For the training inputs, in order to generate a larger variance of images to train off,
we implemented RandomRotation, RandomPerspective, RandomCrop, and RandomErasing. All of
these modify or distort the input images in some way, which makes training harder, but means
that the network, over a greater period of time, will value the important or distinguishing
features more effectively. We carefully designated the percentages of our transforms, so that
a completely unaltered image occurs 12% of the time, and also that the more distortive of the
transforms (RandomCrop, RandomErasing) occur less. This allows our network a larger dataset to train
on without necessarily learning some of the patterns associated with the more distortive transforms.
For both Testing and Training, we normalized the inputs, based on a mean of 0.4299 and an
standard deviation of 0.2116. We specifically calculated these values based on the averages
of the images, in a method that is described in more detail here
(https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html).
    - Neural Network Architecture -
We explored many different ideas and implementations over the course of our development, and
eventually found that this particular dataset is prone to overfitting on networks that are too deep.
As such, our network only has 10 layers, the majority of which are Convolutional layers. Our overall
architecture draws much inspiration from the layer design of ResNet18, outlined in Table 1 of
'Deep Residual Learning for Image Recognition' (https://arxiv.org/abs/1512.03385, pg.5). As the paper
states, residual layers are only noticeably effective in significantly deep networks (pg.5), and as we
found that this dataset isnt one that requires a deep network, we felt it unnecessary to implement any
sort of residual calculation. We also condensed the architecture from 18 layers to 10, which we found far
more accurate. Thus, our Network starts out similar to ResNet18, with a Convolutional Layer of kernel_size 7,
and 64 outputs, which is batch-normalised, activated with ReLU. This is then run through a MaxPool2d function,
containing the same kernel_size (3) and stride (2) as outlined in the paper. We then have two layers,
each containing 2 'blocks', of which each block contains two Convolutional layers. We initalize the weights of these
convolutional layers according to the Xavier Uniform function These layers progress in
size from 64 to 512 at the end of the second block, and each layer is batch-normalised and activated with ReLU.
Each block has a dropout rate of p=0.2 applied on output, which limits overfitting in our network
and improved our accuracy noticeably. After these two layers of blocks, we apply an AdaptiveAvgPool2d of 1x1, as
outlined in the paper, before resizing the output and running it through a Linear layer which results in an output
size of 14, classifying one of the 14 characters.
    - Weight initialization - 
We used Kaiming normal initialization as according to research it is best suited for non-linear layers using ReLU 
activation (https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138).
With testing, we found slightly better results using the fan_out mode over the fan_in mode.
    - Loss Function -
After trying multiple functions, we settled on a standard CrossEntropyLoss function, which we found most effective
and researching seemed to back this up, as it is normally used for classification problems
    - Optimizer/Metaparameters -
We kept the learning rate at 0.001, which was most effective combined with our optimizer and architecture. A batch
size of 64 also yielded best results after much testing, as well as 200 epochs, after which we found greatly
diminishing returns in terms of improving accuracy/loss v time-to-train. We chose Adam as our optimizer after
research and testing, all of which pointed to it being far-and-away the most effective.
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
        nn.init.kaiming_normal_(m.weight, mode='fan_out')

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

        # define activation and pooling functions
        self.activation = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # block layers
        self.layer1 = nn.Sequential(block(64, 64, 1), block(64, 128, 1))
        self.layer2 = nn.Sequential(block(128, 256, 2), block(256, 512, 2))

        # average pooling and linear output layer
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.h1 = nn.Linear(512, 14)

        # weight initialization
        for module in self.modules():
            init_weights(module)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
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
train_val_split = 1
batch_size = 64
epochs = 200
optimiser = optim.Adam(net.parameters(), lr=0.001)