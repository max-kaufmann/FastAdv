import torch as t
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from torchvision import datasets, transforms
import math
from datasets import DropoutDataset

import sys
sys.path.append(".")

transform = transforms.Compose([transforms.ToTensor()])


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

def get_test_model(args):
    return nn.Sequential(nn.Flatten(),nn.Linear(100,10))



def get_model(args):

    return nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

def get_train_dataset(args):
    transform = transforms.ToTensor()

    data_proportion = 1.
    training_dataset = DropoutDataset.DropoutDataset(torchvision.datasets.MNIST("../datasets/MNIST/MNIST_train", train=True, download=True, transform=transform))

    return training_dataset

def get_test_dataset(args):

    test_dataset = DropoutDataset.DropoutDataset(torchvision.datasets.MNIST("../datasets/MNIST/MNIST_test", train=False, download=True, transform=transform))

    return test_dataset

def get_optimiser_and_scheduler(model,args):
    optimiser = t.optim.SGD(model.parameters(), lr = args.config.lr_max,weight_decay=args.config.weight_decay,momentum=args.config.momentum) #TODO: In paper they set it to lr.max at start, check if that matters
    scheduler = t.optim.lr_scheduler.CyclicLR(optimiser, base_lr=args.config.lr_min, max_lr=args.config.lr_max, step_size_up=args.epochs * args.config.dataset_length *0.5, step_size_down=args.epochs * args.config.dataset_length *0.5)
    return optimiser,scheduler