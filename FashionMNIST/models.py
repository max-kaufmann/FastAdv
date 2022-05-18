import torch as t
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from torchvision import datasets, transforms,models
import config
import math

transform_train =  transforms.Compose([
        #transforms.Resize(28),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
transform_test = transforms.Compose([#transforms.Resize(28),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


#TODO: Rewrite this as a normal thing
class IndexDataset(t.utils.data.Dataset):
    def __init__(self,dataset):
        self.dataset= dataset
        self.dataset.targets = np.array(self.dataset.targets)

    def __getitem__(self,index):
        data,target = self.dataset[index]

        return data,target,index


    def __len__(self):
        return len(self.dataset)

    def remove(self,indices):
        mask = np.ones(shape=len(self.dataset),dtype=bool)
        mask[indices] = False
        self.dataset.data = self.dataset.data[mask]
        self.dataset.targets = self.dataset.targets[mask]
        self.data = self.dataset.data

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



def get_mnist_architecture(args):
    model = models.resnet18(pretrained=True)
    # for param in model.parameters():
    #    param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model = nn.Sequential(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), model)
    return model

def get_mnist_train_dataloader(args):
    transform = transforms.ToTensor()

    data_proportion = 1.
    training_dataset = IndexDataset(torchvision.datasets.FashionMNIST("../data/Fashion_MNIST_test", train=True, download=True, transform=transform_train))

    training_dataloader = t.utils.data.DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=4, pin_memory=True)

    return training_dataloader

def get_mnist_test_dataloader(args):

    test_dataset = IndexDataset(torchvision.datasets.FashionMNIST("../data/Fashion_MNIST_test", train=False, download=True, transform=transform_test))
    test_dataloader = t.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return test_dataloader