import torch as t
import config
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import config
import math
import numpy as np

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config.default_mean, config.default_std),
    ])


transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize(config.default_mean, config.default_std)])


class PreActBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class IndexDataset(t.utils.data.Dataset):
    def __init__(self,transform):
        self.cifar = torchvision.datasets.CIFAR10("./data/CIFAR10_train", train=True, download=True, transform= transform )
        self.cifar.targets = np.array(self.cifar.targets)

    def __getitem__(self,index):
        data,target = self.cifar[index]

        return data,target,index


    def __len__(self):
        return len(self.cifar)

    def remove(self,indices):
        mask = np.ones(shape=len(self.cifar),dtype=bool)
        mask[indices] = False
        self.cifar.data = self.cifar.data[mask]
        self.cifar.targets = self.cifar.targets[mask]
        self.data = self.cifar.data

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


class CIFAR10_classifier(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(CIFAR10_classifier, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class test_network(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3,10)

    def forward(self,x):
        x = t.flattten(x,1)
        x = self.fc1(x)
        return x


class CIFAR10_classifier_small(nn.Module):

    def __init__(self): #This just holds what is occuring

        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, (3,3))
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5,inplace=False)
        self.conv2 = nn.Conv2d(16, 32, (3,3))
        self.fc1 = nn.Linear(1152, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x): #This is the forward pass

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = t.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def get_cifar10_architecture(args,normalised=True):

    if args.model == "resnet":
        model = CIFAR10_classifier(PreActBlock,[2,2,2,2])

        if not normalised:
            model = nn.Sequential(transforms.Normalize(config.default_mean, config.default_std),model)

    if args.model == "small":
        model = CIFAR10_classifier_small()

    return model



def get_cifar10_train_dataloader(args,normalised=True):

    transform = transforms.ToTensor()
    if normalised:
        transform = transform_train

    if args.debug:
        transform = transform_test

    data_proportion = 1.
    training_dataset = IndexDataset(transform)


    training_dataloader = t.utils.data.DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,pin_memory = True)


    return training_dataloader

def get_cifar10_test_dataloader(args,normalised=True):

    transform = transforms.ToTensor()
    if normalised:
        transform = transform_test

    pin = False
    if args.device =='gpu':
        pin = True

    test_dataset = torchvision.datasets.CIFAR10("./data/CIFAR10_test", train=False, download=True, transform=transform)
    test_dataloader = t.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,pin_memory=pin)

    return test_dataloader

    #We want to make it broadcastable, so we change it to this size