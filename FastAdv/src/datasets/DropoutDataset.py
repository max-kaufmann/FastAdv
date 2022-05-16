import torch as t
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DropoutDataset(t.utils.data.Dataset):
    def __init__(self,dataset):
        self.data = dataset
        self.data.targets = np.array(self.data.targets)

    def __getitem__(self,index):
        data,target = self.data[index]

        return data,target,index


    def __len__(self):
        return len(self.data)

    def remove(self,indices):
        mask = np.ones(shape=len(self.data),dtype=bool)
        mask[indices] = False
        self.data.data = self.data.data[mask]
        self.data.targets = self.data.targets[mask]