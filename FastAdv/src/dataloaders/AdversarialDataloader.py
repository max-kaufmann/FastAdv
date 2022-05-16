import torch as t
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import abc
import gc
import autoattack

class AdversarialDataloader():

    def __init__(self,dataloader):
        self.dataloader=dataloader

    @abc.abstractmethod
    def generate_attacks(self,data):
        return None

    def __iter__(self):
        class AdversarialIterator:
            def __init__(s, dataloader):
                s.dataloader = dataloader
                s.dataloader_iterator = iter(dataloader)

            def __iter__(s):
                return s

            def __len__(s):
                return len(s.dataloader)

            def __next__(s):
                current_batch = next(s.dataloader_iterator)
                return self.generate_attack(current_batch)

        return AdversarialIterator(self.dataloader)

    def __len__(self):
        return len(self.dataloader)





