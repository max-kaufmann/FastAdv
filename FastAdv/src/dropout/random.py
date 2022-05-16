import pdb
import torch as t
import torch.nn as nn
import os
import time
import torch.nn.functional as F
import copy
import math
import numpy as np
import gc
import pdb

def dropout(model,dataset,dataloader,args):

    proportion = math.floor(len(dataset) * args.data_proportion)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    dataset.remove(indices[0:-proportion])
