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
from  FastAdv.src.dataloaders.PgdDistanceDataloader import *
import FastAdv.src.dropout as dropout

def dropout(model,dataset,dataloader,args):
        model.eval()
        distance_attack = PgdDistanceDataloader(dataloader,model, args.epsilon, args.alpha, args.num_steps, args.num_restarts, args)

        iteration_list = list(iter(distance_attack))
        xs,inds,ks = [t.concat(x) for x in zip(*iteration_list)] # returns xs,indices, and distances

        ks_sorted = t.argsort(ks)
        len_remove = math.floor(args.data_proportion * len(ks_sorted))
        to_remove = inds[ks_sorted[len_remove:]]

        dataset.remove(to_remove)
        model.train()