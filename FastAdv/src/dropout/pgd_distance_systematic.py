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
from  dataloaders.PgdDistanceDataloader import *
import FastAdv.src.dropout as dropout

def dropout(model,dataset,dataloader,args):
        model.eval()
        distance_attack = PgdDistanceDataloader(dataloader,model, args.epsilon, args.alpha, args.num_steps, args.num_restarts, args)
        start_time = time.time()

        iteration_list = list(iter(distance_attack))
        xs,inds,ks = [t.concat(x) for x in zip(*iteration_list)] # returns xs,indices, and distances
        # We know the indices returned are the correct indices for the xs as dataset.cifar[inds[i]][0] varies from x[i] only in adversarial perturbation
        targets = dataset.cifar.targets

        ys = targets[inds]
        ys_ = t.argsort(t.tensor(ys,device=args.device))
        #his works, and we know it works because the related indices are vibing

        class_locs= t.split(ys_,config.class_split) #this splits all the input into the correct indices, we know this works as
        class_inds = [inds[i] for i in class_locs]#as this returns classes all from the same index
        class_ks = [ks[i] for i in class_locs]

        to_remove = []
        for c in range(0,config.num_classes):
            c_ks_ = t.argsort(class_ks[c])
            c_inds = class_inds[c]
            rem_k = c_ks_[math.floor(args.data_proportion*len(c_ks_)):]
            to_remove.append(c_inds[rem_k])
        to_remove = t.concat(to_remove)
        #THE REASON THIS SHIT IS ALL IN ORDER IS BECAUSE ARGSORT IS IN PLACE YOU CUNT


        dataset.remove(to_remove)
        model.train()