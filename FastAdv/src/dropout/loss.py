import pdb
import config
import torch as t
import torch.nn as nn
import os
import time
import torch.nn.functional as F
import copy
import models
import math
import numpy as np
import torch as t
import gc
import pdb
import FastAdv.src.dropout as dropout
def get_losses(model,dataloader,args):

    indices = []
    losses = []
    loss_func = F.cross_entropy
    model.eval()
    with t.no_grad():
        for xs,ys,inds in iter(dataloader):

            if args.device == "gpu":
                xs,ys = xs.cuda(),ys.cuda()

            losses.append(F.cross_entropy(model(xs),ys,reduction="none").cpu())
            indices.append(inds.cpu())

    model.train()

    return indices,losses

def dropout(model,dataset,train_loader,args):
    indices, loss_values = get_losses(model, train_loader, args)
    indices = np.concatenate(indices)
    loss_values = np.concatenate(loss_values)
    mean_loss_vals = np.mean(loss_values)
    indices_as = np.argsort(np.abs(loss_values - mean_loss_vals))
    upper = int(args.data_proportion * len(dataset))
    dataset.remove(indices[indices_as[upper:]])

