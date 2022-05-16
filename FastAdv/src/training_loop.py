import pdb

import config
import torch as t
import torch.nn as nn
import os
import time
import torch.nn.functional as F
import test

import copy
import models
import math
import numpy as np
import gc
import pdb
import dropout


#IDEAS: Does amp change behaviour? Explore this, although probably not as you don't use amp in your other implementation
#Your dataloaders are differnet? Again, seems unlikely as I'm pretty sure that it all worked
#Your models are different? Seems like cap as they were like intalised the same?


def train(model,dataset,train_loader,args,strong_attack,dropout_method,optimiser,scheduler):


    if args.device == 'cuda':
        model = model.cuda()

    #Here, we have our optimiser (thing doing the optimisation on the neural network)

    epoch_size = len(train_loader)
    start_time = time.time()

    args.config.logger.info("epoch,train_loss,train_acc,time,test_acc,test_acc_adv")

    if args.early_stopping:
        previous_accuracy = 0.

    for epoch in range(0, args.epochs):

        if args.early_stopping:
            best_model = copy.deepcopy(model.state_dict())

        total_loss = 0
        total_correct = 0


        #Method: At the end of the whole thing, do a list of the adversarial iterator, and then turn it into


        for  (X, y,ind) in iter(train_loader):

            output = model(X)
            loss = F.cross_entropy(output, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()
            total_loss += loss

        if args.early_stopping and (epoch > 3 or args.epochs <= 3):
            model.eval()

            xs_test,ys_test = strong_attack.generate_attack((X,y))

            test_correct = (t.argmax(model(xs_test),dim=1) == ys_test).sum().item()
            test_accuracy = float(test_correct)/len(ys_test)

            if args.verbose_training:
                print(f"Current Accuracy: {test_accuracy}")

            if test_accuracy - previous_accuracy < -args.early_stopping_threshold:
                print("EARLY STOPPING")
                break

            best_model = copy.deepcopy(model.state_dict())
            previous_accuracy = test_accuracy

            model.train()


        if epoch + 1 == args.sampling_epoch and args.data_proportion != 1:

            dropout_method(model,dataset,train_loader)

        end_time = int(time.time()) - int(start_time)
        print(f"avg_loss: {total_loss/len(train_loader)} len: {len(train_loader)}")
        test.log_training(model,train_loader,epoch, total_loss, total_correct, epoch_size, end_time,optimiser,scheduler,args)

    if not args.early_stopping:
        best_model = model.state_dict()

    model.load_state_dict(best_model)
    model.float()
    model.eval()

    return model
