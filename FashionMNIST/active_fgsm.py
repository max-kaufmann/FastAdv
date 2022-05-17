import config
import torch as t
import torch.nn as nn
import testing
import os
import time
import torch.nn.functional as F
import attacks
import copy
import models
import math
import numpy as np
import gc
import pdb

#IDEAS: Does amp change behaviour? Explore this, although probably not as you don't use amp in your other implementation
#Your dataloaders are differnet? Again, seems unlikely as I'm pretty sure that it all worked
#Your models are different? Seems like cap as they were like intalised the same?

def clamp(data,min,max):
    return t.max(t.min(data,max),min)

def train(model,train_loader,args,saved_model=None):

    print("ACTIVE TRAINING STARTING")

    if args.device == 'gpu':
        model = model.cuda()

    model.train()

    #Here, we have our optimiser (thing doing the optimisation on the neural network)
    optimiser = t.optim.Adam(model.parameters(), lr = config.lr_max) #TODO: In paper they set it to lr.max at start, check if that matters
    lr_generator = lambda t: np.interp(t, [0, args.epochs * 2 // 5, args.epochs], [0, config.lr_max, 0])

    epoch_size = len(train_loader)
    start_time = time.time()

    config.logger.info("epoch,train_loss,train_acc,time,test_acc,test_acc_adv")

    if args.early_stopping:
        previous_accuracy = 0.
        attack = attacks.pgd_attack(model, args.epsilon_test, args.alpha_test, config.early_stopping_steps, config.early_stopping_restarts, args)
        best_model = None

    device = "cpu" if args.device == "cpu" else "cuda"
    for epoch in range(0, args.epochs):

        if args.early_stopping:
            best_model = copy.deepcopy(model.state_dict())

        total_loss = 0
        total_correct = 0


        #Method: At the end of the whole thing, do a list of the adversarial iterator, and then turn it into


        for (i, (X, y,ind))in enumerate(train_loader):

            lr = lr_generator(epoch + (i+1)/len(train_loader))
            optimiser.param_groups[0].update(lr=lr)

            delta = t.zeros_like(X)
            if args.device == 'gpu':
                X, y = X.cuda(), y.cuda()
                delta = delta.cuda()

            delta = (0.5 - t.rand(X.shape, device=device)) * 2 * args.epsilon_training

            delta.data = clamp(delta, config.min_vals - X, config.max_vals - X)
            delta.requires_grad = True
            output = model(X + delta[:X.size(0)])
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            delta.data = clamp(delta + (args.alpha * t.sign(grad)), -args.epsilon_training, args.epsilon_training)
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], config.min_vals - X, config.max_vals - X)
            delta = delta.detach()
            output = model(X + delta[:X.size(0)])
            loss = F.cross_entropy(output, y,reduction="none")
            comb_loss = t.sum(loss) / args.batch_size #remember that you changed this :)
            optimiser.zero_grad()
            comb_loss.backward()
            optimiser.step()
            total_loss += comb_loss

        if epoch + 1 == args.sampling_epoch and args.data_proportion != 1:

            model.eval()

            distance_attack = attacks.ga_pgd_attack(model, args.epsilon_training, args.alpha, args.pgd_steps,
                                                    args.pgd_restarts_train, args)
            start_time = time.time()

            iteration_list = list(distance_attack.iterator_from_dataloader(train_loader))
            xs, inds, ks = [t.concat(x) for x in zip(*iteration_list)]  # returns xs,indices, and distances
            # We know the indices returned are the correct indices for the xs as dataset.cifar[inds[i]][0] varies from x[i] only in adversarial perturbation
            dataset = train_loader.dataset
            targets = dataset.mnist.targets

            if args.systematic_sampling:
                ys = targets[inds]
                ys_ = t.argsort(t.tensor(ys, device=device))
                # his works, and we know it works because the related indices are vibing

                class_locs = t.split(ys_,
                                     config.class_split)  # this splits all the input into the correct indices, we know this works as
                class_inds = [inds[i] for i in class_locs]  # as this returns classes all from the same index
                class_ks = [ks[i] for i in class_locs]

                to_remove = []
                for c in range(0, config.num_classes):
                    c_ks_ = t.argsort(class_ks[c])
                    c_inds = class_inds[c]
                    rem_k = c_ks_[math.floor(args.data_proportion * len(c_ks_)):]
                    to_remove.append(c_inds[rem_k])
                to_remove = t.concat(to_remove)
                # THE REASON THIS SHIT IS ALL IN ORDER IS BECAUSE ARGSORT IS IN PLACE YOU CUNT

            if args.debug:
                pdb.set_trace()


            else:
                ks_sorted = t.argsort(ks)
                len_remove = math.floor(args.data_proportion * len(ks_sorted))
                to_remove = inds[ks_sorted[len_remove:]]



        end_time = int(time.time()) - int(start_time)
        print(f"avg_loss: {total_loss/len(train_loader)} len: {len(train_loader)}")
        testing.log_training(model,epoch, total_loss, total_correct, epoch_size, end_time,optimiser,None,args)

        if args.early_stopping and (epoch > 3 or args.epochs <= 3):
            model.eval()

            xs_test,ys_test = attack.generate_attack((X,y))

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

    if not args.early_stopping:
        best_model = model.state_dict()

    return_model = models.get_mnist_train_dataloader(args).cuda()
    return_model.load_state_dict(best_model)
    return_model.float()
    return_model.eval()

    return return_model

if t.__version__ == "1.7.1":
    t.clamp = clamp