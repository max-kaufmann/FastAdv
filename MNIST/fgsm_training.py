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

#IDEAS: Does amp change behaviour? Explore this, although probably not as you don't use amp in your other implementation
#Your dataloaders are differnet? Again, seems unlikely as I'm pretty sure that it all worked
#Your models are different? Seems like cap as they were like intalised the same?

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

def clamp(data,min,max):
    return t.max(t.min(data,max),min)

def train(model,train_loader,args,saved_model=None):

    print("FGSM TRAINING STARTING")

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
        attack = attacks.Auto_attack(model,  args)
        best_model = None

    for epoch in range(0, args.epochs):


        if args.early_stopping:
            best_model = copy.deepcopy(model.state_dict())

        total_loss = 0
        total_correct = 0

        if args.sampling_epoch - 1 == epoch and args.data_proportion != 1:
            indices = [] #TODO: Ask Ilia and Yiren if this is slow
            loss_values = []


        device = "cpu" if args.device=="cpu" else "cuda"
        for (i, (X, y,ind))in enumerate(train_loader):

            lr = lr_generator(epoch + (i+1)/len(train_loader))
            optimiser.param_groups[0].update(lr=lr)

            delta = (0.5 - t.rand(X.shape, device=device)) * 2 * args.epsilon_training

            if args.device == 'gpu':
                X, y = X.cuda(), y.cuda()
                delta = delta.cuda()
            delta.requires_grad = True
            output = model(X + delta[:X.size(0)])
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            delta.data = t.clamp(delta + (args.alpha * t.sign(grad)), -args.epsilon_training, args.epsilon_training)
            delta.data[:X.size(0)] = t.clamp(delta[:X.size(0)], config.min_vals - X, config.max_vals - X)
            delta.data = t.clamp(delta, config.min_vals - X, config.max_vals - X)
            delta = delta.detach()
            output = model(X + delta[:X.size(0)])
            loss = F.cross_entropy(output, y,reduction="none")
            comb_loss = t.sum(loss) / args.batch_size #remember that you changed this :)
            optimiser.zero_grad()
            comb_loss.backward()
            optimiser.step()
            total_loss += comb_loss


        if epoch + 1 == args.sampling_epoch and args.data_proportion != 1 and args.sampling_method == "drop_extremes":
            indices, loss_values = get_losses(model, train_loader, args)
            indices = np.concatenate(indices)
            loss_values = np.concatenate(loss_values)
            mean_loss_vals = np.mean(loss_values)
            indices_as = np.argsort(np.abs(loss_values - mean_loss_vals))
            dataset = train_loader.dataset
            upper = int(args.data_proportion * len(dataset))
            dataset.remove(indices[indices_as[upper:]])
            train_loader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        if epoch + 1 == args.sampling_epoch and args.data_proportion != 1 and args.sampling_method == "high":
            indices,loss_values=get_losses(model,train_loader,args)
            indices = np.concatenate(indices)
            loss_values = np.concatenate(loss_values)
            mean_loss_vals = np.mean(loss_values)
            indices_as = np.argsort(loss_values - mean_loss_vals)
            dataset = train_loader.dataset
            upper = int((args.data_proportion*len(dataset)))
            dataset.remove(indices[indices_as[upper:]])
            train_loader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        if epoch + 1 == args.sampling_epoch and args.data_proportion != 1 and args.sampling_method == "low":
            indices, loss_values = get_losses(model, train_loader, args)
            indices = np.concatenate(indices)
            loss_values = np.concatenate(loss_values)
            mean_loss_vals = np.mean(loss_values)
            indices_as = np.argsort(loss_values - mean_loss_vals)
            dataset = train_loader.dataset
            upper = int((1 - args.data_proportion) * len(dataset))
            dataset.remove(indices[indices_as[:upper]])
            train_loader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)



        elif epoch + 1 == args.sampling_epoch and args.data_proportion != 1 and args.sampling_method == "random":

            dataset = train_loader.dataset
            proportion = math.floor(len(dataset)*args.data_proportion)
            indices = np.arange(len(dataset))
            np.random.shuffle(indices)
            dataset.remove(indices[0:-proportion])
            train_loader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


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

    return_model = models.get_mnist_architecture(args).cuda()
    return_model.load_state_dict(best_model)
    return_model.float()
    return_model.eval()

    return return_model

if t.__version__ == "1.7.1":
    t.clamp = clamp