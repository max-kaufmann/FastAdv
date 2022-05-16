import torch.nn as nn
import models
import config
import torch as t
import numpy as np
#TODO: Make notes on

"""We accumulate gradients so that we can do multiple pass throught and then update on them all"""

def main():
    train_loader = models.get_cifar10_train_dataloader()
    return train(models.get_cifar10_architecture(),train_loader)

def train(model,train_loader,args):

    print("PGD ADVERSARIAL TRAINING STARTING")

    model.train()
    loss_function = nn.CrossEntropyLoss()

    #Here, we have our optimiser (thing doing the optimisation on the neural network)
    optimiser = t.optim.Adam(model.parameters(), lr=config.lr_max)  # TODO: In paper they set it to lr.max at start, check if that matters
    lr_generator = lambda t: np.interp(t, [0, args.epochs * 2 // 5, args.epochs], [0, config.lr_max, 0])

    for epoch in range(0, args.epochs):
        for i,(xs,ys,ind) in enumerate(iter(train_loader)):

            lr = lr_generator(epoch + (i+1)/len(train_loader))
            optimiser.param_groups[0].update(lr=lr)

            if args.device =='gpu':
                xs,ys = xs.cuda(), ys.cuda()

            #Randomly initalise our starting gradient
            if args.device == 'cpu':
                delta = (0.5 - t.rand(xs.shape,device='cpu'))*2*args.epsilon_training #TODO: Add "device = 'cuda'" and check this is right.
            if args.device == 'gpu':
                delta = (0.5 - t.rand(xs.shape,device='cuda'))*2*args.epsilon_training

            delta = t.clamp(delta, config.min_vals - xs, config.max_vals - xs)


            for ii in range(0, args.pgd_steps_train):
                delta.requires_grad = True
                output = model(xs + delta)
                loss = loss_function(output, ys)
                optimiser.zero_grad()
                loss.backward()
                grad = delta.grad.detach()
                I = output.max(1)[1] == ys
                delta.data[I] = t.clamp(delta + args.alpha * t.sign(grad), -args.epsilon_training, args.epsilon_training)[I]
                delta.data[I] = t.max(t.min(1 - xs, delta.data), 0 - xs)[I]
            delta=delta.detach()

            optimiser.zero_grad()
            out = model(xs + delta)
            loss = loss_function(out,ys)
            loss.backward()
            optimiser.step()

    return model

if __name__ == "__main__":
    main()