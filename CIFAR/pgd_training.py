import torch.nn as nn
import models
import config
import torch as t

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
    optimiser = t.optim.SGD(model.parameters(), lr = config.lr_min, momentum= config.momentum, weight_decay=config.weight_decay ) #TODO: In paper they set it to lr.max at start, check if that matters
    scheduler = t.optim.lr_scheduler.CyclicLR(optimiser, base_lr=config.lr_min, max_lr=config.lr_max, step_size_up=args.epochs * len(train_loader) / 2., step_size_down=args.epochs * len(train_loader) / 2)
    for epoch in range(0, args.epochs):
        for i,(xs,ys,ind) in enumerate(iter(train_loader)):

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
                loss = loss_function(output,ys)
                grad = t.autograd.grad(loss,delta)[0]
                delta = t.clamp(delta + args.alpha * t.sign(grad), -args.epsilon_training, args.epsilon_training) #TODO: Why not find minimal thing here, like you did in the attack
                delta = t.clamp(delta, config.min_vals - xs,  config.max_vals - xs)
                delta = delta.detach()

            optimiser.zero_grad()
            out = model(xs + delta)
            loss = loss_function(out,ys)
            loss.backward()
            optimiser.step()
            scheduler.step()

    return model

if __name__ == "__main__":
    main()