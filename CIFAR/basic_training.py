import torch as t
import torch.nn as nn
import config
import testing
import time

def train(model, train_dataloader,args):

    if args.device == 'gpu':
        model = model.cuda()
    cross_entropy = nn.CrossEntropyLoss()
    total_correct =1
    epoch_size = 1

    optimiser = t.optim.SGD(model.parameters(), lr = config.lr_min, momentum= config.momentum, weight_decay=config.weight_decay ) #TODO: In paper they set it to lr.max at start, check if that matters
    scheduler = t.optim.lr_scheduler.CyclicLR(optimiser, base_lr=config.lr_min, max_lr=config.lr_max, step_size_up=args.epochs* len(train_dataloader) / 2., step_size_down=args.epochs* len(train_dataloader) / 2.)

    start_time = time.time()
    for epoch in range(args.epochs):
        total_loss = 0
        for i,(input,target,ind) in enumerate(train_dataloader):

            if args.device == 'gpu':
                input,target = input.cuda(),target.cuda()


            optimiser.zero_grad() #This zeroes the gradient. Question: Why is this needed?
            outputs = model(input)
            loss = cross_entropy(outputs,target)
            loss.backward()
            optimiser.step()
            scheduler.step()
            total_loss += loss

        end_time = int(time.time()) - int(start_time)
        testing.log_training(model, epoch, total_loss, total_correct, epoch_size, end_time, optimiser, scheduler, args)

    return model