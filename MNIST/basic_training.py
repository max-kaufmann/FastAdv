import torch as t
import torch.nn as nn
import config
import testing
import time
import numpy as np

def train(model, train_dataloader,args):

    if args.device == 'gpu':
        model = model.cuda()
    cross_entropy = nn.CrossEntropyLoss()
    total_correct =1
    epoch_size = 1

    optimiser = t.optim.Adam(model.parameters(), lr=config.lr_max)  # TODO: In paper they set it to lr.max at start, check if that matters
    lr_generator = lambda t: np.interp(t, [0, args.epochs * 2 // 5, args.epochs], [0, config.lr_max, 0])

    start_time = time.time()
    for epoch in range(args.epochs):
        total_loss = 0
        for i,(input,target,ind) in enumerate(train_dataloader):

            lr = lr_generator(epoch + (i+1)/len(train_dataloader))
            optimiser.param_groups[0].update(lr=lr)

            if args.device == 'gpu':
                input,target = input.cuda(),target.cuda()


            optimiser.zero_grad() #This zeroes the gradient. Question: Why is this needed?
            outputs = model(input)
            loss = cross_entropy(outputs,target)
            loss.backward()
            optimiser.step()
            total_loss += loss

        end_time = int(time.time()) - int(start_time)
        testing.log_training(model, epoch, total_loss, total_correct, epoch_size, end_time, optimiser, None, args)

    return model