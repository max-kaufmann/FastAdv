#!/home/max/anaconda3/bin/python3.8
import attacks
import torch as t
import torch.nn as nn
import argparse
import models
import torch.nn.functional as F
import math
import time
import config
import autoattack
import io
import torchvision.transforms as transforms
import sys
sys.path.append(f"{__file__[:10]}/..")
print(__file__)
import numpy as np
#import Tests.CLEVER as CLEVER

def get_accuracy(model,dataloader_iterator,args):

    print("Getting Accuracy")

    total_correct = 0
    dataloader_length = 0
    for i,(xs,ys,_) in enumerate(dataloader_iterator):

        if args.device == 'gpu':
            xs,ys = xs.cuda(),ys.cuda()

        predictions = model(xs)
        total_correct += (t.argmax(predictions,dim=1) == ys).sum()
        dataloader_length += len(ys)

    return total_correct/(dataloader_length)


def log_training(model,epoch,total_loss,total_correct,epoch_size,end_time,optimiser,scheduler,args):

    if epoch == 0:
        print("Epoch num,avg. loss, accuracy,end_time,test_loss,tess_accuracy")

    data_output = f"{epoch},{total_loss / epoch_size},{total_correct / epoch_size * args.batch_size},{end_time},"

    if (epoch % 5 == 1) and args.backup_model:
        print(args.backup_model)
        t.save({'model':model,'epoch':epoch,'optim':optimiser,'scheduler':scheduler},f"../data/models/checkpoint_" + args.name)

    if "b" in args.log_level and (epoch % 5 == 0):
        print("Dataset Test")
        test_acc = get_accuracy(model, iter(config.test_dataloader))
        data_output += str(test_acc)

    data_output += ','


    if "a" in args.log_level and (epoch % 5 == 0):
        print("Adversarial Test")
        attack = attacks.pgd_attack(model, args.epsilon_testing, args.alpha_test, args.pgd_steps_test, args.pgd_restarts_test)
        test_acc_adv = get_accuracy(model, attack.iterator_from_dataloader(config.test_dataloader))
        data_output += str(test_acc_adv)


    config.logger.info(data_output)

def compute_clever(model,args):
    from art.metrics import clever
    n = iter(config.test_dataloader)
    x,y,i = next(n)
    values = []
    for i in x:
        values.append(clever(model,i.numpy(),args.clever_batches,args.clever_samples,args.epsilon,np.inf))
    return np.array(values)


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--steps', default=25, type=int)
    parser.add_argument('--alpha', default=0.01, type=float)
    parser.add_argument('--device', default='gpu', type=str, choices=['cpu', 'gpu'])
    parser.add_argument('--restarts', default=5, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--name', type=str)
    parser.add_argument('--model',type=str,default="resnet")
    parser.add_argument('--dir',default="./",type=str)
    parser.add_argument('--log',type=str, default="log.txt")
    parser.add_argument('--test',default="pgd",choices=["pgd","autoattack","clever"])
    parser.add_argument('--clever_batches',default=400,type=int)
    parser.add_argument('--clever_samples',default=650,type=int)
    parser.add_argument('--clever_num_images',default=32,type=int)
    parser.add_argument('--normal_accuracy', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--clever_perturb',default = 5./255,type = float)

    return parser.parse_args()

def set_args(args):
    args.epsilon_training = args.epsilon

    if not (args.dir[-1] == '/'):
        args.dir += '/'


    if not (args.seed is None):
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)


    if args.test == "clever":
        args.batch_size = args.clever_num_images



def main():
    args = get_args()
    set_args(args)
    if not(args.seed is None):
        t.manual_seed(args.seed)
        np.random.seed(args.seed)

    config.test_dataloader = models.get_mnist_test_dataloader(args)

    model = models.get_mnist_architecture(args)
    model.load_state_dict(t.load(args.dir + args.name))

    if args.device == 'gpu':
        model = model.cuda()
        
    model.eval()

    if args.normal_accuracy:
        test_acc = get_accuracy(model,iter(config.test_dataloader),args)
    else:
        test_acc = 0.

    if args.test == "autoattack":
        attack = attacks.Auto_attack(model,args)
        adv_test_acc = get_accuracy(model,attack.iterator_from_dataloader(config.test_dataloader),args)
    elif args.test == "pgd":
        attack = attacks.pgd_attack(model, args.epsilon, args.alpha, args.steps, args.restarts, args)
        adv_test_acc = get_accuracy(model,attack.iterator_from_dataloader(config.test_dataloader),args)
    elif args.test == "clever":
        batch = next(iter(config.test_dataloader))[0]
        if args.device == "gpu":
            batch = batch.cuda()
        from art.estimators.classification import PyTorchClassifier
        m = PyTorchClassifier(model,F.cross_entropy,(28,28,3),10)
        vals=compute_clever(m,args)
        print(vals)
        t.save(vals,args.dir[-5:] + "vals")
        mins=np.min(vals,axis=1)
        print(mins)
        clever_score=np.sum(mins)/len(mins)


    if args.test == "autoattack" or args.test == "pgd":
        with open(args.dir + args.log, mode="a+") as f:
            f.seek(0)
            if f.readlines() == []:
                f.write("name,normal_acc,adv_acc\n")
            f.seek(0,io.SEEK_END)
            f.write(f"{args.name},{test_acc},{adv_test_acc}\n")

        print("Adversarial:" + str(adv_test_acc))
        print("Normal: " + str(test_acc))

    elif args.test == "clever":
        with open(args.dir + args.log, mode="a+") as f:
            f.seek(0)
            if f.readlines() == []:
                f.write("name,clever_score\n")
            f.seek(0, io.SEEK_END)
            f.write(f"{args.name},{clever_score}\n")

        print("Normal: " + str(test_acc))
        print("CLEVER: " + str(clever_score))



if __name__ == "__main__":
    main()