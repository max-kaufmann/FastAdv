#!/home/max/anaconda3/bin/python3.8
import sys
sys.path.append(".")
from repo_config import path
sys.path.append(path)
import torch as t
import torch.nn as nn
import argparse
import models
import config
import io
import torchvision.transforms as transforms
import sys
import numpy as np
import config
import importlib
import train
import evaluation.clever as clever
import evaluation.attack_based

def get_accuracy(model,dataloader_iterator,args):

    print("Getting Accuracy")

    total_correct = 0
    dataloader_length = 0
    for i,(xs,ys) in enumerate(dataloader_iterator):

        if args.device == 'cuda':
            xs,ys = xs.cuda(),ys.cuda()

        predictions = model(xs)
        total_correct += (t.argmax(predictions,dim=1) == ys).sum()
        dataloader_length += len(ys)


    return total_correct/(dataloader_length)


def log_training(model,dataloader,epoch,total_loss,total_correct,epoch_size,end_time,optimiser,scheduler,args):

    if epoch == 0:
        print("Epoch num,avg. loss, accuracy,end_time,test_loss,tess_accuracy")

    data_output = f"{epoch},{total_loss / epoch_size},{total_correct / epoch_size * args.batch_size},{end_time},"

    if (epoch % 5 == 1) and args.backup_model:
        print(args.backup_model)
        t.save({'model':model,'epoch':epoch,'optim':optimiser,'scheduler':scheduler},args.checkpoint_location + args.name)

    if "b" in args.log_level and (epoch % 5 == 0):
        print("Dataset Test")
        test_acc = get_accuracy(model, iter(dataloader))
        data_output += str(test_acc)

    data_output += ','


    if "a" in args.log_level and (epoch % 5 == 0):
        print("Adversarial Test")
        test_acc_adv = get_accuracy(model, args.config.test_datlaoader())
        data_output += str(test_acc_adv)

    args.config.logger.info(data_output)

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--epsilon', default=8./255, type=float)
    parser.add_argument('--steps', default=25, type=int)
    parser.add_argument('--alpha', default=2./255, type=float)
    parser.add_argument('--device', default='cuda', type=str, choices=['cpu', 'cuda'])
    parser.add_argument('--restarts', default=5, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--name', type=str)
    parser.add_argument("--dataset", default="cifar")
    parser.add_argument('--attack',type=str,default="AutoAttack")
    parser.add_argument('--model',type=str,default="resnet")
    parser.add_argument("--metric",type=str,default="attack_based",choices=["clever","attack_based","standard_accuracy"])
    parser.add_argument('--dir',default="./",type=str)
    parser.add_argument('--log',type=str, default="log.txt")
    parser.add_argument('--clever_batches',default=400,type=int)
    parser.add_argument('--clever_samples',default=650,type=int)
    parser.add_argument("--norm",default="inf")
    parser.add_argument('--clever_num_images',default=40,type=int)
    parser.add_argument('--normal_accuracy', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--clever_perturb',type = float)
    parser.add_argument('--statistic',type=str)

    return parser.parse_args()

def set_args(args):

    args.config=importlib.import_module("config." + args.dataset + "_config" )
    if args.metric =="clever":
        args.attack = "standard"
        args.batch_size = args.clever_num_images

    if not (args.dir[-1] == '/'):
        args.dir += '/'


    if not (args.seed is None):
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)

    config.args = args


def main(args):
    set_args(args)
    if not(args.seed is None):
        t.manual_seed(args.seed)
        np.random.seed(args.seed)

    model = train.get_model(args)
    state_dict,_ = t.load(args.dir + args.name) #returns model and its seed
    model.load_state_dict(state_dict)
    dataset = train.get_test_dataset(args)
    dataloader = train.get_dataloader(dataset,model,args)

    if args.device == 'cuda':
        model = model.cuda()
    model.eval()

    if args.metric== "attack_based":
        metric = evaluation.attack_based.get_score(model,dataloader,args)
    elif args.metric == "clever":
        data=next(iter(dataloader))[0]
        if args.device == "cuda":
            data = data.cuda()
        metric = evaluation.clever.clever_u(model,data,args)
        print(metric)
    elif args.metric == "standard_accuracy":
        metric = get_accuracy(model,train.get_dataloader(dataset,model,args))


    with open(args.dir + args.log, mode="a+") as f:
        f.seek(0)
        if f.readlines() == []:
            f.write("name,metric,seed")
        f.seek(0,io.SEEK_END)
        f.write(f"{args.name},{metric},{args.seed}\n")

if __name__ == "__main__":
    args = get_args()
    main(args)
