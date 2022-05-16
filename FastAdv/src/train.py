#!/home/max/anaconda3/bin/python3.8
# We normalize channel values on a per channel basis, trying our best to sta
import sys
sys.path.append(".")
from repo_config import path
sys.path.append(path)
import torch as t
import importlib
import time
import argparse
import logging
import config
import training_loop as train
import models as models
import importlib
import dataloaders as loaders


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",default="cifar")
    parser.add_argument('--attack', default='RsFgsm')
    parser.add_argument('--device', default='cuda', type=str, choices=['cpu', 'cuda'])
    parser.add_argument('--dropout',default="random")
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument("--metric",default="clever",choices=["clever","attack_dependant"])
    parser.add_argument("--attack",default="autoattack")
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--epsilon', default=None, type=float)
    parser.add_argument('--alpha_fgsm', default=None, type=float)
    parser.add_argument('--alpha_pgd',default=None,type=float)
    parser.add_argument("--checkpoint_location",default="./",type=str)
    parser.add_argument('--pgd_steps', default=7, type=int)
    parser.add_argument('--pgd_restarts_train', default=1, type=int)
    parser.add_argument('--pgd_restarts_test', default=3, type=int)
    parser.add_argument('--pgd_steps_test', default=25, type=int)
    parser.add_argument('--model',default="resnet",type=str,choices=["resnet","small"])
    parser.add_argument('--pgd_steps_train',default=7,type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--dir',default="./",type=str)
    parser.add_argument('--name', default="model", type=str)
    parser.add_argument('--alpha_test', default=0.00784313725, type=float)
    parser.add_argument('--load_from_checkpoint', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--backup_model', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--log_level', default='', type=str)
    parser.add_argument('--verbose_training', default=True,action=argparse.BooleanOptionalAction)
    parser.add_argument('--data_proportion', default=1, type=float)
    parser.add_argument('--log', default="log.txt", type=str)
    parser.add_argument('--early_stopping', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--early_stopping_threshold', default=0.2, type=float)
    parser.add_argument('--evaluate',default=False,action=argparse.BooleanOptionalAction)
    parser.add_argument('--sampling_epoch',default=5,type=int)
    parser.add_argument('--time_log',default="time_log.txt")
    parser.add_argument('--debug',action=argparse.BooleanOptionalAction,default=False)

    return parser.parse_args()

def set_args(args):

    dataset = args.dataset
    args.config=importlib.import_module("config." + dataset + "_config",".")
    default_parameters=["batch_size","epochs","epsilon","alpha_fgsm","alpha_pgd"]

    for p in default_parameters:
        if getattr(args,p) == None:
            setattr(args,p,getattr(args.config,"default_" + p)) #set default parameters

    if args.dir[-1] != "/":
        args.dir += "/"


    logging.basicConfig(filename=args.dir + f"log_{args.name}", filemode='a+', level=logging.DEBUG,
                        format="%(message)s")
    args.config.logger = logging.getLogger(args.name)


def get_dataloader(dataset,model,args):


    pin = True if args.device == "cuda" else False
    base_dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,pin_memory=pin)

    if args.attack == "standard":
        return base_dataloader


    dataloader_name = args.attack + "Dataloader"
    dataloaders = importlib.import_module("dataloaders." + dataloader_name)
    dataloader = getattr(dataloaders,dataloader_name)
    dataloader = dataloader(model,base_dataloader,args)

    return dataloader

def get_train_dataset(args):
    dataset = args.dataset
    models_name = dataset + "_models"
    models = importlib.import_module("models." + models_name,".")
    return models.get_train_dataset(args)

def get_test_dataset(args):
    dataset = args.dataset
    models_name = dataset + "_models"

    models = importlib.import_module("models." + models_name,".")
    return models.get_test_dataset(args)

def get_model(args):
    dataset = args.dataset
    models_name = dataset + "_models"

    models = importlib.import_module("models." + models_name,".")
    return models.get_model(args)


def get_dropout(args):
    dropout = args.dropout
    dropout = importlib.import_module("dropout." + dropout)
    return dropout

def get_optimiser_and_scheduler(model,args):
    dataset = args.dataset
    models_name = dataset + "_models"
    models = importlib.import_module("models." + models_name,".")
    return models.get_optimiser_and_scheduler(model,args)


def main():
    args = get_args()
    set_args(args)
    print(args)

    model = get_model(args)
    train_dataset = get_train_dataset(args)
    test_dataset = get_test_dataset(args)
    train_loader = get_dataloader(train_dataset,model,args)
    test_loader = get_dataloader(test_dataset,model,args)
    dropout_method = get_dropout(args)
    optimiser,scheduler = get_optimiser_and_scheduler(model,args)

    strong_attack = test_loader

    if args.device == 'cuda':
        model = model.cuda()

    model.train()
    start_time = time.time()
    model = train.train(model,train_dataset, train_loader, args, strong_attack, dropout_method,optimiser,scheduler)
    total_time = ((time.time() - start_time) / 60)
    model.float()
    model.eval()

    t.save((model.state_dict(),args.seed),args.dir + args.name + str(args.seed))

    with open(args.dir + args.time_log,"a+") as f:
        f.write(args.name + "," + str(total_time) + "\n")

if __name__ == "__main__":
    main()