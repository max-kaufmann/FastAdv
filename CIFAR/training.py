#!/home/max/anaconda3/bin/python3.8
# We normalize channel values on a per channel basis, trying our best to sta
import torch as t
import attacks
import pgd_training
import models
import active_fgsm
import time
import torch.nn as nn
import argparse
import basic_training
import logging
import config
import fgsm_training
import testing
from torchvision import datasets, transforms
import numpy as np




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--training', default='fgsm', choices=['normal', 'pgd', 'fgsm','fgsm_active'])
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--epsilon_training', default=8./255, type=float)
    parser.add_argument('--epsilon_test', default=8./255, type=float)
    parser.add_argument('--pgd_steps', default=7, type=int)
    parser.add_argument('--alpha', default=10./255, type=float)
    parser.add_argument('--device', default='gpu', type=str, choices=['cpu', 'gpu'])
    parser.add_argument('--pgd_restarts_train', default=1, type=int)
    parser.add_argument('--pgd_restarts_test', default=3, type=int)
    parser.add_argument('--pgd_steps_test', default=25, type=int)
    parser.add_argument('--model',default="resnet",type=str,choices=["resnet","small"])
    parser.add_argument('--pgd_steps_train',default=7,type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--dir',default="./",type=str)
    parser.add_argument('--name', default=None, type=str)
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
    parser.add_argument('--sampling_method',default='drop_extremes',choices=["drop_extremes","random","pgd-distance","low"])
    parser.add_argument('--time_log',default="time_log.txt")
    parser.add_argument('--debug',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--systematic_sampling', action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def set_args(args):

    if args.dir[-1] != "/":
        args.dir += "/"

    if args.name is None:
        args.name = f"CIFAR_{args.training}_e{args.epochs}_b{args.batch_size}_p_{args.data_proportion}"

    if args.seed is not None:
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)

    if args.device == "gpu":
        config.mn = config.mn.cuda()
        config.std=config.std.cuda()
        config.max_vals = config.max_vals.cuda()
        config.min_vals = config.min_vals.cuda()

    args.epsilon_training = args.epsilon_training/config.std
    args.epsilon_test = args.epsilon_test/config.std
    args.alpha = args.alpha/config.std
    args.alpha_test = args.alpha_test/config.std

    logging.basicConfig(filename=args.dir + f"log_{args.name}", filemode='a+', level=logging.DEBUG,
                        format="%(message)s")
    config.logger = logging.getLogger(args.name)

    config.test_dataloader = models.get_cifar10_test_dataloader(args)
    config.args = args

    # TODO: Add model name, add saving model, add actually training the model, logging correctly


def train(model, dataloader, args, saved_model=None, type='pgd'):
    if type == 'pgd':
        model = pgd_training.train(model, dataloader, args)
    elif type == 'fgsm':
        model = fgsm_training.train(model, dataloader, args)
    elif type == 'normal':
        model = basic_training.train(model, dataloader, args)
    elif type == "fgsm_active":
        model = active_fgsm.train(model,dataloader,args)
    else:
        raise ValueError()

    return model


def main():
    args = get_args()
    set_args(args)
    print(args)

    dataloader = models.get_cifar10_train_dataloader(args)

    model_data = None
    if args.load_from_checkpoint:
        model_data = t.load(f"checkpoint_" + args.name)
        model = model_data['model']
        config.num_epochs = args.num_epochs - model_data['epoch']
    else:
        model = models.get_cifar10_architecture(args)

    if args.device == 'gpu':
        model = model.cuda()

    model.train()
    start_time = time.time()
    model = train(model, dataloader, args, model_data, args.training)
    total_time = ((time.time() - start_time) / 60)
    model.float()
    model.eval()

    t.save(model.state_dict(),args.dir + args.name)

    with open(args.dir + args.time_log,"a+") as f:
        f.write(args.name + "," + str(total_time) + "\n")

    if args.evaluate:
        attack = attacks.pgd_attack(model, args.epsilon_test, args.alpha_test, args.pgd_steps_test,
                                       args.pgd_restarts_test, args)

        accuracy = testing.get_accuracy(model, iter(config.test_dataloader), args)
        config.logger.info(f"Accuracy: {accuracy}")
        adv_accuracy = testing.get_accuracy(model, attack.iterator_from_dataloader(config.test_dataloader), args)

        if args.log is not None:
            with open(args.dir + args.log, "a+") as f:
                f.write(f"{args.name},{args.data_proportion},{accuracy},{adv_accuracy}\n")

        config.logger.info(f"Adversarial Accuracy: {adv_accuracy}")
        logging.shutdown()


if __name__ == "__main__":
    main()