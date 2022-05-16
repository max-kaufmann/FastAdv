import torch as t
default_mean = (0.4914, 0.4822, 0.4465)
default_std = (0.2471, 0.2435, 0.2616)
num_classes=10
class_split = [5000]*10
mn = t.tensor(default_mean).view(3, 1, 1)  # .cuda()
std = t.tensor(default_std).view(3, 1, 1)  # .cuda()
args = None
early_stopping_steps = 10
early_stopping_restarts = 3

max_vals = ((1 - mn) / std)
min_vals = ((0 - mn) / std)

momentum = 0.9
weight_decay = 5e-4
lr_min = 0.
lr_max = 0.2
test_dataloader = None
model_save_location = "../data/models/CIFAR/"
logger = None