import torch as t
max_vals = 1
min_vals = 0
mn = 0
std = 1
momentum = 0.9
weight_decay = 5e-4
lr_min = 0.001
early_stopping_steps = 10
early_stopping_restarts = 3
num_classes=10
class_split = [5000]*10
lr_max = 0.1
test_dataloader = None
logger = ""
model_save_location = "../data/models/MNIST/"
steps_early_stopping = 10
restarts_early_stopping = 3


