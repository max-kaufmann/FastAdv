default_batch_size=128
default_epochs=15
default_epsilon=8./255
default_alpha_fgsm=10./255
default_alpha_pgd=0.00784313725
num_classes=10
class_split = [5000]*10
dataset_length=50000
default_mean = (0.4914, 0.4822, 0.4465)
default_std = (0.2471, 0.2435, 0.2616)


max_vals = 1
min_vals = 0

momentum = 0.9
weight_decay = 5e-4
lr_min = 0.
lr_max = 0.2
test_dataloader = None
model_save_location = "../data/models/CIFAR/"
logger = None