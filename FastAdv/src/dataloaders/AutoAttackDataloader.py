import torch as t
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import gc
import autoattack
from .  import AdversarialDataloader

class AutoAttackDataloader(AdversarialDataloader.AdversarialDataloader):

    def __init__(self,model,dataloader,args):
        super(AutoAttackDataloader,self).__init__(dataloader)
        device = "cpu" if args.device == "cpu" else "cuda"
        self.args = args
        self.model =model
        self.adversary = autoattack.AutoAttack(model, norm='Linf', eps=args.epsilon, version='standard',device=device)
        self.adversary.attacks_to_run= ['apgd-ce']

    def generate_attack(self,data):
        xs,ys,i = data
        if self.args.device == 'gpu':
            xs,ys = xs.cuda(),ys.cuda()
        self.model.test()
        xs_adv = self.adversary.run_standard_evaluation(xs,ys,bs=self.args.batch_size)
        self.model.train()

        return xs_adv,ys,i