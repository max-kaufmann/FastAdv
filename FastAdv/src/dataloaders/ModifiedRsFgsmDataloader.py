from . import AdversarialDataloader
import FastAdv.src.config.mnist_config as config
import torch as t
import torch.nn as nn
import torch.nn.functional as F

class ModifiedRsFgsmDataloader(AdversarialDataloader.AdversarialDataloader):

    def __init__(self, model,dataloader, args):
        super(ModifiedRsFgsmDataloader,self).__init__(dataloader)
        self.model = model
        self.epsilon = args.epsilon
        self.alpha = args.alpha_fgsm
        self.args = args

    def generate_attack(self, data):

        X,y,i = data
        args=self.args

        delta = t.zeros_like(X,device=args)
        if self.args.device == 'cuda':
            X, y = X.cuda(), y.cuda()

        delta.uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True
        output = self.model(X + delta[:X.size(0)])
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = t.clamp(delta + (self.alpha * t.sign(grad)), -self.epsilon, self.epsilon)
        delta.data[:X.size(0)] = t.clamp(delta[:X.size(0)], config.min_vals - X, config.max_vals - X)
        delta = delta.detach()



        return X+ delta, y, i