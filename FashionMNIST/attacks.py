import torch as t
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import models
import config
import gc
import autoattack
from fgsm_training import clamp

class ga_pgd_attack():

    def __init__(self, model, epsilon, alpha, num_steps, num_restarts, args):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.num_restarts = num_restarts
        self.args = args

    def generate_attack(self, data):

        device = 'cuda' if self.args.device == 'gpu' else 'cpu'

        xs, ys, inds = data

        if self.args.device == 'gpu':
            xs, ys = xs.cuda(), ys.cuda()

        curr_ys = t.argmax(self.model(xs),dim=1)
        ks = t.zeros(size=curr_ys.shape,device=device)

        loss_function = nn.CrossEntropyLoss()

        max_loss = t.zeros(
            ys.shape[0],device=device)  # .cuda() # This stores the best loss we have found so far for each memeber of our minibatch
        max_delta = t.zeros(xs.shape,device=device)  # .cuda() #This stores which perturbations were the best

        if self.args.device == 'gpu':
            max_loss, max_delta= max_loss.cuda(), max_delta.cuda()

        for r in range(self.num_restarts):

            delta = t.zeros(xs.shape, device=device)

            for j in range(0, len(self.epsilon)):
                delta[:, j, :, :].uniform_(-self.epsilon[j][0][0].item(), self.epsilon[j][0][0].item())
            delta = t.clamp(delta,config.min_vals -xs,config.max_vals-xs)

            curr_ks = t.zeros(curr_ys.size(),device=device)
            for i in range(self.num_steps):

                curr_ys = t.argmax(self.model(xs), dim=1)
                curr_ks = curr_ks + (curr_ys == ys)  # count number of iterations needed to reach the adversarial example

                delta.requires_grad = True
                output = self.model(xs + delta)
                loss = loss_function(output, ys)
                loss.backward()
                grad = delta.grad.detach()

                delta = delta + self.alpha * t.sign(grad)
                delta = clamp(delta, -self.epsilon, self.epsilon)
                delta = clamp(delta, config.min_vals - xs, config.max_vals - xs)
                delta = delta.detach()

            ks = t.minimum(ks,curr_ks)

            per_batch_loss = F.cross_entropy(self.model(xs + delta), ys,reduction='none').detach()
            max_delta[per_batch_loss >= max_loss] = delta[per_batch_loss >= max_loss]
            max_loss = t.max(max_loss, per_batch_loss)




        return xs + max_delta, inds, ks

    def iterator_from_dataloader(self, dataloader):

        class AdversarialIterator:
            def __init__(s, dataloader):
                s.dataloader = dataloader
                s.dataloader_iterator = iter(dataloader)

            def __iter__(s):
                return s

            def __len__(s):
                return len(s.dataloader)

            def __next__(s):
                current_batch = next(s.dataloader_iterator)
                return self.generate_attack(current_batch)

        return AdversarialIterator(dataloader)

class Auto_attack():



    def __init__(self,model,args):
        self.model = nn.Sequential(transforms.Normalize(config.mn,config.std),model)
        self.args = args
        self.adversary = autoattack.AutoAttack(model, norm='Linf', eps=args.epsilon_training,version='standard')
        self.adversary.attacks_to_run= ['apgd-ce']

    def generate_attack(self,data):
        xs,ys,i_s = data
        if self.args.device == 'gpu':
            xs,ys = xs.cuda(),ys.cuda()
        xs_adv = self.adversary.run_standard_evaluation(xs,ys,bs=self.args.batch_size)
        return xs_adv,ys,i_s

    def iterator_from_dataloader(self, dataloader):

        class AdversarialIterator:
            def __init__(s, dataloader):
                s.dataloader = dataloader
                s.dataloader_iterator = iter(dataloader)

            def __iter__(s):
                return s

            def __len__(s):
                return len(s.dataloader)

            def __next__(s):
                current_batch = next(s.dataloader_iterator)
                return self.generate_attack(current_batch)

        return AdversarialIterator(dataloader)
class pgd_attack():

    def __init__(self, model, epsilon, alpha, num_steps, num_restarts, args):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.num_restarts = num_restarts
        self.args = args

    def generate_attack(self, data):
        xs, ys = data

        if self.args.device == 'gpu':
            xs, ys = xs.cuda(), ys.cuda()

        loss_function = nn.CrossEntropyLoss()

        max_loss = t.zeros(
            ys.shape[0])  # .cuda() # This stores the best loss we have found so far for each memeber of our minibatch
        max_delta = t.zeros(xs.shape)  # .cuda() #This stores which perturbations were the best

        if self.args.device == 'gpu':
            max_loss, max_delta = max_loss.cuda(), max_delta.cuda()

        for r in range(self.num_restarts):


            device = 'cuda' if self.args.device == 'gpu' else 'cpu'
            delta = t.zeros(xs.shape, device=device)

            for j in range(0, len(self.epsilon)):
                delta[:, j, :, :].uniform_(-self.epsilon[j][0][0].item(), self.epsilon[j][0][0].item())
            delta = t.clamp(delta,config.min_vals -xs,config.max_vals-xs)

            for i in range(self.num_steps):
                delta.requires_grad = True
                output = self.model(xs + delta)
                loss = loss_function(output, ys)
                loss.backward()
                grad = delta.grad.detach()

                delta = delta + self.alpha * t.sign(grad)
                delta = clamp(delta, -self.epsilon, self.epsilon)
                delta = clamp(delta, config.min_vals - xs, config.max_vals - xs)
                delta = delta.detach()

            per_batch_loss = F.cross_entropy(self.model(xs + delta), ys,reduction='none').detach()
            max_delta[per_batch_loss >= max_loss] = delta[per_batch_loss >= max_loss]
            max_loss = t.max(max_loss, per_batch_loss)




        return xs + max_delta, ys

    def iterator_from_dataloader(self, dataloader):

        class AdversarialIterator:
            def __init__(s, dataloader):
                s.dataloader = dataloader
                s.dataloader_iterator = iter(dataloader)

            def __iter__(s):
                return s

            def __len__(s):
                return len(s.dataloader)

            def __next__(s):
                current_batch = next(s.dataloader_iterator)
                return self.generate_attack(current_batch)

        return AdversarialIterator(dataloader)