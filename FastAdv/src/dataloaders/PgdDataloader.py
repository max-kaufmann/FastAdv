from .import AdversarialDataloader
import FastAdv.src.config.mnist_config as config
import torch as t
import torch.nn as nn
import torch.nn.functional as F

class PgdDataloader(AdversarialDataloader.AdversarialDataloader):

    def __init__(self,model,dataloader, args):
        super(PgdDataloader,self).__init__(dataloader)
        self.model = model
        self.epsilon = args.epsilon
        self.alpha = args.alpha_pgd
        self.num_steps = args.pgd_num_steps
        self.num_restarts = args.num_restarts
        self.args = args

    def generate_attack(self, data):
        xs, ys,i = data

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
            delta = t.clamp(delta, args.config.min_vals - xs, args.config.max_vals - xs)

            for i in range(self.num_steps):
                delta.requires_grad = True
                output = self.model(xs + delta)
                loss = loss_function(output, ys)
                loss.backward()
                grad = delta.grad.detach()

                delta = delta + self.alpha * t.sign(grad)
                delta = t.clamp(delta, -self.epsilon, self.epsilon)
                delta = t.clamp(delta, args.config.min_vals - xs, args.config.max_vals - xs)
                delta = delta.detach()

            per_batch_loss = F.cross_entropy(self.model(xs + delta), ys, reduction='none').detach()
            max_delta[per_batch_loss >= max_loss] = delta[per_batch_loss >= max_loss]
            max_loss = t.max(max_loss, per_batch_loss)

        return xs + max_delta, ys,i
