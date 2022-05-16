import pdb

import numpy as np
import torch as t
import scipy
import scipy.stats
import torch.nn.functional as F


def gen_ball(data,max_peturb,norm,max,min,args):

    original_size = data.size()
    data = data.reshape((data.size()[0],-1))

    if norm == "inf":
        delta = (np.random.random(data.shape) - 0.5) * 2
    else:
        xs = scipy.stats.gennorm.rvs(norm,size=data.size())
        denom = np.sum(xs**norm,dim=1)
        denom += np.random.exponential(scale=1,size=original_size[0])
        denom = denom**(1/norm)
        delta = xs / (denom.reshape(-1,1))

    delta = delta.reshape(original_size)*max_peturb
    delta = t.tensor(delta,dtype=t.float)
    if args.device == "gpu":
        delta = delta.cuda()
    data = data.reshape(original_size)

    return t.clamp(data + delta,min,max)

def get_gradients(model,samples,current_class,target_class,args):

    samples.requires_grad = True
    output = model(samples)
    target_classes = output[:,target_class]
    current_classes = output[:,current_class]
    gs = current_classes - target_classes
    diff = t.sum(gs)
    grads = t.autograd.grad(diff,samples)


    return grads[0]

def estimate(S,args):

    weibull_max = scipy.stats.weibull_max
    estimates = []
    for i in range(0,S.size()[1]):
        sample = S[:,i]
        (_,loc,_) = weibull_max.fit(sample.cpu())
        estimates.append(loc)

    to_return = t.from_numpy(np.stack(estimates))
    to_return = to_return.float()
    if args.device == "gpu":
        to_return = to_return.cuda()

    return to_return

def clever_t(model,data,norm,max_perturb,target_class,num_batches,num_samples,max,min,args):

    outputs = model(data)
    current_class= t.argmax(outputs, axis=1)

    different = target_class != current_class
    data_ = data[different]
    current_class_updated = current_class[different]



    S = []
    if norm =="inf":
        q = 1
    else:
        q = norm / (norm - 1)

    for i in range(0,num_batches):
        samples = []
        for j in range(0,num_samples):
            #We generate samples of the L_p ball around the data. Same shape as data
            points = gen_ball(data_,max_perturb,norm,max,min,args)
            #We get the gradients of the data with respect to the difference in class outputs
            gradients = get_gradients(model,points,current_class_updated,target_class,args)
            #We reshape the gradients, such that we can compute the norm down one axis
            gradients = gradients.reshape((data_.shape[0],-1))
            #we add the size of the gradients to our list
            samples.append(t.norm(gradients,p=q,dim=1))

        #These are samples, of size (num_samples,batch_size)

        S.append(t.max(t.stack(samples),dim=0)[0])


    S = t.stack(S)
    #Again, samples of (num_samples, batch size)
    lipchits_estimates = estimate(S,args)
    #Here we add back our classes which had target class equal to predicted class


    lipchits_estimates_normalised = t.ones(size=(data.shape[0],))

    if args.device == "gpu":
        lipchits_estimates_normalised = lipchits_estimates_normalised.cuda()

    lipchits_estimates_normalised[different] = lipchits_estimates



    g_outputs = outputs[:,current_class] - outputs[:,target_class]  + max_perturb
    g_outputs[different] -= max_perturb


    max_perturb = t.tensor(max_perturb)

    if args.device == "gpu":
        max_perturb = max_perturb.cuda()

    return t.minimum(g_outputs / lipchits_estimates_normalised,max_perturb)

def clever_u(model,data,norm,max_perturb,num_batches,num_samples,max,min,args,num_classes=10):

    mle_estimates = []
    for target in range(0,num_classes):
        estimates = clever_t(model,data,norm,max_perturb,target,num_batches,num_samples,max,min,args)
        mle_estimates.append(estimates)

    mle_estimates = t.stack(mle_estimates)

    import pdb
    pdb.set_trace()

    return t.min(mle_estimates,dim=1)[0]






