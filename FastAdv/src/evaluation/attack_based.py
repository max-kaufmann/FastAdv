
import torch as t
import torch.nn.functional as F

def get_loss(model,dataloader,args,loss_functtion):
    loss = 0.
    for (xs, ys) in iter(dataloader):
        if args.device == 'gpu':
            xs, ys = xs.cuda(), ys.cuda()
        loss += F.cross_entropy(model(xs),ys)

    return loss

def get_confusion(model,dataloader,args,normalised=True):
    num_classes = args.config.num_classes
    class_scores = t.zeros((num_classes,4))
    for  (xs, ys) in iter(dataloader):
        if args.device == 'gpu':
            xs, ys = xs.cuda(), ys.cuda()
        predictions = t.argmax(model(xs),dim=1)
        correct = predictions == ys
        class_scores[0] += correct
        for i in range(0,len(predictions)):
            tp = t.sum((i == ys) and (predictions == i))
            fn = t.sum((i==ys) and (predictions != i))
            fp = t.sum((i!=ys) and (predictions ==i))
            tn = t.sum((i != ys) and (predictions != i))
            class_scores[0] += tp
            class_scores[1] += fp
            class_scores[2] += tp
            class_scores[3] += fn

    if normalised == True:
        class_scores = class_scores/len(dataloader)
    return class_scores

def compute_recall(confusion,flatten):
    recall= confusion[0]/(confusion[0] + confusion[1])
    if flatten:
        recall = t.mean(recall)
    return recall
def get_recall(model,dataloader,args,flatten=False):
    confusion = get_confusion(model,dataloader,args)
    recall = compute_recall(confusion)
    return recall


def compute_precision(confusion,flatten):
    precision=confusion[0]/(confusion[3]+confusion[0])
    if flatten:
        recall = t.mean(precision)
    return precision

def get_precision(model,dataloader,args,flatten=False):
    confusion = get_confusion(model,dataloader,args)
    recall = compute_precision(confusion)
    if flatten:
        recall = t.mean(recall)
    return recall

def get_f1(model,dataloader,args,flatten=False):
    confusion = get_confusion(model,dataloader,args)
    recall = compute_recall(confusion,flatten)
    precision = compute_precision(confusion,flatten)


    return 2*(recall*precision)/(recall + precision)

def get_score(model,dataloader,args):

    if args.statistic == "f1":
        return get_f1(model,dataloader,args)

    if args.statistic == "recall":
        return get_recall(model,dataloader,args)

    if args.statistic == "confusion":
        return get_confusion(model,dataloader,args)

    if args.statistic == "loss":
        return get_loss(model,dataloader,args)



