import torch
from torch.nn import functional as F
from torch.nn.modules import Module
from modelsPytorch.KmeansUtils import assign_cluster, update_cluster

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target, input_unlabeled = None):
        return prototypical_loss(input, target, self.n_support, input_unlabeled)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support, input_unlabeled = None):
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    if input_unlabeled is not None:
        input_unlabeled = input_unlabeled.to('cpu')

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = len(target_cpu) // n_classes - n_support

    support_idxs = []
    for i in range(n_classes):
        support_idxs.append(supp_idxs(i))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

    if input_unlabeled is not None:
        prob_unlabel = assign_cluster(prototypes, input_unlabeled).squeeze()
        target_cpu = target_cpu.unsqueeze(1)
        target_onehot = torch.zeros(target_cpu.shape[0], torch.max(target_cpu) + 1).scatter_(1,target_cpu,1)
        prob_all = torch.cat([target_onehot, prob_unlabel], 0)
        prob_all.detach()
        prototypes = update_cluster(torch.cat([input, input_unlabeled], 0), prob_all)

    query_idxs = torch.LongTensor(torch.arange(n_classes*n_support, target_cpu.shape[0]))

    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = target_cpu[query_idxs].long().unsqueeze(1)
    log_p_y = log_p_y.view(-1, log_p_y.shape[2])
    if input_unlabeled is not None:
        target_inds = target_inds.squeeze(1)
    loss_val = -log_p_y.gather(1, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(1)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val

