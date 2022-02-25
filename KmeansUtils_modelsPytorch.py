from __future__ import division, print_function
import torch

def compute_logits(cluster_centers, data):
    cluster_centers = cluster_centers.unsqueeze(1)
    data = data.unsqueeze(2)
    neg_dist = -torch.square(data - cluster_centers).sum(-1)
    return neg_dist

def assign_cluster(cluster_centers, data):
    if len(cluster_centers.shape) == 2 and len(data.shape) == 2:
        cluster_centers = cluster_centers.unsqueeze(0)
        data = data.unsqueeze(0)
    logits = compute_logits(cluster_centers, data)
    logits_shape = logits.shape

    bsize = logits_shape[0]
    ndata = logits_shape[1]
    ncluster = logits_shape[2]

    logits = logits.view(-1, ncluster)
    prob = torch.softmax(logits, dim = -1)
    prob = prob.view(bsize, ndata, ncluster)
    return prob

def update_cluster(data, prob, fix_last_row=False):
    if fix_last_row:
        prob_ = prob[:, :, :-1]
    else:
        prob_ = prob

    prob_sum = prob_.sum(0, keepdim=True)
    # prob_sum += torch.eq(prob_sum, 0).float()
    prob2 = prob_ / prob_sum
    cluster_centers = (data.unsqueeze(1) * prob2.unsqueeze(2)).sum(0)
    if fix_last_row:
        cluster_centers = torch.cat([cluster_centers,torch.zeros_like(cluster_centers[:, 0:1, :])], 1)
    return cluster_centers
