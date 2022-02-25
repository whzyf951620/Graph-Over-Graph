import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, MaxPool1d, Linear, Dropout
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import to_dense_batch

from typing import Optional, Tuple
from torch import Tensor
from torch_scatter import scatter_add


class Model(nn.Module):
    def __init__(self, num_features, num_classes = 1, distanceRate = 0.1, k = 30):
        super(Model, self).__init__()

        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.conv4 = GCNConv(64, 1)
        self.conv5 = Conv1d(1, 64, 193, 193)
        self.conv6 = Conv1d(64, 64, 5, 1)
        self.pool = MaxPool1d(2, 2)
        self.classifier_1 = Linear(704, 1600)
        self.drop_out = Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        self.distanceRate = distanceRate
        self.k = k

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _ = remove_self_loops(edge_index)
        x_1 = torch.tanh(self.conv1(x, edge_index))
        x_2 = torch.tanh(self.conv2(x_1, edge_index))
        x_3 = torch.tanh(self.conv3(x_2, edge_index))
        x_4 = torch.tanh(self.conv4(x_3, edge_index))
        x = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
        # x, residual_x = self.SortPooling(x, batch, self.k)
        # x = self.SimilarityPooling(x, batch, self.k - k1)
        x = self.mixPooling(x, batch)
        # x = global_sort_pool(x, batch, k=30)
        x = x.view(x.size(0), 1, x.size(-1))
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = self.relu(self.conv6(x))
        x = x.view(x.size(0), -1)
        out = self.relu(self.classifier_1(x))
        out = self.drop_out(out)
        # classes = F.log_softmax(self.classifier_2(out), dim=-1)

        return out

    def computeSimilarity(self, x):
        if len(x.shape) > 2:
            tmp = x.view(x.shape[0], -1)
        else:
            tmp = x

        tmp1 = tmp.unsqueeze(1)
        tmp2 = tmp.unsqueeze(0)
        tmp = torch.pow(tmp1 - tmp2, 2).sum(1).sum(1)
        return tmp.squeeze()

    def SimilarityPooling(self, batch_x, batch, k, num_nodes):
        # batch_x, mask= self.Test_to_dense_batch(x, batch)  #######in this function, the batch_x should be provided
        # num_nodes = scatter_add(batch.new_ones(batch_x.size(0)), batch, dim=0, dim_size=int(batch.max()) + 1)
        B, N, D = batch_x.size()
        for index, item in enumerate(batch_x):
            item = batch_x[index]
            tmp = item[:num_nodes[index]]
            similarityNp = self.computeSimilarity(tmp)
            _, SimilaritysortIndices = similarityNp.sort(dim=-1, descending=True)
            batch_x[index][:num_nodes[index]] = item[:num_nodes[index]][SimilaritysortIndices]

        if N >= k:
            batch_x = batch_x[:, :k].contiguous()
        else:
            expand_batch_x = batch_x.new_full((B, k - N, D), 0)
            batch_x = torch.cat([batch_x, expand_batch_x], dim=1)

        x = batch_x.view(B, k * D)
        return x

    def SortPooling(self, x, batch, k):
        fill_value = x.min().item() - 1
        batch_x, num_nodes = self.Test_to_dense_batch(x, batch, fill_value)
        B, N, D = batch_x.size()

        _, perm = batch_x[:, :, -1].sort(dim=-1, descending=True) ######sorted by the final feature, i.e., x_4 in the forward function
        arange = torch.arange(B, dtype=torch.long, device=perm.device) * N
        perm = perm + arange.view(-1, 1)

        batch_x = batch_x.view(B * N, D)
        batch_x = batch_x[perm]
        batch_x = batch_x.view(B, N, D)

        if N > k:
            batch_x_sort = batch_x[:, :k].contiguous()
            residual_x = batch_x[:, k:].contiguous()
        else:
            expand_batch_x = batch_x.new_full((B, k - N, D), fill_value)
            batch_x_sort = torch.cat([batch_x, expand_batch_x], dim=1)
            residual_x = None

        batch_x_sort[batch_x_sort == fill_value] = 0
        x = batch_x_sort.view(B, k * D)

        return x, residual_x, num_nodes

    def mixPooling(self, x, batch):
        k1 = int(self.k * self.distanceRate)
        k2 = self.k - k1

        batch_x_sort, residual_x, num_nodes = self.SortPooling(x, batch, k2)
        if residual_x is not None:
            batch_x_simi = self.SimilarityPooling(residual_x, batch, k1, num_nodes)
        else:
            fill_value = 0.0
            D = x.shape[-1]
            batch_x_simi = batch_x_sort.new_full((batch_x_sort.shape[0], k1 * D), fill_value)
        x = torch.cat([batch_x_sort, batch_x_simi], 1)

        return x

    def Test_to_dense_batch(self, x: Tensor, batch: Optional[Tensor] = None,
                       fill_value: float = 0., max_num_nodes: Optional[int] = None,
                       batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        if batch is None and max_num_nodes is None:
            mask = torch.ones(1, x.size(0), dtype=torch.bool, device=x.device)
            return x.unsqueeze(0), mask

        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        if batch_size is None:
            batch_size = int(batch.max()) + 1

        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0,
                                dim_size=batch_size)
        cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
        if max_num_nodes is None:
            max_num_nodes = int(num_nodes.max())

        idx = torch.arange(batch.size(0), dtype=torch.long, device=x.device) #######[0, 1, 2, ..., num_nodes]
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)

        size = [batch_size * max_num_nodes] + list(x.size())[1:]
        out = x.new_full(size, fill_value)
        out[idx] = x
        out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

        mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool,
                           device=x.device)
        mask[idx] = 1
        mask = mask.view(batch_size, max_num_nodes)

        return out, num_nodes


class RelationNetwork(nn.Module):
    """Graph Construction Module"""

    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1))

        self.fc3 = nn.Linear(2 * 2, 8)
        self.fc4 = nn.Linear(8, 1)

        self.m0 = nn.MaxPool2d(2)  # max-pool without padding
        self.m1 = nn.MaxPool2d(2, padding=1)  # max-pool with padding

    def forward(self, x, rn):
        x = x.view(-1, 64, 5, 5)

        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)  # no relu

        out = out.view(out.size(0), -1)  # bs*1

        return out

class LabelPropagation(nn.Module):
    """Label Propagation"""

    def __init__(self, NumFeatures, NumClasses, config, data_type = 'NCI1', dsitanceRate = 1, k = 30):
        super(LabelPropagation, self).__init__()
        self.NumClasses = NumClasses
        self.NumFeatures = NumFeatures
        self.encoder = Model(NumFeatures)
        self.relation = RelationNetwork()
        self.data_type = data_type
        self.config = config

        self.rn = int(config.get(self.data_type, 'rn'))
        self.alpha = float(config.get(self.data_type, 'alpha'))
        self.TPNK = int(config.get(self.data_type, 'TNPK'))

        if self.rn == 300:  # learned sigma, fixed alpha
            self.alpha = torch.tensor([self.alpha], requires_grad=False)
        elif self.rn == 30:  # learned sigma, learned alpha
            self.alpha = nn.Parameter(torch.tensor([self.alpha]), requires_grad=True)

    def forward(self, data):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        # init
        eps = np.finfo(float).eps  ####无穷小

        num_support = int(self.config.get(self.data_type, 'NShots'))
        # NumunLabeled = int(self.config.get(self.data_type, 'Nunlabeled'))
        num_queries = int(self.config.get(self.data_type, 'NQuery'))

        ys = data.y[:self.NumClasses * num_support]
        ys = ys.unsqueeze(1)
        ys = torch.zeros(ys.shape[0], self.NumClasses).scatter_(1, ys, 1)
        yq_gt = data.y[self.NumClasses * num_support:]
        yq_gt = yq_gt.unsqueeze(1)
        yq_gt = torch.zeros(yq_gt.shape[0], self.NumClasses).scatter_(1, yq_gt, 1)
        # Step1: Embedding
        emb_all = self.encoder(data)
        N, d = emb_all.shape[0], emb_all.shape[1]

        # Step2: Graph Construction
        ## sigmma
        if self.rn in [30, 300]:
            self.sigma = self.relation(emb_all, self.rn)

            ## W
            emb_all = emb_all / (self.sigma + eps)  # N*d
            emb1 = torch.unsqueeze(emb_all, 1)  # N*1*d
            emb2 = torch.unsqueeze(emb_all, 0)  # 1*N*d
            W = ((emb1 - emb2) ** 2).mean(2)  # N*N*d -> N*N
            W = torch.exp(-W / 2)

        ## keep top-k values
        if self.TPNK > 0:
            topk, indices = torch.topk(W, self.TPNK)
            mask = torch.zeros_like(W)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # union, kNN graph
            # mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
            W = W * mask

        ## normalized Laplacian Matrix
        D = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
        S = D1 * W * D2

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        # yu = torch.zeros(self.NumClasses * num_queries, self.NumClasses)
        yu = torch.ones(self.NumClasses * num_queries, self.NumClasses) / self.NumClasses
        # yu = (torch.ones(num_classes*num_queries, num_classes)/num_classes).cuda(0)
        y = torch.cat((ys, yu), 0)
        tmp = torch.inverse(torch.eye(N) - self.alpha * S + eps)
        F = torch.matmul(tmp, y)
        Fq = F[self.NumClasses * num_support:, :]  # query predictions

        # Step4: Cross-Entropy Loss
        ce = nn.CrossEntropyLoss().cuda(0)
        ## both support and query loss
        gt = torch.argmax(torch.cat((ys, yq_gt), 0), 1)
        loss = ce(F, gt)
        ## acc
        predq = torch.argmax(Fq, 1)
        gtq = torch.argmax(yq_gt, 1)
        correct = (predq == gtq).sum()
        total = num_queries * self.NumClasses
        acc = 1.0 * correct.float() / float(total)

        return loss, acc
