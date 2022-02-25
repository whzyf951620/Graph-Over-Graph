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
    def __init__(self, num_features, num_classes, distanceRate = 1, k = 30):
        super(Model, self).__init__()

        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 1)
        self.conv5 = Conv1d(1, 16, 97, 97)
        self.conv6 = Conv1d(16, 32, 5, 1)
        self.pool = MaxPool1d(2, 2)
        self.classifier_1 = Linear(352, 128)
        self.drop_out = Dropout(0.5)
        self.classifier_2 = Linear(128, num_classes)
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
        k1 = int((1-self.distanceRate) * self.k)
        x = self.SortPooling(x, batch, k1)
        # x = self.SimilarityPooling(x, batch, self.k - k1)
        # x = self.mixPooling(x, batch, k)
        # x = global_sort_pool(x, batch, k=30)
        x = x.view(x.size(0), 1, x.size(-1))
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = self.relu(self.conv6(x))
        x = x.view(x.size(0), -1)
        out = self.relu(self.classifier_1(x))
        out = self.drop_out(out)
        classes = F.log_softmax(self.classifier_2(out), dim=-1)

        return classes

    def computeSimilarity(self, x):
        if len(x.shape) > 2:
            tmp = x.view(x.shape[0], -1)
        else:
            tmp = x

        tmp1 = tmp.unsqueeze(1)
        tmp2 = tmp.unsqueeze(0)
        tmp = torch.pow(tmp1 - tmp2, 2).sum(1).sum(1)
        return tmp.squeeze()
        # for item1 in tmp:
        #     similarity = 0
        #     for item2 in tmp:
        #         similarity += torch.pow(item1 - item2, 2).sum().item()
        #     similarityList.append(similarity)
        # return torch.tensor(similarityList, dtype=float).squeeze()

    # def SimilarityPooling(self, x, batch, k):
    #     similarityNp = self.computeSimilarity(x)
    #     SimilaritysortIndices = np.argsort(similarityNp)
    #     batch_size = int(batch.max()) + 1
    #     num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0, dim_size=batch_size)
    #     cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    #     max_num_nodes = int(num_nodes.max())
    #
    #     # idx = torch.arange(batch.size(0), dtype=torch.long, device=x.device)  #######[0, 1, 2, ..., num_nodes]
    #     # idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
    #     SimilaritysortIndicesBatch = (SimilaritysortIndices - cum_nodes[batch[SimilaritysortIndices]]) + \
    #                             (batch[SimilaritysortIndices] * max_num_nodes)
    #
    #
    #     size = [batch_size * max_num_nodes] + list(x.size())[1:]
    #     out = x.new_full(size, 0)
    #     out[SimilaritysortIndicesBatch] = x[SimilaritysortIndices]
    #     out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])
    def SimilarityPooling(self, x, batch, k):
        batch_x, num_nodes = self.Test_to_dense_batch(x, batch)
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

        batch_x, _ = self.Test_to_dense_batch(x, batch, fill_value)
        B, N, D = batch_x.size()

        _, perm = batch_x[:, :, -1].sort(dim=-1, descending=True) ######sorted by the final feature, i.e., x_4 in the forward function
        arange = torch.arange(B, dtype=torch.long, device=perm.device) * N
        perm = perm + arange.view(-1, 1)

        batch_x = batch_x.view(B * N, D)
        batch_x = batch_x[perm]
        batch_x = batch_x.view(B, N, D)

        if N >= k:
            batch_x = batch_x[:, :k].contiguous()
        else:
            expand_batch_x = batch_x.new_full((B, k - N, D), fill_value)
            batch_x = torch.cat([batch_x, expand_batch_x], dim=1)

        batch_x[batch_x == fill_value] = 0
        x = batch_x.view(B, k * D)

        return x

    def mixPooling(self, x, batch, k):
        k1 = k * (1 - self.distanceRate)
        k2 = k - k1

    def Test_to_dense_batch(self, x: Tensor, batch: Optional[Tensor] = None,
                       fill_value: float = 0., max_num_nodes: Optional[int] = None,
                       batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        r"""Given a sparse batch of node features
        :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}` (with
        :math:`N_i` indicating the number of nodes in graph :math:`i`), creates a
        dense node feature tensor
        :math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}` (with
        :math:`N_{\max} = \max_i^B N_i`).
        In addition, a mask of shape :math:`\mathbf{M} \in \{ 0, 1 \}^{B \times
        N_{\max}}` is returned, holding information about the existence of
        fake-nodes in the dense representation.
        Args:
            x (Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
            batch (LongTensor, optional): Batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
                node to a specific example. Must be ordered. (default: :obj:`None`)
            fill_value (float, optional): The value for invalid entries in the
                resulting dense output tensor. (default: :obj:`0`)
            max_num_nodes (int, optional): The size of the output node dimension.
                (default: :obj:`None`)
            batch_size (int, optional) The batch size. (default: :obj:`None`)
        :rtype: (:class:`Tensor`, :class:`BoolTensor`)
        """
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
