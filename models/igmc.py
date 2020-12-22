import torch
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import dropout_adj
import torch.nn.functional as F
from torch.nn import Linear
import time


class IGMCModel(torch.nn.Module):
    def __init__(self, num_features, dim=[32, 32, 32, 32], num_rel=5, num_base=4, dropout=0.2):
        super(IGMCModel, self).__init__()
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(RGCNConv(num_features, dim[0], num_rel, num_base))
        for i in range(0, len(dim)-1):
            self.convs.append(RGCNConv(dim[i], dim[i+1], num_relations=num_rel, num_bases=num_base))

        self.lin1 = Linear(2*sum(dim), 128)
        self.lin2 = Linear(128, 1)

    @classmethod
    def code(cls):
        return 'igmc'

    def forward(self, data):
        # start = time.time()
        edge_index, edge_type, x = data.edge_index, data.edge_type, data.x
        edge_index, edge_type = dropout_adj(
            edge_index, edge_type,
            p=self.dropout,
            num_nodes=len(x),
            training=self.training
        )

        h_i = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index, edge_type))
            h_i.append(x)
        h_i = torch.cat(h_i, dim=1)

        u = data.x[:, 0] == 1
        i = data.x[:, 0] == 1
        x = torch.cat([h_i[u], h_i[i]], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # end = time.time()
        # print("forward took:")
        return x[:, 0]
