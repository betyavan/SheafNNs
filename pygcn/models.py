import torch.nn as nn
import torch.nn.functional as F
from .layers import GCNConv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, sheaf_laplacian=None):
        super(GCN, self).__init__()

        self.gc1 = GCNConv(nfeat, nclass, sheaf_laplacian)
        # self.gc2 = GCNConv(nhid, nclass)
        # self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
