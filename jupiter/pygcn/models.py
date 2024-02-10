import torch.nn as nn
import torch.nn.functional as F
from .layers import GCNConv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, laplac_size=None):
        super(GCN, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid, laplac_size)
        self.gc2 = GCNConv(nhid, nclass, laplac_size)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
                
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
