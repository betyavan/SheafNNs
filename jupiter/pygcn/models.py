import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution
from .sheaf import *


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        
        self.transform_features = transform_features

    def forward(self, x, adj):
        
        x = self.transform_features(x)
        
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
