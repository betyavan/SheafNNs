import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GCNConv
from .utils import accuracy
import pytorch_lightning as pl



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
    


class GCN_module(pl.LightningModule):
    def __init__(self, n_feat, n_hid, n_class,
                       dropout, sheaf_laplacian,
                       learning_rate=0.01, weight_decay=5e-4):
        super(GCN_module, self).__init__()
        self.model = GCN(nfeat=n_feat,
                         nhid=n_hid,
                         nclass=n_class,
                         dropout=dropout,
                         sheaf_laplacian=sheaf_laplacian)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = F.nll_loss
        
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                           weight_decay=self.weight_decay)
        return optim
    
    def forward(self, features, adj):
        return self.model(features, adj)
    
    
    def training_step(self, train_batch, batch_idx):
        # x, edge_index, y, train_mask, val_mask = train_batch
        # x, edge_index, y = x[0], edge_index[0], y[0]
        # train_mask, val_mask = train_mask[0], val_mask[0]
        
        output = self.model(train_batch.ndata["feat"], train_batch.edges_tensor)
        
        loss_train = self.loss_fn(
            output[train_batch.ndata["train_mask"]], train_batch.ndata["label"][train_batch.ndata["train_mask"]]
        )
        acc_train = accuracy(
            output[train_batch.ndata["train_mask"]], train_batch.ndata["label"][train_batch.ndata["train_mask"]]
        )
        
        loss_val = self.loss_fn(
            output[train_batch.ndata["test_mask"]], train_batch.ndata["label"][train_batch.ndata["test_mask"]]
        )
        acc_val = accuracy(
            output[train_batch.ndata["test_mask"]], train_batch.ndata["label"][train_batch.ndata["test_mask"]]
        )
        
        self.log("loss_train", loss_train, prog_bar=True)
        self.log("acc_train", acc_train, prog_bar=True)
        
        self.log("loss_val", loss_val, prog_bar=True)
        self.log("acc_val", acc_val, prog_bar=True)
        
        return loss_train
        
    def validation_step(self, val_batch, batch_idx):
        pass