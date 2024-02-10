import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.datasets import Planetoid


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)


    
class CORA_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(CORA_Dataset).__init__()
        self.data = Planetoid(root='./cora/', name='cora')[0]
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.data.x, self.data.edge_index, self.data.y,\
               self.data.train_mask, self.data.val_mask, self.data.test_mask
    

    
class GCN_module(pl.LightningModule):
    def __init__(self, nfeat, nhid, nclass, dropout, laplac_size, learning_rate=0.01, weight_decay=5e-4):
        super(GCN_module, self).__init__()
        self.model = GCN(nfeat=nfeat,
                         nhid=hidden,
                         nclass=nclass,
                         dropout=dropout,
                         laplac_size=laplac_size)
        
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
        x, edge_index, y, train_mask, val_mask, _ = train_batch
        x, edge_index, y = x[0], edge_index[0], y[0]
        train_mask, val_mask = train_mask[0], val_mask[0]
        
        output = self.model(x, edge_index)
        
        loss_train = self.loss_fn(output[train_mask], y[train_mask])
        acc_train = accuracy(output[train_mask], y[train_mask])
        
        loss_val = self.loss_fn(output[val_mask], y[val_mask])
        acc_val = accuracy(output[val_mask], y[val_mask])
        
        self.log("loss_train", loss_train, prog_bar=True)
        self.log("acc_train", acc_train, prog_bar=True)
        
        self.log("loss_val", loss_val, prog_bar=True)
        self.log("acc_val", acc_val, prog_bar=True)
        
        return loss_train
        
    def validation_step(self, val_batch, batch_idx):
        pass
        
    

if __name__ == "__main__":
    set_seed()
    
    dataset = CORA_Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    nfeat = dataset.data.x.size(1)
    hidden = 16
    nclass = dataset.data.y.max().item() + 1
    dropout = 0.5
    laplac_size = dataset.data.edge_index.size(1) + dataset.data.x.size(0)
    weight_decay = 5e-4
    
    module = GCN_module(nfeat, hidden, nclass, dropout, laplac_size)
    
    device = "cpu"
    trainer = pl.Trainer(max_epochs=400, accelerator=device)
    
    trainer.fit(module, dataloader, dataloader)
    
    
    
    