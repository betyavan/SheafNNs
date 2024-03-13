import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.models import GCN
import pytorch_lightning as pl
from torch_geometric.datasets import Planetoid

from sklearn.preprocessing import StandardScaler


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)


    
class CORA_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(CORA_Dataset).__init__()
        self.data = Planetoid(root='datasets/cora/', name='cora')[0]

        scaler = StandardScaler()
        self.data.x[self.data.train_mask] = torch.FloatTensor(scaler.fit_transform(self.data.x[self.data.train_mask]))
        self.data.x[self.data.val_mask] = torch.FloatTensor(scaler.transform(self.data.x[self.data.val_mask]))
        self.data.x[self.data.test_mask] = torch.FloatTensor(scaler.transform(self.data.x[self.data.test_mask]))

        self.data.x = self.data.x.to(torch.float32)
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.data.x, self.data.edge_index, self.data.y,\
               self.data.train_mask | self.data.val_mask, self.data.test_mask
    

    
class GCN_module(pl.LightningModule):
    def __init__(self, n_feat, n_hid, n_class,
                       dropout, sheaf_laplacian,
                       learning_rate=0.01, weight_decay=5e-4):
        super(GCN_module, self).__init__()
        self.model = GCN(nfeat=n_feat,
                         nhid=h_idden,
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
        x, edge_index, y, train_mask, val_mask = train_batch
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


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="name of training dataset")
    parser.add_argument("-d", type=int, help="dimension of edge space")
    parser.add_argument("--device", type=str, help="device for evaluation")
    parser.add_argument("--pretrained_sheaf", type=bool, help="use pretrained weights for sheaf")

    return parser
        
    

if __name__ == "__main__":
    
    os.makedirs("weights/", exist_ok=True)

    parser = init_parser()
    args = parser.parse_args()

    set_seed()
    
    dataset = CORA_Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    n_feat = dataset.data.x.size(1)
    hidden = 16
    n_class = dataset.data.y.max().item() + 1
    dropout = 0.5

    if args.pretrained_sheaf:
        sheaf_laplacian = torch.load(f"weights/slaplac_{args.dataset}_{args.d}.pt")
    else:
        from pygcn.train_sheaf import build_sheaf_laplacian
        sheaf_laplacian = build_sheaf_laplacian(dataset.data.x, dataset.data.edge_index, hidden, device=args.device)
        torch.save(sheaf_laplacian, f"weights/slaplac_{args.dataset}_{args.d}.pt")

    weight_decay = 5e-4
    
    module = GCN_module(n_feat, hidden, n_class, dropout, sheaf_laplacian)
    
    trainer = pl.Trainer(max_epochs=300, accelerator=args.device)
    
    trainer.fit(module, dataloader, dataloader)
    torch.save(module.state_dict(), f"weights/gcn_{args.dataset}_{args.d}.pt")
    
    
    
    