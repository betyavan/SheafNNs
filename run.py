import os
import argparse
import numpy as np

from pygcn.models import GCN_module

import dgl

from pygcn.data import *

import torch
# from torch_geometric.datasets import Planetoid
# from torch_geometric.transforms import NormalizeFeatures
# from torch_geometric.loader import DataLoader

import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger


import os 
import random
import numpy as np 

DEFAULT_RANDOM_SEED = 42

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
# torch random seed
import torch
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
# basic + tensorflow + torch 
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)


    
# class CORA_Dataset(torch.utils.data.Dataset):
#     def __init__(self):
#         super(CORA_Dataset).__init__()
#         self.data = Planetoid(root='datasets/cora/', name='cora', transform=NormalizeFeatures())

#         # scaler = StandardScaler()
#         # self.data.x[self.data.train_mask] = torch.FloatTensor(scaler.fit_transform(self.data.x[self.data.train_mask]))
#         # self.data.x[self.data.val_mask] = torch.FloatTensor(scaler.transform(self.data.x[self.data.val_mask]))
#         # self.data.x[self.data.test_mask] = torch.FloatTensor(scaler.transform(self.data.x[self.data.test_mask]))
#         # self.data.x = self.data.x.to(torch.float32)
        
#     def __len__(self):
#         return 1
    
#     def __getitem__(self, idx):
#         return self.data.x, self.data.edge_index, self.data.y,\
#                self.data.train_mask | self.data.val_mask, self.data.test_mask
    


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="name of training dataset")
    parser.add_argument("-d", type=int, help="dimension of edge space")
    parser.add_argument("--device", type=str, help="device for evaluation")
    parser.add_argument("--pretrained_sheaf", action="store_true", help="use pretrained weights for sheaf")

    return parser
        
    

if __name__ == "__main__":
    
    os.makedirs("weights/", exist_ok=True)

    parser = init_parser()
    args = parser.parse_args()

    seedEverything()
    
    # dataset = CORA_Dataset()
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # dataloader = DataLoader(
    #     dataset.data,
    #     batch_size=1
    # )
    
    g = SheafDataset(args.dataset).data
    dataloader = dgl.dataloading.DataLoader(g, torch.arange(3), MySampler(),
                                        batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    
    # for i in dataloader:
    #     print(i.ndata["feat"])
    # exit()
    
    n_feat = g.ndata["feat"].size(1)
    d = args.d
    n_class = g.ndata["label"].max().item() + 1
    dropout = 0.5

    if args.pretrained_sheaf:
        sheaf_laplacian = {
            "local_pca": torch.load(f"weights/pca_{args.dataset}_{args.d}.pt"),
            "local_mean": torch.load(f"weights/mean_{args.dataset}_{args.d}.pt")
        }
        print("Sheaf Laplacian loaded successfully\n")
    else:
        from pygcn.train_sheaf import build_sheaf_laplacian
        sheaf_laplacian = build_sheaf_laplacian(g.ndata["feat"], g.edges_tensor, d, device=args.device)
        # print('local_pca', sheaf_laplacian['local_pca'].size())
        # print('local_mean', sheaf_laplacian['local_mean'].size())
        torch.save(sheaf_laplacian["local_pca"], f"weights/pca_{args.dataset}_{args.d}.pt")
        torch.save(sheaf_laplacian["local_mean"], f"weights/mean_{args.dataset}_{args.d}.pt")

    weight_decay = 5e-4
    
    module = GCN_module(n_feat, d, n_class, dropout, sheaf_laplacian)
    
    logger = TensorBoardLogger(f"logs/{args.dataset}", name=f"my_model_{args.d}")
    trainer = pl.Trainer(max_epochs=100, accelerator=args.device, logger=logger, log_every_n_steps=5)
    
    trainer.fit(module, dataloader, dataloader)
    torch.save(module.state_dict(), f"weights/gcn_{args.dataset}_{args.d}.pt")
    
    
    
    