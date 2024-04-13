from torch_geometric.transforms import NormalizeFeatures
import dgl
import torch

class SheafDataset(torch.utils.data.Dataset):
    def __init__(self, name: str):
        super(SheafDataset).__init__()
        # self.data = Planetoid(root='datasets/cora/', name='cora', transform=NormalizeFeatures())[0]

        dataset_names = {
            "texas": dgl.data.TexasDataset,
            "wisconsin": dgl.data.WisconsinDataset,
            "film": dgl.data.ActorDataset,
            "squirrel": dgl.data.SquirrelDataset,
            "chameleon": dgl.data.ChameleonDataset,
            "cornell": dgl.data.CornellDataset,
            "citeseer": dgl.data.CiteseerGraphDataset,
            "cora": dgl.data.CoraGraphDataset,
        }

        assert name in dataset_names

        self.data = dataset_names[name](raw_dir=f'datasets/{name}/')#, transform=NormalizeFeatures())

        # scaler = StandardScaler()
        # self.data.x[self.data.train_mask] = torch.FloatTensor(scaler.fit_transform(self.data.x[self.data.train_mask]))
        # self.data.x[self.data.val_mask] = torch.FloatTensor(scaler.transform(self.data.x[self.data.val_mask]))
        # self.data.x[self.data.test_mask] = torch.FloatTensor(scaler.transform(self.data.x[self.data.test_mask]))
        # self.data.x = self.data.x.to(torch.float32)
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.data.x, self.data.edge_index, self.data.y,\
               self.data.train_mask | self.data.val_mask, self.data.test_mask
    



if __name__ == "__main__":
    dataset = SheafDataset("cora")
    print(dataset.data[0])