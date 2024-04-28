import dgl.data as dgl_data
from dgl.dataloading import Sampler
import torch
from sklearn.preprocessing import StandardScaler

class SheafDataset(torch.utils.data.Dataset):
    def __init__(self, name: str):
        super(SheafDataset).__init__()
        # self.data = Planetoid(root='datasets/cora/', name='cora', transform=NormalizeFeatures())[0]

        dataset_names = {
            "texas": dgl_data.TexasDataset,
            "wisconsin": dgl_data.WisconsinDataset,
            "film": dgl_data.ActorDataset,
            "squirrel": dgl_data.SquirrelDataset,
            "chameleon": dgl_data.ChameleonDataset,
            "cornell": dgl_data.CornellDataset,
            "citeseer": dgl_data.CiteseerGraphDataset,
            "cora": dgl_data.CoraGraphDataset,
        }

        assert name in dataset_names

        self.data = dataset_names[name](raw_dir=f'datasets/{name}/')[0]#, transform=NormalizeFeatures())
        
        # print(self.data.ndata["feat"])
        
        feat = torch.clone(self.data.ndata["feat"])
        train_mask = self.data.ndata["train_mask"] | self.data.ndata["val_mask"]
        self.data.ndata["train_mask"] = train_mask
        test_mask = self.data.ndata["test_mask"]
        
        # print(train_mask)
        # print(test_mask)

        if len(train_mask.size()) > 1:
            train_mask = train_mask[:, 0]
            test_mask = test_mask[:, 0]

        self.data.ndata["train_mask"] = train_mask
        self.data.ndata["test_mask"] = test_mask

        scaler = StandardScaler()
        self.data.ndata["feat"][train_mask] = torch.FloatTensor(scaler.fit_transform(feat[train_mask]))
        self.data.ndata["feat"][test_mask] = torch.FloatTensor(scaler.transform(feat[test_mask]))
        # self.data.x = self.data.x.to(torch.float32)
        
        edges = torch.empty(2, self.data.num_edges(), dtype=int)
        edges[0] = self.data.edges()[0]
        edges[1] = self.data.edges()[1]
        self.data.edges_tensor = edges
        
        del feat, train_mask, test_mask, scaler, edges        
        
class MySampler(Sampler):
    def __init__(self):
        super().__init__()

    def sample(self, g, indices):
        return g
    



if __name__ == "__main__":
    dataset = SheafDataset("cora")
    print(dataset.data.edges_tensor)