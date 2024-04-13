import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph
from scipy.spatial.distance import cdist
import torch


def build_sheaf_laplacian(x, edge_index, d, is_self_loops=True, device='cuda'):
    print("Start bulding sheaf Laplacian...")
    if device == "cuda":
        assert torch.cuda.is_available()
    n = x.size(1)
    O_matrices = torch.empty(x.size(0), n, d, device=device)
    O_means = torch.empty(x.size(0), n, device=device)
    dists = torch.tensor(cdist(x, x))
    for i in tqdm(range(x.size(0))):
        local_nbhood = k_hop_subgraph(i, 1, edge_index, relabel_nodes=False)[0]
        if len(local_nbhood) != d:
            dists_i = torch.clone(dists[i])
            if len(local_nbhood) < d:
                dists_i[local_nbhood] = np.inf
            else:
                dists_i[~local_nbhood] = np.inf
            nearests = dists_i.argsort()[: d if len(local_nbhood) > d else d - local_nbhood.size(0)]
            if local_nbhood.size(0) < d:
                local_nbhood = torch.concat([local_nbhood, nearests])
            else:
                local_nbhood = nearests     
        x_local = x[local_nbhood].T.to(device)
        U, _, _ = torch.linalg.svd(x_local) 
        O_matrices[i] = U[:, :d] # n x d
        O_means[i] = x[local_nbhood].mean(dim=0)
        
    # sheaf_laplacian = torch.empty(edge_index.size(1), n, n)
        
    # for k in tqdm(range(edge_index.size(1))[:5]):
    #     i, j = edge_index[:, k]
    #     mul = torch.matmul(O_matrices[i], O_matrices[j].T) # n x n
    #     # print(mul.get_device())
    #     U, _, V_T = torch.linalg.svd(mul, driver='gesvd')
    #     sheaf_laplacian[k] = torch.matmul(U, V_T).to('cpu')
        
    # if is_self_loops:
    #     self_laplac = torch.concat([torch.eye(n).unsqueeze(0) for _ in range(x.size(0))])
    #     sheaf_laplacian = torch.concat([sheaf_laplacian, self_laplac], axis=0)
    
    # return sheaf_laplacian
        
    print("Finished bulding sheaf Laplacian!")

    return {"local_pca": O_matrices, "local_mean": O_means}
    


# def build_sheaf_laplacian(x, edge_index, d, is_self_loops=True):
#     print("Start bulding sheaf Laplacian...")
#     n = x.size(1)
#     O_matrices = torch.empty(x.size(0), n, d)
#     dists = cdist(x, x)
#     for i in tqdm(range(x.size(0))):
#         local_nbhood = k_hop_subgraph(i, 1, edge_index, relabel_nodes=False)[0]
#         if len(local_nbhood) != d:
#             dists_i = dists[i].copy()
#             if len(local_nbhood) < d:
#                 dists_i[local_nbhood] = np.inf
#             else:
#                 dists_i[~local_nbhood] = np.inf
#             ind = dists_i.argsort()[: d if len(local_nbhood) > d else d - local_nbhood.size(0)]
#             nearests = torch.tensor(ind)
#             if local_nbhood.size(0) < d:
#                 local_nbhood = torch.concat([local_nbhood, nearests])
#             else:
#                 local_nbhood = nearests

#         U, _, _ = np.linalg.svd(x[local_nbhood].T) 
#         O_matrices[i] = torch.from_numpy(U[:, :d]) # n x d
        
#     sheaf_laplacian = torch.empty(edge_index.size(1), n, n)
        
#     for k in tqdm(range(edge_index.size(1))):
#         i, j = edge_index[:, k]
#         mul = torch.matmul(O_matrices[i], O_matrices[j].T) # n x n
#         U, _, V_T = np.linalg.svd(mul)
#         sheaf_laplacian[k] = torch.tensor(np.dot(U, V_T))
        
#     if is_self_loops:
#         self_laplac = torch.concat([torch.eye(n).unsqueeze(0) for _ in range(x.size(0))])
#         sheaf_laplacian = torch.concat([sheaf_laplacian, self_laplac], axis=0)
    
#     print("Finished bulding sheaf Laplacian!")
#     return sheaf_laplacian
    
        
        