import math

import torch
from torch.nn.modules.module import Module

from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from tqdm import tqdm

# def build_align_matrix(sheaf_laplac, edge_index):
    
#     n = sheaf_laplac.size(1)
#     align_matrices = torch.empty(edge_index.size(1), n, n)
        
#     for k in tqdm(range(edge_index.size(1))):
#         i, j = edge_index[:, k]
#         mul = torch.matmul(sheaf_laplac[i], sheaf_laplac[j].T) # n x n
#         if mul.get_device() == -1:
#             U, _, V_T = torch.linalg.svd(mul)
#         else:
#             U, _, V_T = torch.linalg.svd(mul, driver='gesvd')
#         align_matrices[k] = torch.matmul(U, V_T)#.to('cpu')
        
#     # if is_self_loops:
#     #     self_laplac = torch.concat([torch.eye(n).unsqueeze(0) for _ in range(x.size(0))])
#     #     sheaf_laplacian = torch.concat([sheaf_laplacian, self_laplac], axis=0)
    
#     return align_matrices


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, sheaf_laplacian=None):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        
        self.sheaf_laplacian = sheaf_laplacian

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        edge_index = edge_index.to(torch.device(x.get_device()))
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        # x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # eval align matrices
        # self.local_pca, self.local_mean = build_align_matrix(self.sheaf_laplacian, edge_index)
        self.edge_index = edge_index

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        
        edge_index = self.edge_index
        
        if self.sheaf_laplacian is not None:
            n, d = x_j.size(1), self.sheaf_laplacian['local_pca'].size(2)
            # print("x_j", x_j.size())
            # print("self.local_mean['i']", self.local_mean['i'].size())
            # print("self.local_mean['j']", self.local_mean['j'].size())
            # print("self.local_pca['i']", self.local_pca['i'].size())
            # print("self.local_pca['j']", self.local_pca['j'].size())
            # print("="*50, edge_index.size(), "="*50)
            x_j = x_j - self.sheaf_laplacian['local_mean'][edge_index[0]]
            x_j = torch.matmul(torch.transpose(self.sheaf_laplacian['local_pca'][edge_index[0]], 1, 2),
                               x_j.unsqueeze(2)).view(-1, d)
            x_j = torch.matmul(self.sheaf_laplacian['local_pca'][edge_index[1]],
                               x_j.unsqueeze(2)).view(-1, n)
            x_j = x_j + self.sheaf_laplacian['local_mean'][edge_index[1]]
        
        x_j = self.lin(x_j)

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

