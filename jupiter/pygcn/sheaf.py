import torch

def transform_features(inp: torch.FloatTensor):
    W = torch.eye(inp.size()[-1])
    return torch.matmul(inp, W)

