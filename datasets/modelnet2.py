from ._registry import DATASETS
import torch
import numpy as np
from torch_geometric.datasets import ModelNet as PyGModelNet
import torch_geometric.transforms as T


def pc_normalize(pc):
    centroid = torch.mean(pc, dim=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc * pc, dim=1)))
    pc = pc / m
    return pc


@DATASETS.register
class ModelNet2(PyGModelNet):
    def __init__(self, root='data/ModelNet', name='40', train=True):
        # Default setting
        pre_transform = T.NormalizeScale()
        transform = T.SamplePoints(1024)
        pre_filter = None

        super().__init__(root+name, name=name, train=train,
                         transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        # import pdb; pdb.set_trace()
        # TODO, no effective normal available
        data.pos = torch.cat((pc_normalize(data.pos), data.pos), dim=-1)
        return data
