from ._registry import DATASETS
from torch_geometric.datasets import ModelNet as PyGModelNet
import torch_geometric.transforms as T


@DATASETS.register
class ModelNet(PyGModelNet):
    def __init__(self, root='data/ModelNet', name='40', train=True):
        # Default setting
        pre_transform = T.NormalizeScale()
        transform = T.SamplePoints(1024)
        pre_filter = None

        super().__init__(root+name, name=name, train=train,
                         transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)


