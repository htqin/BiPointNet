from ._registry import MODELS
from .basic import BiLinear, BiLinearXNOR, \
    BiLinearIRNet, BiLinearLSR, MLP, BiMLP, BwMLP, FirstBiMLP

from functools import wraps
import time
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import fps, radius, global_max_pool, global_mean_pool
from torch_geometric.nn import knn_interpolate
from torch_geometric.utils import intersection_and_union as i_and_u
import numpy as np

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops


seg_classes = {
    'Airplane': [0, 1, 2, 3],
    'Bag': [4, 5],
    'Cap': [6, 7],
    'Car': [8, 9, 10, 11],
    'Chair': [12, 13, 14, 15],
    'Earphone': [16, 17, 18],
    'Guitar': [19, 20, 21],
    'Knife': [22, 23],
    'Lamp': [24, 25, 26, 27],
    'Laptop': [28, 29],
    'Motorbike': [30, 31, 32, 33, 34, 35],
    'Mug': [36, 37],
    'Pistol': [38, 39, 40],
    'Rocket': [41, 42, 43],
    'Skateboard': [44, 45, 46],
    'Table': [47, 48, 49],
}

offset_map = {
  64: -2.2983,
  1024: -3.2041
}


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class PointConv(MessagePassing):
    """Modified __init__ from PyG"""

    def __init__(self, local_nn=None, global_nn=None, **kwargs):
        super(PointConv, self).__init__(**kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn)

    def forward(self, x, pos, edge_index):
        if torch.is_tensor(pos):  # Add self-loops for symmetric adjacencies.
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=pos.size(0))

        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j, pos_i, pos_j):
        msg = pos_j - pos_i
        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def update(self, aggr_out):
        if self.global_nn is not None:
            aggr_out = self.global_nn(aggr_out)
        return aggr_out

    def __repr__(self):
        return '{}(local_nn={}, global_nn={})'.format(
            self.__class__.__name__, self.local_nn, self.global_nn)


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn, aggr='max', random_start=True):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.random_start = random_start
        if aggr == 'max':
            self.aggr = 'max'
            self.ema_max = False
        elif aggr == 'mean':
            self.aggr = 'mean'
            self.ema_max = False
        elif aggr == 'bmax':
            self.aggr = 'max'
            self.ema_max = True
        self.conv = PointConv(nn, aggr=self.aggr)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio, random_start=self.random_start)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        if self.ema_max:
            x = x + offset_map[64]
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn, aggr='max'):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

        self.aggr = aggr
        assert self.aggr in ['max', 'mean', 'ema-max']

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))

        if self.aggr == 'max':
            x = global_max_pool(x, batch)
        elif self.aggr == 'mean':
            x = global_mean_pool(x, batch)
        elif self.aggr == 'ema-max':
            x = global_max_pool(x, batch) + offset_map[1024]
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip

# (Classification) Full-Precision Models


class BasicPointNet2(torch.nn.Module):
    def __init__(self, pool):
        super().__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]), pool)
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]), pool)
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]), pool)

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 40)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return {
            'out': F.log_softmax(x, dim=-1),
            'sa1_out': sa1_out,
            'sa2_out': sa2_out
        }

# (Classification) Binary Models


class BasicBiPointNet2(torch.nn.Module):
    def __init__(self, BiLinear=BiLinear, pool='max'):
        super().__init__()
        self.sa1_module = SAModule(0.5,
                                   0.2,
                                   Seq(Lin(3, 64), BiMLP([64, 64, 128], BiLinear=BiLinear)),
                                   pool,
                                   random_start=False)
        self.sa2_module = SAModule(0.25, 0.4, BiMLP([128 + 3, 128, 128, 256], BiLinear=BiLinear), pool, random_start=False)
        self.sa3_module = GlobalSAModule(BiMLP([256 + 3, 256, 512, 1024], BiLinear=BiLinear), pool)
        self.lin1 = BiLinear(1024, 512)
        self.lin2 = BiLinear(512, 256)
        self.lin3 = Lin(256, 40)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        x = F.hardtanh(self.lin1(x))
        x = F.hardtanh(self.lin2(x))
        x = self.lin3(x)
        return {
            'out': F.log_softmax(x, dim=-1),
            'sa1_out': sa1_out,
            'sa2_out': sa2_out
        }


@MODELS.register
class PointNet2Max(BasicPointNet2):
    def __init__(self):
        super().__init__(pool='max')


@MODELS.register
class BiPointNet2XNORMax(BasicBiPointNet2):
    def __init__(self):
        super().__init__(BiLinear=BiLinearXNOR, pool='max')


@MODELS.register
class BiPointNet2LSREMax(BasicBiPointNet2):
    def __init__(self):
        super().__init__(BiLinear=BiLinearLSR, pool='ema-max')
