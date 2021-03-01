from ._registry import MODELS
from .basic import BinaryQuantize, BiLinear, BiLinearXNOR, BiLinearLSR, MLP, BiMLP

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, ReLU
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, global_mean_pool

offset_map = {
  20: -1.8268,
  1024: -3.2041
}


class BasicDGCNN(torch.nn.Module):
    def __init__(self, out_channels=40, k=20, pool='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, pool)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, pool)
        self.lin1 = MLP([128 + 64, 1024])
        self.pool = pool

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        if self.pool == 'max':
            out = global_max_pool(out, batch)
        else:
            out = global_mean_pool(out, batch)
        out = self.mlp(out)
        return {
            'out': F.log_softmax(out, dim=1)
        }


class BasicBiDGCNN(torch.nn.Module):
    def __init__(self, out_channels=40, k=20, BiLinear=BiLinear, pool='max'):
        super().__init__()

        if pool == 'mean':
            self.pool = 'mean'
            self.ema_max = False
        elif pool == 'max':
            self.pool = 'max'
            self.ema_max = False
        elif pool == 'ema-max':
            self.pool = 'max'
            self.ema_max = True
        self.conv1 = DynamicEdgeConv(Seq(
                         Lin(2 * 3, 64),
                         BiMLP([64, 64, 64], activation=ReLU, BiLinear=BiLinear)), 
                         k,
                         self.pool)
        self.conv2 = DynamicEdgeConv(BiMLP([2 * 64, 128], activation=ReLU, BiLinear=BiLinear), k, self.pool)
        self.lin1 = BiMLP([128 + 64, 1024], activation=ReLU, BiLinear=BiLinear)

        self.mlp = Seq(
            BiMLP([1024, 512], activation=ReLU, BiLinear=BiLinear),
            BiMLP([512, 256], activation=ReLU, BiLinear=BiLinear),
            Lin(256, out_channels))

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        if self.ema_max:
            x1 = x1 + offset_map[20]
        x2 = self.conv2(x1, batch)
        if self.ema_max:
            x2 = x2 + offset_map[20]
        out = self.lin1(torch.cat([x1, x2], dim=1))
        if self.pool == 'max':
            out = global_max_pool(out, batch)
            if self.ema_max:
                out = out + offset_map[1024]
        else:
            out = global_mean_pool(out, batch)
        out = self.mlp(out)
        return {
            'out': F.log_softmax(out, dim=1)
        }


@MODELS.register
class DGCNNMean(BasicDGCNN):
    def __init__(self):
        super().__init__(pool='mean')


@MODELS.register
class DGCNNMax(BasicDGCNN):
    def __init__(self):
        super().__init__(pool='max')


@MODELS.register
class BiDGCNNXNORMax(BasicBiDGCNN):
    def __init__(self):
        super().__init__(BiLinear=BiLinearXNOR, pool='max')


@MODELS.register
class BiDGCNNLSREMax(BasicBiDGCNN):
    def __init__(self):
        super().__init__(BiLinear=BiLinearLSR, pool='ema-max')


@MODELS.register
class BiDGCNNLSRMean(BasicBiDGCNN):
    def __init__(self):
        super().__init__(BiLinear=BiLinearLSR, pool='mean')
