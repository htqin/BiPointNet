from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin, Sequential as S, BatchNorm1d as BN, ELU, Hardtanh
from torch_geometric.nn import XConv, fps, global_mean_pool, global_max_pool, Reshape

from ._registry import MODELS
from .basic import BinaryQuantize, BiLinear, MLP, BiMLP, BiConv1d, BiConv1dLSR, BiLinearLSR, BiLinearXNOR, BiConv1dXNOR


try:
    from torch_cluster import knn_graph
except ImportError:
    knn_graph = None

offset_map = {
  258: -2.7829
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


class BiXConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim, kernel_size,
                 hidden_channels=None, dilation=1, bias=True, BiLinear=BiLinear, BiConv1d=BiConv1d, ifFirst=False, **kwargs):
        super(BiXConv, self).__init__()

        if knn_graph is None:
            raise ImportError('`XConv` requires `torch-cluster`.')

        self.in_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels // 4
        assert hidden_channels > 0
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.kwargs = kwargs

        C_in, C_delta, C_out = in_channels, hidden_channels, out_channels
        D, K = dim, kernel_size

        if ifFirst:
            Lin1 = Lin
        else:
            Lin1 = BiLinear

        self.mlp1 = S(
            Lin1(dim, C_delta),
            Hardtanh(),
            BN(C_delta),
            BiLinear(C_delta, C_delta),
            Hardtanh(),
            BN(C_delta),
            Reshape(-1, K, C_delta),
        )

        self.mlp2 = S(
            Lin1(D * K, K**2),
            Hardtanh(),
            BN(K**2),
            Reshape(-1, K, K),
            BiConv1d(K, K**2, K, groups=K),
            Hardtanh(),
            BN(K**2),
            Reshape(-1, K, K),
            BiConv1d(K, K**2, K, groups=K),
            BN(K**2),
            Reshape(-1, K, K),
        )

        C_in = C_in + C_delta
        depth_multiplier = int(ceil(C_out / C_in))
        self.conv = S(
            BiConv1d(C_in, C_in * depth_multiplier, K, groups=C_in),
            Reshape(-1, C_in * depth_multiplier),
            BiLinear(C_in * depth_multiplier, C_out, bias=bias),
        )

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp1)
        reset(self.mlp2)
        reset(self.conv)

    def forward(self, x, pos, batch=None):
        """"""
        pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
        (N, D), K = pos.size(), self.kernel_size

        row, col = knn_graph(pos, K * self.dilation, batch, loop=True,
                             flow='target_to_source', **self.kwargs)

        if self.dilation > 1:
            dil = self.dilation
            index = torch.randint(K * dil, (N, K), dtype=torch.long,
                                  device=row.device)
            arange = torch.arange(N, dtype=torch.long, device=row.device)
            arange = arange * (K * dil)
            index = (index + arange.view(-1, 1)).view(-1)
            row, col = row[index], col[index]

        pos = pos[col] - pos[row]

        x_star = self.mlp1(pos.view(N * K, D))
        if x is not None:
            x = x.unsqueeze(-1) if x.dim() == 1 else x
            x = x[col].view(N, K, self.in_channels)
            x_star = torch.cat([x_star, x], dim=-1)
        x_star = x_star.transpose(1, 2).contiguous()
        x_star = x_star.view(N, self.in_channels + self.hidden_channels, K, 1)

        transform_matrix = self.mlp2(pos.view(N, K * D))
        transform_matrix = transform_matrix.view(N, 1, K, K)

        x_transformed = torch.matmul(transform_matrix, x_star)
        x_transformed = x_transformed.view(N, -1, K)

        out = self.conv(x_transformed)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class BasicPointCNN(torch.nn.Module):
    def __init__(self, pool='mean'):
        super().__init__()

        self.conv1 = XConv(0, 48, dim=3, kernel_size=8, hidden_channels=32)
        self.conv2 = XConv(
            48, 96, dim=3, kernel_size=12, hidden_channels=64, dilation=2)
        self.conv3 = XConv(
            96, 192, dim=3, kernel_size=16, hidden_channels=128, dilation=2)
        self.conv4 = XConv(
            192, 384, dim=3, kernel_size=16, hidden_channels=256, dilation=2)

        self.lin1 = Lin(384, 256)
        self.lin2 = Lin(256, 128)
        self.lin3 = Lin(128, 40)
        self.pool = pool


    def forward(self, data):
        x, pos, batch = data.x, data.pos[:, :3], data.batch
        x = F.relu(self.conv1(None, pos, batch))

        idx = fps(pos, batch, ratio=0.375)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.conv2(x, pos, batch))

        idx = fps(pos, batch, ratio=0.334)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.conv3(x, pos, batch))
        x = F.relu(self.conv4(x, pos, batch))

        if self.pool == 'max':
            x = global_max_pool(x, batch)
        elif self.pool == 'mean':
            x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return {
            'out': F.log_softmax(x, dim=-1)
        }


class BasicBiPointCNN(torch.nn.Module):
    def __init__(self, BiLinear=BiLinear, BiConv1d=BiConv1d, pool='mean'):
        super().__init__()

        self.conv1 = BiXConv(0, 48, dim=3, kernel_size=8, hidden_channels=32, BiLinear=BiLinear, BiConv1d=BiConv1d, ifFirst=True)
        self.conv2 = BiXConv(
            48, 96, dim=3, kernel_size=12, hidden_channels=64, dilation=2, BiLinear=BiLinear, BiConv1d=BiConv1d, ifFirst=True)
        self.conv3 = BiXConv(
            96, 192, dim=3, kernel_size=16, hidden_channels=128, dilation=2, BiLinear=BiLinear, BiConv1d=BiConv1d, ifFirst=True)
        self.conv4 = BiXConv(
            192, 384, dim=3, kernel_size=16, hidden_channels=256, dilation=2, BiLinear=BiLinear, BiConv1d=BiConv1d, ifFirst=True)

        self.lin1 = BiLinear(384, 256)
        self.lin2 = BiLinear(256, 128)
        self.lin3 = Lin(128, 40)
        self.pool = pool

    def forward(self, data):
        x, pos, batch = data.x, data.pos[:, :3], data.batch
        x = F.hardtanh(self.conv1(None, pos, batch))

        idx = fps(pos, batch, ratio=0.375)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.hardtanh(self.conv2(x, pos, batch))

        idx = fps(pos, batch, ratio=0.334)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.hardtanh(self.conv3(x, pos, batch))
        x = F.hardtanh(self.conv4(x, pos, batch))
        if self.pool == 'max':
            x = global_max_pool(x, batch)
        elif self.pool == 'mean':
            x = global_mean_pool(x, batch)

        x = F.hardtanh(self.lin1(x))
        x = F.hardtanh(self.lin2(x))
        x = self.lin3(x)
        return {
            'out': F.log_softmax(x, dim=-1)
        }


@MODELS.register
class PointCNNMean(BasicPointCNN):
    def __init__(self):
        super().__init__(pool='mean')


@MODELS.register
class BiPointCNNMean(BasicBiPointCNN):
    def __init__(self):
        super().__init__(BiLinear=BiLinear, BiConv1d=BiConv1d, pool='mean')


@MODELS.register
class BiPointCNNXNORMean(BasicBiPointCNN):
    def __init__(self):
        super().__init__(BiLinear=BiLinearXNOR, BiConv1d=BiConv1dXNOR, pool='mean')


@MODELS.register
class BiPointCNNLSRMean(BasicBiPointCNN):
    def __init__(self):
        super().__init__(BiLinear=BiLinearLSR, BiConv1d=BiConv1dLSR, pool='mean')
