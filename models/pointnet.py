from ._registry import MODELS
from .basic import MeanShift, BiLinear, BiLinearXNOR, BiLinearIRNet, BiLinearLSR, BiLinearBiReal
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


biLinears = {
    'BiLinear': BiLinear,
    'BiLinearXNOR': BiLinearXNOR,
    'BiLinearIRNet': BiLinearIRNet,
    'BiLinearLSR': BiLinearLSR
}


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
    1024: -3.2041,
    2048: -3.4025,
    4096: -3.5836
}


class Conv1d(nn.Module):
    def __init__(self, inplane, outplane, Linear):
        super().__init__()
        self.lin = Linear(inplane, outplane)

    def forward(self, x):
        B, C, N = x.shape
        x = x.permute(0, 2, 1).contiguous().view(-1, C)
        x = self.lin(x).view(B, N, -1).permute(0, 2, 1).contiguous()
        return x


class STN3d(nn.Module):
    def __init__(self, channel, Linear, pool='max'):
        super(STN3d, self).__init__()
        self.conv1 = Conv1d(channel, 64, nn.Linear)
        self.conv2 = Conv1d(64, 128, Linear)
        self.conv3 = Conv1d(128, 1024, Linear)
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.pool = pool

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        if self.pool == 'max':
            x = torch.max(x, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            x = torch.mean(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64, Linear=nn.Linear, pool='max'):
        super(STNkd, self).__init__()
        self.conv1 = Conv1d(k, 64, Linear)
        self.conv2 = Conv1d(64, 128, Linear)
        self.conv3 = Conv1d(128, 1024, Linear)
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.pool = pool

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        if self.pool == 'max':
            x = torch.max(x, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            x = torch.mean(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, Linear, global_feat=True, feature_transform=False, channel=3, pool='max', tnet=True):
        super(PointNetEncoder, self).__init__()
        self.tnet = tnet
        if self.tnet:
            self.stn = STN3d(channel, Linear, pool)
        self.conv1 = Conv1d(channel, 64, Linear)
        self.conv2 = Conv1d(64, 128, Linear)
        self.conv3 = Conv1d(128, 1024, Linear)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.tnet and self.feature_transform:
            self.fstn = STNkd(k=64, Linear=Linear, pool=pool)
        self.pool = pool

    def forward(self, x):
        B, D, N = x.size()

        if self.tnet:
            trans = self.stn(x)
        else:
            trans = None

        x = x.transpose(2, 1)
        if D == 6:
            x, feature = x.split(3, dim=2)
        elif D == 9:
            x, feature = x.split([3, 6], dim=2)
        if self.tnet:
            x = torch.bmm(x, trans)

        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.tnet and self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.pool == 'max':
            x = torch.max(x, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            x = torch.mean(x, 2, keepdim=True)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class BasicPointNet(nn.Module):
    def __init__(self, k=40, normal_channel=False, Linear=nn.Linear, tnet=True, pool='max'):
        super().__init__()
        self.normal_channel = normal_channel
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(Linear, global_feat=True, feature_transform=True, channel=channel, tnet=tnet, pool=pool)
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, data):
        x = data.pos
        batch = torch.max(data.batch) + 1
        pos_list = []
        for i in range(batch):
            pos_list.append(x[data.batch == i])
        x = torch.stack(pos_list).permute(0, 2, 1).contiguous()
        if not self.normal_channel:
            x = x[:, :3, :]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return {
            'trans': trans,
            'trans_feat': trans_feat,
            'out': F.log_softmax(x, dim=-1),
        }

# BiPointNet


class BiSTN3d(nn.Module):
    def __init__(self, channel, Linear=BiLinear, pool='max', affine=True, bi_first=False):
        super(BiSTN3d, self).__init__()
        if bi_first:
            self.conv1 = Conv1d(channel, 64, Linear)
        else:
            self.conv1 = Conv1d(channel, 64, nn.Linear)
        self.conv2 = Conv1d(64, 128, Linear)
        self.conv3 = Conv1d(128, 1024, Linear)
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64, affine=affine)
        self.bn2 = nn.BatchNorm1d(128, affine=affine)
        self.bn3 = nn.BatchNorm1d(1024, affine=affine)
        self.bn4 = nn.BatchNorm1d(512, affine=affine)
        self.bn5 = nn.BatchNorm1d(256, affine=affine)
        self.pool = pool

    def forward(self, x):

        batchsize, D, N = x.size()
        x = F.hardtanh(self.bn1(self.conv1(x)))
        x = F.hardtanh(self.bn2(self.conv2(x)))

        if self.pool == 'max':
            x = F.hardtanh(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            x = F.hardtanh(self.bn3(self.conv3(x)))
            x = torch.mean(x, 2, keepdim=True)
        elif self.pool == 'ema-max':
            x = self.bn3(self.conv3(x)) + offset_map[N]
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
        x = x.view(-1, 1024)

        x = F.hardtanh(self.bn4(self.fc1(x)))
        x = F.hardtanh(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)

        return x


class BiSTNkd(nn.Module):
    def __init__(self, k=64, Linear=BiLinear, pool='max', affine=True, bi_first=False):
        super(BiSTNkd, self).__init__()
        self.conv1 = Conv1d(k, 64, Linear)
        self.conv2 = Conv1d(64, 128, Linear)
        self.conv3 = Conv1d(128, 1024, Linear)
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64, affine=affine)
        self.bn2 = nn.BatchNorm1d(128, affine=affine)
        self.bn3 = nn.BatchNorm1d(1024, affine=affine)
        self.bn4 = nn.BatchNorm1d(512, affine=affine)
        self.bn5 = nn.BatchNorm1d(256, affine=affine)
        self.k = k
        self.pool = pool

    def forward(self, x):
        batchsize, D, N = x.size()
        x = F.hardtanh(self.bn1(self.conv1(x)))
        x = F.hardtanh(self.bn2(self.conv2(x)))
        if self.pool == 'max':
            x = F.hardtanh(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            x = F.hardtanh(self.bn3(self.conv3(x)))
            x = torch.mean(x, 2, keepdim=True)
        elif self.pool == 'ema-max':
            x = self.bn3(self.conv3(x)) + offset_map[N]
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
        x = x.view(-1, 1024)

        x = F.hardtanh(self.bn4(self.fc1(x)))
        x = F.hardtanh(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class BiPointNetEncoder(nn.Module):
    def __init__(self, Linear, global_feat=True, feature_transform=False, channel=3, pool='max', affine=True, tnet=True, bi_first=False, use_bn=True):
        super(BiPointNetEncoder, self).__init__()
        self.tnet = tnet
        if self.tnet:
            self.stn = BiSTN3d(channel, Linear, pool=pool, affine=affine, bi_first=bi_first)
        if bi_first:
            self.conv1 = Conv1d(channel, 64, Linear)
        else:
            self.conv1 = Conv1d(channel, 64, nn.Linear)
        self.conv2 = Conv1d(64, 128, Linear)
        self.conv3 = Conv1d(128, 1024, Linear)
        self.bn1 = nn.BatchNorm1d(64, affine=affine)
        self.bn2 = nn.BatchNorm1d(128, affine=affine)
        self.bn3 = nn.BatchNorm1d(1024, affine=affine)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.tnet and self.feature_transform:
            self.fstn = BiSTNkd(k=64, Linear=Linear, pool=pool, affine=affine, bi_first=bi_first)
        self.pool = pool
        self.use_bn = use_bn

    def forward(self, x):
        B, D, N = x.size()
        if self.tnet:
            trans = self.stn(x)
        else:
            trans = None

        x = x.transpose(2, 1)
        if D == 6:
            x, feature = x.split(3, dim=2)
        elif D == 9:
            x, feature = x.split([3, 6], dim=2)
        if self.tnet:
            x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        if self.use_bn:
            x = F.hardtanh(self.bn1(self.conv1(x)))
        else:
            x = F.hardtanh(self.conv1(x))

        if self.tnet and self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        if self.use_bn:
            x = F.hardtanh(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
        else:
            x = F.hardtanh(self.conv2(x))
            x = self.conv3(x)

        if self.pool == 'max':
            x = torch.max(x, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            x = torch.mean(x, 2, keepdim=True)
        elif self.pool == 'ema-max':
            if self.use_bn:
                x = torch.max(x, 2, keepdim=True)[0] + offset_map[N]
            else:
                x = torch.max(x, 2, keepdim=True)[0] - 0.3
            x = x.view(-1, 1024)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class BasicBiPointNet(nn.Module):
    def __init__(self, k=40, normal_channel=False, Linear=BiLinear, pool='max', affine=True, tnet=True, bi_first=False, bi_last=False, use_bn=True):
        super(BasicBiPointNet, self).__init__()
        self.normal_channel = normal_channel
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = BiPointNetEncoder(Linear, global_feat=True, feature_transform=True, channel=channel, pool=pool, affine=affine, tnet=tnet, bi_first=bi_first, use_bn=use_bn)
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 256)
        if bi_last:
            self.fc3 = Linear(256, k)
        else:
            self.fc3 = nn.Linear(256, k)
        self.use_bn = use_bn
        self.bn1 = nn.BatchNorm1d(512, affine=affine)
        self.bn2 = nn.BatchNorm1d(256, affine=affine)

    def forward(self, data):
        x = data.pos
        batch = torch.max(data.batch) + 1
        pos_list = []
        for i in range(batch):
            pos_list.append(x[data.batch == i])
        x = torch.stack(pos_list).permute(0, 2, 1).contiguous()
        if not self.normal_channel:
            x = x[:, :3, :]
        x, trans, trans_feat = self.feat(x)
        if self.use_bn:
            x = F.hardtanh(self.bn1(self.fc1(x)))
            x = F.hardtanh(self.bn2(self.fc2(x)))
        else:
            x = F.hardtanh(self.fc1(x))
            x = F.hardtanh(self.fc2(x))
        x = self.fc3(x)
        return {
            'trans': trans,
            'trans_feat': trans_feat,
            'out': F.log_softmax(x, dim=-1),
        }


class BasicPointNetPartSeg(nn.Module):
    def __init__(self, categories='Airplane', normal_channel=True, Linear=nn.Linear, pool='max', affine=True, Tnet=True, concat_global=True):
        super().__init__()
        self.normal_channel = normal_channel
        self.pool = pool
        if normal_channel:
            channel = 6
        else:
            channel = 3
        part_num = len(seg_classes[categories])
        self.part_num = part_num
        self.stn = STN3d(channel, Linear)
        self.conv1 = Conv1d(channel, 64, nn.Linear)
        self.conv2 = Conv1d(64, 128, Linear)
        self.conv3 = Conv1d(128, 128, Linear)
        self.conv4 = Conv1d(128, 512, Linear)
        self.conv5 = Conv1d(512, 2048, Linear)
        self.bn1 = nn.BatchNorm1d(64, affine=affine)
        self.bn2 = nn.BatchNorm1d(128, affine=affine)
        self.bn3 = nn.BatchNorm1d(128, affine=affine)
        self.bn4 = nn.BatchNorm1d(512, affine=affine)
        self.bn5 = nn.BatchNorm1d(2048, affine=affine)
        self.fstn = STNkd(k=128, Linear=Linear)
        if concat_global:
            self.convs1 = Conv1d(4928, 256, Linear)  # modified, one model for one class
        else:
            self.convs1 = Conv1d(2880, 256, Linear)

        self.convs2 = Conv1d(256, 256, Linear)
        self.convs3 = Conv1d(256, 128, Linear)
        self.convs4 = Conv1d(128, part_num, nn.Linear)
        self.bns1 = nn.BatchNorm1d(256, affine=affine)
        self.bns2 = nn.BatchNorm1d(256, affine=affine)
        self.bns3 = nn.BatchNorm1d(128, affine=affine)

        self.Tnet = Tnet
        self.concat_global = concat_global

    def forward(self, data):
        pos = data.pos  # xyz
        normal = data.x  # normals
        batch = torch.max(data.batch) + 1

        pos_list = []
        normal_list = []
        for i in range(batch):
            pos_list.append(pos[data.batch == i])
            normal_list.append(normal[data.batch == i])

        pos = torch.stack(pos_list).permute(0, 2, 1).contiguous()
        normal = torch.stack(normal_list).permute(0, 2, 1).contiguous()
        if self.normal_channel:
            point_cloud = torch.cat([pos, normal], dim=1)
        else:
            point_cloud = pos

        B, D, N = point_cloud.size()

        if self.Tnet:
            trans = self.stn(point_cloud)
            point_cloud = point_cloud.transpose(2, 1)
            if D > 3:
                point_cloud, feature = point_cloud.split(3, dim=2)
            point_cloud = torch.bmm(point_cloud, trans)
            if D > 3:
                point_cloud = torch.cat([point_cloud, feature], dim=2)
            point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        if self.Tnet:
            trans_feat = self.fstn(out3)
            x = out3.transpose(2, 1)
            net_transformed = torch.bmm(x, trans_feat)
            net_transformed = net_transformed.transpose(2, 1)
        else:
            net_transformed = out3

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        if self.pool == 'max':
            out_pool = torch.max(out5, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            out_pool = torch.mean(out5, 2, keepdim=True)
        elif self.pool == 'bmax':
            out_pool = torch.max(out5, 2, keepdim=True)[0] + offset_map[N]
        out_pool = out_pool.view(-1, 2048)

        expand = out_pool.view(-1, 2048, 1).repeat(1, 1, N)
        if self.concat_global:
            concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        else:
            concat = torch.cat([out1, out2, out3, out4, out5], 1)
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        net = net.view(-1, self.part_num)

        out_dict = {'out': net}
        if self.Tnet:
            out_dict['trans'] = trans
            out_dict['trans_feat'] = trans_feat
        return out_dict


class BasicPointNetSemSeg(nn.Module):
    def __init__(self, num_class=13, more_features=True, Linear=nn.Linear, pool='max', affine=True):
        super().__init__()
        self.more_features = more_features
        if more_features:
            channel = 9
        else:
            channel = 3
        self.k = num_class
        self.feat = PointNetEncoder(Linear, global_feat=False, feature_transform=True, channel=channel, pool=pool)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512, affine=affine)
        self.bn2 = nn.BatchNorm1d(256, affine=affine)
        self.bn3 = nn.BatchNorm1d(128, affine=affine)

    def forward(self, data):
        pos = data.pos  # xyz

        # ref: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/S3DISDataLoader.py
        rgb = data.x  # rgb and 3 additional features

        batch = torch.max(data.batch) + 1

        pos_list = []
        rgb_list = []
        for i in range(batch):
            pos_list.append(pos[data.batch == i])
            rgb_list.append(rgb[data.batch == i])

        pos = torch.stack(pos_list).permute(0, 2, 1).contiguous()
        rgb = torch.stack(rgb_list).permute(0, 2, 1).contiguous()
        if self.more_features:
            point_cloud = torch.cat([pos, rgb], dim=1)
        else:
            point_cloud = pos

        x, trans, trans_feat = self.feat(point_cloud)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(-1, self.k)

        return {
            'out': x,
            'trans': trans,
            'trans_feat': trans_feat,
        }


class BasicBiPointNetPartSeg(nn.Module):
    def __init__(self, categories='Airplane', normal_channel=True, Linear=BiLinear, pool='max', affine=True, Tnet=True, concat_global=True):
        super(BasicBiPointNetPartSeg, self).__init__()
        self.normal_channel = normal_channel
        self.pool = pool
        if normal_channel:
            channel = 6
        else:
            channel = 3
        part_num = len(seg_classes[categories])
        self.part_num = part_num
        self.stn = BiSTN3d(channel, Linear, pool=pool, affine=affine)
        self.conv1 = Conv1d(channel, 64, nn.Linear)
        self.conv2 = Conv1d(64, 128, Linear)
        self.conv3 = Conv1d(128, 128, Linear)
        self.conv4 = Conv1d(128, 512, Linear)
        self.conv5 = Conv1d(512, 2048, Linear)
        self.bn1 = nn.BatchNorm1d(64, affine=affine)
        self.bn2 = nn.BatchNorm1d(128, affine=affine)
        self.bn3 = nn.BatchNorm1d(128, affine=affine)
        self.bn4 = nn.BatchNorm1d(512, affine=affine)
        self.bn5 = nn.BatchNorm1d(2048, affine=affine)
        self.fstn = BiSTNkd(k=128, Linear=Linear, pool=pool, affine=affine)
        # self.convs1 = Conv1d(4944, 256, Linear)
        if concat_global:
            self.convs1 = Conv1d(4928, 256, Linear)  # modified, one model for one class
        else:
            self.convs1 = Conv1d(2880, 256, Linear)

        self.convs2 = Conv1d(256, 256, Linear)
        self.convs3 = Conv1d(256, 128, Linear)
        self.convs4 = Conv1d(128, part_num, nn.Linear)
        self.bns1 = nn.BatchNorm1d(256, affine=affine)
        self.bns2 = nn.BatchNorm1d(256, affine=affine)
        self.bns3 = nn.BatchNorm1d(128, affine=affine)

        self.Tnet = Tnet
        self.concat_global = concat_global

    def forward(self, data):
        pos = data.pos  # xyz
        normal = data.x  # normals
        batch = torch.max(data.batch) + 1

        pos_list = []
        normal_list = []
        for i in range(batch):
            pos_list.append(pos[data.batch == i])
            normal_list.append(normal[data.batch == i])

        pos = torch.stack(pos_list).permute(0, 2, 1).contiguous()
        normal = torch.stack(normal_list).permute(0, 2, 1).contiguous()
        if self.normal_channel:
            point_cloud = torch.cat([pos, normal], dim=1)
        else:
            point_cloud = pos

        B, D, N = point_cloud.size()

        if self.Tnet:
            trans = self.stn(point_cloud)
            point_cloud = point_cloud.transpose(2, 1)
            if D > 3:
                point_cloud, feature = point_cloud.split(3, dim=2)
            point_cloud = torch.bmm(point_cloud, trans)
            if D > 3:
                point_cloud = torch.cat([point_cloud, feature], dim=2)
            point_cloud = point_cloud.transpose(2, 1)

        out1 = F.hardtanh(self.bn1(self.conv1(point_cloud)))
        out2 = F.hardtanh(self.bn2(self.conv2(out1)))
        out3 = F.hardtanh(self.bn3(self.conv3(out2)))

        if self.Tnet:
            trans_feat = self.fstn(out3)
            x = out3.transpose(2, 1)
            net_transformed = torch.bmm(x, trans_feat)
            net_transformed = net_transformed.transpose(2, 1)
        else:
            net_transformed = out3

        out4 = F.hardtanh(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        if self.pool == 'max':
            out_pool = torch.max(out5, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            out_pool = torch.mean(out5, 2, keepdim=True)
        elif self.pool == 'ema-max':
            out_pool = torch.max(out5, 2, keepdim=True)[0] + offset_map[N]
        out_pool = out_pool.view(-1, 2048)

        # out_pool = torch.cat([out_pool, label.squeeze(1)], 1)
        # expand = out_pool.view(-1, 2048+16, 1).repeat(1, 1, N)
        expand = out_pool.view(-1, 2048, 1).repeat(1, 1, N)
        if self.concat_global:
            concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        else:
            concat = torch.cat([out1, out2, out3, out4, out5], 1)

        net = F.hardtanh(self.bns1(self.convs1(concat)))
        net = F.hardtanh(self.bns2(self.convs2(net)))
        net = F.hardtanh(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        net = net.view(-1, self.part_num)

        out_dict = {'out': net}
        if self.Tnet:
            out_dict['trans'] = trans
            out_dict['trans_feat'] = trans_feat
        return out_dict


class BasicBiPointNetSemSeg(nn.Module):
    def __init__(self, num_class=13, more_features=True, Linear=BiLinear, pool='max', affine=True):
        super(BasicBiPointNetSemSeg, self).__init__()
        self.more_features = more_features
        if more_features:
            channel = 9
        else:
            channel = 3
        self.k = num_class
        self.feat = BiPointNetEncoder(Linear, global_feat=False, feature_transform=True, channel=channel, pool=pool)
        self.conv1 = Conv1d(1088, 512, Linear)
        self.conv2 = Conv1d(512, 256, Linear)
        self.conv3 = Conv1d(256, 128, Linear)
        self.conv4 = Conv1d(128, self.k, nn.Linear)
        self.bn1 = nn.BatchNorm1d(512, affine=affine)
        self.bn2 = nn.BatchNorm1d(256, affine=affine)
        self.bn3 = nn.BatchNorm1d(128, affine=affine)

    def forward(self, data):
        pos = data.pos  # xyz

        # ref: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/S3DISDataLoader.py
        rgb = data.x  # rgb and 3 additional features

        batch = torch.max(data.batch) + 1

        pos_list = []
        rgb_list = []
        for i in range(batch):
            pos_list.append(pos[data.batch == i])
            rgb_list.append(rgb[data.batch == i])

        pos = torch.stack(pos_list).permute(0, 2, 1).contiguous()
        rgb = torch.stack(rgb_list).permute(0, 2, 1).contiguous()
        if self.more_features:
            point_cloud = torch.cat([pos, rgb], dim=1)
        else:
            point_cloud = pos

        x, trans, trans_feat = self.feat(point_cloud)
        x = F.hardtanh(self.bn1(self.conv1(x)))
        x = F.hardtanh(self.bn2(self.conv2(x)))
        x = F.hardtanh(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(-1, self.k)

        return {
            'out': x,
            'trans': trans,
            'trans_feat': trans_feat,
        }


@MODELS.register
class PointNet(BasicPointNet):
    def __init__(self):
        super().__init__()


@MODELS.register
class PointNetMean(BasicPointNet):
    def __init__(self):
        super().__init__(pool='mean')


@MODELS.register
class PointNetVanillaMax(BasicPointNet):
    def __init__(self):
        super().__init__(pool='max', tnet=False)


@MODELS.register
class PointNetVanillaMean(BasicPointNet):
    def __init__(self):
        super().__init__(pool='mean', tnet=False)


@MODELS.register
class BiPointNetBNNMax(BasicBiPointNet):
    def __init__(self):
        super().__init__(Linear=BiLinear, pool='max')


@MODELS.register
class BiPointNetBNNMean(BasicBiPointNet):
    def __init__(self):
        super().__init__(Linear=BiLinear, pool='mean')


@MODELS.register
class BiPointNetBNNEMax(BasicBiPointNet):
    def __init__(self):
        super().__init__(Linear=BiLinear, pool='ema-max')


@MODELS.register
class BiPointNetLSRMax(BasicBiPointNet):
    def __init__(self):
        super().__init__(Linear=BiLinearLSR, pool='max')


@MODELS.register
class BiPointNetLSRMean(BasicBiPointNet):
    def __init__(self):
        super().__init__(Linear=BiLinearLSR, pool='mean')


@MODELS.register
class BiPointNetLSREMax(BasicBiPointNet):
    def __init__(self):
        super().__init__(Linear=BiLinearLSR, pool='ema-max')


@MODELS.register
class BiPointNetIRNetMean(BasicBiPointNet):
    def __init__(self):
        super().__init__(Linear=BiLinearIRNet, pool='mean')


@MODELS.register
class BiPointNetIRNetMax(BasicBiPointNet):
    def __init__(self):
        super().__init__(Linear=BiLinearIRNet, pool='max')


@MODELS.register
class BiPointNetIRNetEMax(BasicBiPointNet):
    def __init__(self):
        super().__init__(Linear=BiLinearIRNet, pool='ema-max')


@MODELS.register
class BiPointNetXNORMax(BasicBiPointNet):
    def __init__(self):
        super().__init__(Linear=BiLinearXNOR, pool='max')


@MODELS.register
class BiPointNetXNORMean(BasicBiPointNet):
    def __init__(self):
        super().__init__(Linear=BiLinearXNOR, pool='mean')


@MODELS.register
class BiPointNetXNOREMax(BasicBiPointNet):
    def __init__(self):
        super().__init__(Linear=BiLinearXNOR, pool='ema-max')


@MODELS.register
class BiPointNetBiRealMax(BasicBiPointNet):
    def __init__(self):
        super().__init__(Linear=BiLinearBiReal, pool='max')


@MODELS.register
class BiPointNetBiRealMean(BasicBiPointNet):
    def __init__(self):
        super().__init__(Linear=BiLinearBiReal, pool='mean')


@MODELS.register
class BiPointNetBiRealEMax(BasicBiPointNet):
    def __init__(self):
        super().__init__(Linear=BiLinearBiReal, pool='ema-max')


@MODELS.register
class BiPointNetVanillaXNORMax(BasicBiPointNet):
    def __init__(self):
        super().__init__(Linear=BiLinearXNOR, pool='max', tnet=False)


@MODELS.register
class BiPointNetVanillaLSREMax(BasicBiPointNet):
    def __init__(self):
        super().__init__(Linear=BiLinearLSR, pool='ema-max', tnet=False)


@MODELS.register
class PointNetPartSegMax(BasicPointNetPartSeg):
    def __init__(self, categories='Airplane'):
        super().__init__(categories=categories, pool='max')


@MODELS.register
class PointNetPartSegMean(BasicPointNetPartSeg):
    def __init__(self, categories='Airplane'):
        super().__init__(categories=categories, pool='mean')


@MODELS.register
class BiPointNetPartSegBNNMax(BasicBiPointNetPartSeg):
    def __init__(self, categories='Airplane'):
        super().__init__(categories=categories, Linear=BiLinear, pool='max')


@MODELS.register
class BiPointNetPartSegBNNMean(BasicBiPointNetPartSeg):
    def __init__(self, categories='Airplane'):
        super().__init__(categories=categories, Linear=BiLinear, pool='mean')


@MODELS.register
class BiPointNetPartSegBNNEMax(BasicBiPointNetPartSeg):
    def __init__(self, categories='Airplane'):
        super().__init__(categories=categories, Linear=BiLinear, pool='ema-max')


@MODELS.register
class BiPointNetPartSegLSRMax(BasicBiPointNetPartSeg):
    def __init__(self, categories='Airplane'):
        super().__init__(categories=categories, Linear=BiLinearLSR, pool='max')


@MODELS.register
class BiPointNetPartSegLSRMean(BasicBiPointNetPartSeg):
    def __init__(self, categories='Airplane'):
        super().__init__(categories=categories, Linear=BiLinearLSR, pool='mean')


@MODELS.register
class BiPointNetPartSegLSREMax(BasicBiPointNetPartSeg):
    def __init__(self, categories='Airplane'):
        super().__init__(categories=categories, Linear=BiLinearLSR, pool='ema-max')


@MODELS.register
class PointNetSemSegMax(BasicPointNetSemSeg):
    def __init__(self):
        super().__init__(pool='max')


@MODELS.register
class PointNetSemSegMean(BasicPointNetSemSeg):
    def __init__(self):
        super().__init__(pool='mean')


@MODELS.register
class BiPointNetSemSegBNNMax(BasicBiPointNetSemSeg):
    def __init__(self):
        super().__init__(Linear=BiLinear, pool='max')


@MODELS.register
class BiPointNetSemSegBNNMean(BasicBiPointNetSemSeg):
    def __init__(self):
        super().__init__(Linear=BiLinear, pool='mean')


@MODELS.register
class BiPointNetSemSegBNNEMax(BasicBiPointNetSemSeg):
    def __init__(self):
        super().__init__(Linear=BiLinear, pool='ema-max')


@MODELS.register
class BiPointNetSemSegLSRMax(BasicBiPointNetSemSeg):
    def __init__(self):
        super().__init__(Linear=BiLinearLSR, pool='max')


@MODELS.register
class BiPointNetSemSegLSRMean(BasicBiPointNetSemSeg):
    def __init__(self):
        super().__init__(Linear=BiLinearLSR, pool='mean')


@MODELS.register
class BiPointNetSemSegLSREMax(BasicBiPointNetSemSeg):
    def __init__(self):
        super().__init__(Linear=BiLinearLSR, pool='ema-max')
