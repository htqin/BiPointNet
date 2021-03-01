import torch
import torch.nn.functional as F
from utils import FunctionRegistry

from collections import defaultdict, namedtuple
import numpy as np
import time
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree

from torch_geometric.nn import radius
from torch_geometric.data import Data
from torch_scatter import scatter_add
import torch.multiprocessing as mp


train_default_cfg = {
    'type': 'train'
}

TRAINS = FunctionRegistry(default_cfg=train_default_cfg)


# ref: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/5e31f2150f2e3e4f726046f2e956632ff4f36759/models/pointnet.py
def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss


@TRAINS.register
def train(model, train_loader, optimizer, regularizer_loss_weight=0.001):
    model.train()

    total_loss = 0
    total_cls_loss = 0
    total_regularizer_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out_dict = model(data)

        out = out_dict['out']
        data_y = torch.cat([d.y for d in data]).to(out.device)
        cls_loss = F.nll_loss(out, data_y)
        total_cls_loss += cls_loss.item() * len(data)

        # orthogonal matrix regularizar loss
        if 'trans_feat' in out_dict and out_dict['trans_feat'] is not None:
            trans_feat = out_dict['trans_feat']
            regularizer_loss = feature_transform_reguliarzer(trans_feat)
            total_regularizer_loss += regularizer_loss.item() * len(data)

            loss = cls_loss + regularizer_loss_weight * regularizer_loss
        else:
            loss = cls_loss

        loss.backward()
        total_loss += loss.item() * len(data)
        optimizer.step()
    return {
        'loss': total_loss / len(train_loader.dataset),
        'cls_loss': total_cls_loss / len(train_loader.dataset),
        'regularizer_loss': total_regularizer_loss / len(train_loader.dataset)
    }


@TRAINS.register
def train_part_seg(model, train_loader, optimizer, regularizer_loss_weight=0.001):
    model.train()

    total_loss = 0
    total_cls_loss = 0
    total_regularizer_loss = 0
    for data in train_loader:
        optimizer.zero_grad()

        # batch size must > 2 per gpu, or bn will fail
        if len(data) < 2 * torch.cuda.device_count(): # num of gpus
            data = [data[i] for i in np.random.choice(len(data), 2 * torch.cuda.device_count()).tolist()]
        out_dict = model(data)

        out = out_dict['out']
        data_y = torch.cat([d.y for d in data]).to(out.device)
        data_y = data_y - data_y.min()  # part class id is added on
        cls_loss = F.nll_loss(out, data_y)
        total_cls_loss += cls_loss.item() * len(data)

        # orthogonal matrix regularizar loss
        trans_feat = out_dict['trans_feat']
        regularizer_loss = feature_transform_reguliarzer(trans_feat)
        total_regularizer_loss += regularizer_loss.item() * len(data)

        loss = cls_loss + regularizer_loss_weight * regularizer_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(data)
    return {
        'loss': total_loss / len(train_loader.dataset),
        'cls_loss': total_cls_loss / len(train_loader.dataset),
        'regularizer_loss': total_regularizer_loss / len(train_loader.dataset)
    }


@TRAINS.register
def train_sem_seg(model, train_loader, optimizer, regularizer_loss_weight=0.001):
    model.train()

    total_loss = 0
    total_cls_loss = 0
    total_regularizer_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        
        # batch size must > 2 per gpu, or bn will fail
        if len(data) < 2 * torch.cuda.device_count(): # num of gpus
            data = [data[i] for i in np.random.choice(len(data), 2 * torch.cuda.device_count()).tolist()]

        out_dict = model(data)

        out = out_dict['out']
        data_y = torch.cat([d.y for d in data]).to(out.device)

        # # ref: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/S3DISDataLoader.py

        weight = train_loader.dataset.weights.to(out.device)
        cls_loss = F.nll_loss(out, data_y, weight=weight)
        # loss = F.nll_loss(out, data_y)
        total_cls_loss += cls_loss.item() * len(data)

        # orthogonal matrix regularizar loss
        trans_feat = out_dict['trans_feat']
        regularizer_loss = feature_transform_reguliarzer(trans_feat)
        total_regularizer_loss += regularizer_loss.item() * len(data)

        loss = cls_loss + regularizer_loss_weight * regularizer_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(data)

    return {
        'loss': total_loss / len(train_loader.dataset),
        'cls_loss': total_cls_loss / len(train_loader.dataset),
        'regularizer_loss': total_regularizer_loss / len(train_loader.dataset)
    }
