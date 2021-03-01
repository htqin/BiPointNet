from ._registry import DATASETS
# from torch_geometric.datasets import ShapeNet as PyGShapeNet
import torch_geometric.transforms as T
import os
import os.path as osp
import shutil
import json
import torch
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.io import read_txt_array
from itertools import repeat
import numpy as np

# TODO
# Currently, repeat points is done by modifying the base ShapeNet class
# A better solution may be implementing a transformation T.RepeatPoints()
@DATASETS.register
class ShapeNet(InMemoryDataset):
    r"""The ShapeNet part level segmentation dataset from the `"A Scalable
    Active Framework for Region Annotation in 3D Shape Collections"
    <http://web.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf>`_
    paper, containing about 17,000 3D shape point clouds from 16 shape
    categories.
    Each category is annotated with 2 to 6 parts.
    Args:
        root (string): Root directory where the dataset should be saved.
        categories (string or [string], optional): The category of the CAD
            models (one or a combination of :obj:`"Airplane"`, :obj:`"Bag"`,
            :obj:`"Cap"`, :obj:`"Car"`, :obj:`"Chair"`, :obj:`"Earphone"`,
            :obj:`"Guitar"`, :obj:`"Knife"`, :obj:`"Lamp"`, :obj:`"Laptop"`,
            :obj:`"Motorbike"`, :obj:`"Mug"`, :obj:`"Pistol"`, :obj:`"Rocket"`,
            :obj:`"Skateboard"`, :obj:`"Table"`).
            Can be explicitly set to :obj:`None` to load all categories.
            (default: :obj:`None`)
        include_normals (bool, optional): If set to :obj:`False`, will not
            include normal vectors as input features. (default: :obj:`True`)
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"trainval"`, loads the training and validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"trainval"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = ('https://shapenet.cs.stanford.edu/media/'
           'shapenetcore_partanno_segmentation_benchmark_v0_normal.zip')

    category_ids = {
        'Airplane': '02691156',
        'Bag': '02773838',
        'Cap': '02954340',
        'Car': '02958343',
        'Chair': '03001627',
        'Earphone': '03261776',
        'Guitar': '03467517',
        'Knife': '03624134',
        'Lamp': '03636649',
        'Laptop': '03642806',
        'Motorbike': '03790512',
        'Mug': '03797390',
        'Pistol': '03948459',
        'Rocket': '04099429',
        'Skateboard': '04225987',
        'Table': '04379243',
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

    def __init__(self, root='data/ShapeNet', train=True, categories=None, include_normals=True,
                 split='trainval', transform=None, pre_transform=None,
                 pre_filter=None, repeat_to=None):  # Modified here to add repeat_to
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories

        # Default settings
        pre_transform = T.NormalizeScale()
        pre_filter = None
        include_normals = True

        if train:
            transform = T.Compose([
                T.RandomTranslate(0.01),
                T.RandomRotate(15, axis=0),
                T.RandomRotate(15, axis=1),
                T.RandomRotate(15, axis=2)
            ])
            split = 'trainval'
        else:
            transform = None
            split = 'test'

        super().__init__(root, transform, pre_transform, pre_filter)  # Modified here to add repeat_to

        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        elif split == 'trainval':
            path = self.processed_paths[3]
        else:
            raise ValueError((f'Split {split} found, but expected either '
                              'train, val, trainval or test'))

        self.data, self.slices = torch.load(path)
        self.data.x = self.data.x if include_normals else None

        self.y_mask = torch.zeros((len(self.seg_classes.keys()), 50),
                                  dtype=torch.bool)
        for i, labels in enumerate(self.seg_classes.values()):
            self.y_mask[i, labels] = 1

        self.repeat_to = repeat_to  # Modified here to add repeat_to

    @property
    def num_classes(self):
        return self.y_mask.size(-1)

    @property
    def raw_file_names(self):
        return list(self.category_ids.values()) + ['train_test_split']

    @property
    def processed_file_names(self):
        cats = '_'.join([cat[:3].lower() for cat in self.categories])
        return [
            os.path.join('{}_{}.pt'.format(cats, split))
            for split in ['train', 'val', 'test', 'trainval']
        ]

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        name = self.url.split('/')[-1].split('.')[0]
        os.rename(osp.join(self.root, name), self.raw_dir)

    def process_filenames(self, filenames):
        data_list = []
        categories_ids = [self.category_ids[cat] for cat in self.categories]
        cat_idx = {categories_ids[i]: i for i in range(len(categories_ids))}

        for name in filenames:
            cat = name.split(osp.sep)[0]
            if cat not in categories_ids:
                continue

            data = read_txt_array(osp.join(self.raw_dir, name))
            pos = data[:, :3]
            x = data[:, 3:6]
            y = data[:, -1].type(torch.long)

            data = Data(pos=pos, x=x, y=y, category=cat_idx[cat])
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list

    def process(self):
        trainval = []
        for i, split in enumerate(['train', 'val', 'test']):
            path = osp.join(self.raw_dir, 'train_test_split',
                            f'shuffled_{split}_file_list.json')
            with open(path, 'r') as f:
                filenames = [
                    osp.sep.join(name.split('/')[1:]) + '.txt'
                    for name in json.load(f)
                ]  # Removing first directory.
            data_list = self.process_filenames(filenames)
            if split == 'train' or split == 'val':
                trainval += data_list
            torch.save(self.collate(data_list), self.processed_paths[i])
        torch.save(self.collate(trainval), self.processed_paths[3])

    def __repr__(self):
        return '{}({}, categories={})'.format(self.__class__.__name__,
                                              len(self), self.categories)

    # Modified here, repeat points
    # overrides get from InMemoryDataset
    # ref: https://github.com/rusty1s/pytorch_geometric/blob/b5e06e8b6d8e27c24ada546482d2ae6c710bdac6/torch_geometric/data/in_memory_dataset.py#L68-L86
    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            # print(slices[idx], slices[idx + 1])
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data[key] = item[s]

        # ref: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/20409315f9a90279021a28346a7406abdf7849c3/data_utils/S3DISDataLoader.py#L141-L144
        if self.repeat_to is not None:
            x = data.x
            y = data.y
            pos = data.pos
            category = data.category

            # ref: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/20409315f9a90279021a28346a7406abdf7849c3/data_utils/S3DISDataLoader.py#L141-L144
            num_points = max([x.size(0), self.repeat_to])
            point_idxs = np.array([i for i in range(x.size(0))])
            point_idxs_repeat = np.random.choice(point_idxs, num_points - x.size(0))
            point_idxs_all = np.concatenate((point_idxs, point_idxs_repeat))
            np.random.shuffle(point_idxs_all)

            pos = pos[point_idxs_all, :]
            x = x[point_idxs_all, :]
            y = y[point_idxs_all]

            data = Data(pos=pos, x=x, y=y, category=category)

        return data
