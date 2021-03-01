from utils import Registry

dataset_default_cfg = {
    'type': 'ModelNet'
}

dataloader_default_cfg = {
    'type': 'DataListLoader'
}

DATASETS = Registry(default_cfg=dataset_default_cfg)
DATALOADERS = Registry(default_cfg=dataloader_default_cfg)
