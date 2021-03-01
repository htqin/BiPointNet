from utils import Registry

model_default_cfg = {
    'type': 'PointNet2'
}

MODELS = Registry(default_cfg=model_default_cfg)
