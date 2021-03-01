from utils import Registry, OptimizerRegistry
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR


optimizer_default_cfg = {
    'type': 'Adam',
    'lr': 0.001
}

lr_scheduler_default_cfg = None

OPTIMIZERS = OptimizerRegistry(default_cfg=optimizer_default_cfg)
OPTIMIZERS.register(Adam)
OPTIMIZERS.register(SGD)

LR_SCHEDULER = Registry(default_cfg=lr_scheduler_default_cfg)
LR_SCHEDULER.register(StepLR)
LR_SCHEDULER.register(CosineAnnealingLR)
LR_SCHEDULER.register(MultiStepLR)
