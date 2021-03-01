from ._registry import DATASETS
from .modelnet import *
from .modelnet2 import *
from .shapenet import *
from .s3dis import *

from ._registry import DATALOADERS
from .datalistloader import *


__all__ = ['DATASETS', 'DATALOADERS']
