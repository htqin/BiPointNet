from os.path import basename, splitext, join, isfile, isdir
from os import makedirs
import pdb
import sys
from glob import glob
import yaml
import logging
from logging import handlers
from datetime import datetime
from torch_geometric.nn import DataParallel
import torch
import numpy as np
import random
from torch.nn import Parameter


# TODO: take a look at
# https://stackoverflow.com/questions/5189232/how-to-auto-register-a-class-when-its-defined

class Registry(object):
    """Registry for classes"""

    def __init__(self, default_cfg):
        self.module_dict = {}
        self.default_cfg = default_cfg

    def __call__(self, model_cfg, *args, **kwargs):
        if model_cfg is None:
            model_cfg = self.default_cfg

        model_name = model_cfg.pop('type')
        module = self.module_dict[model_name](*args, **model_cfg, **kwargs)
        model_cfg['type'] = model_name
        return module

    def register(self, module):
        assert module.__name__ not in self.module_dict
        self.module_dict[module.__name__] = module


class FunctionRegistry(object):
    """Registry for functions"""

    def __init__(self, default_cfg):
        self.func_dict = {}
        self.default_cfg = default_cfg

    def __call__(self, func_cfg):
        if func_cfg is None:
            func_cfg = self.default_cfg

        func_name = func_cfg.pop('type')
        func = self.func_dict[func_name]
        func_cfg['type'] = func_name
        return func

    def register(self, func):
        assert func.__name__ not in self.func_dict
        self.func_dict[func.__name__] = func


class OptimizerRegistry(Registry):
    def __init__(self, default_cfg):
        super().__init__(default_cfg)

    def __call__(self, optimizer_cfg, model, cfgs, **kwargs):
        if optimizer_cfg is None:
            optimizer_cfg = self.default_cfg

        optimizer_name = optimizer_cfg.pop('type')
        optimizer = self.module_dict[optimizer_name](model.parameters(), **optimizer_cfg, **kwargs)
        optimizer_cfg['type'] = optimizer_name
        return optimizer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def _init_fn(worker_id): 
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    torch.cuda.manual_seed_all(10)


def parse_cfg(path):
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    stem, ext = splitext(basename(path))
    return cfg, stem


def create_logger(work_dir):
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    logger.addHandler(ch)

    fh = handlers.RotatingFileHandler(join(work_dir, 'log.txt'), mode='a')
    fh.setFormatter(format)
    logger.addHandler(fh)

    return logger


def resume_last(model, work_dir, cfgs):
    """ parse log to recover from previous training session: load the last epoch """

    log_pathname = join(work_dir, 'log.txt')
    assert isfile(log_pathname), 'No log file found, cannot resume.'
    with open(log_pathname, 'r') as f:
        content = f.readlines()
        idx = 1
        last_line = content[-1]
        while "Resumed from" in last_line:
            idx += 1
            last_line = content[-idx]
    _, _, info = last_line.split('|')
    last_epoch, last_loss, last_test_acc, best_epoch, best_test_acc = info.split(',')

    checkpoint = join(work_dir, 'ckp_last_epoch_{:03d}.pth'.format(int(last_epoch[-3:])))
    last_epoch = int(last_epoch[-3:])
    best_epoch = int(best_epoch[-3:])
    best_test_acc = float(best_test_acc[-6:])

    model.load_state_dict(torch.load(checkpoint))

    return checkpoint, last_epoch, best_epoch, best_test_acc


def resume_best(model, work_dir, cfgs):
    """ parse log to recover from previous training session: load the last epoch """

    log_pathname = join(work_dir, 'log.txt')
    assert isfile(log_pathname), 'No log file found, cannot resume.'
    with open(log_pathname, 'r') as f:
        content = f.readlines()
        idx = 1
        last_line = content[-1]
        while "Resumed from" in last_line:
            idx += 1
            last_line = content[-idx]
    _, _, info = last_line.split('|')
    last_epoch, last_loss, last_test_acc, best_epoch, best_test_acc = info.split(',')

    checkpoint = join(work_dir, 'ckp_best_epoch_{:03d}.pth'.format(int(best_epoch[-3:])))
    last_epoch = int(last_epoch[-3:])
    best_epoch = int(best_epoch[-3:])
    best_test_acc = float(best_test_acc[-6:])

    model.load_state_dict(torch.load(checkpoint))

    return checkpoint, last_epoch, best_epoch, best_test_acc


def resume_from(model, checkpoint, cfgs):
    """ parse log to recover from previous training session: load a specific checkpoint """

    stem, _ = splitext(basename(checkpoint))
    _, _, _, last_epoch = stem.split('_')
    last_epoch = int(last_epoch)
    best_epoch = best_test_acc = -1
    model.load_state_dict(torch.load(checkpoint))

    return last_epoch, best_epoch, best_test_acc


def save_model(model, save_pathname, cfgs):
    torch.save(model.module.state_dict(), save_pathname)


def apply_dataparallel(model, cfgs):
    return DataParallel(model)


def model_to_device(model, device, cfgs):
    model.to(device)
