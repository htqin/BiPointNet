import argparse
from os import makedirs, remove, listdir
from os.path import join, isdir
from importlib import import_module
from glob import glob
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
import math
import json
import numpy as np

from models import MODELS
from datasets import DATASETS, DATALOADERS
from train import TRAINS, OPTIMIZERS, LR_SCHEDULER
from test import TESTS
from utils import _init_fn, set_seed, parse_cfg, create_logger, resume_last, resume_best, resume_from, save_model, apply_dataparallel, model_to_device


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg')
    parser.add_argument('--resume_best', action='store_true', help='Resume the model from the best checkpoint')
    parser.add_argument('--resume_last', action='store_true', help='Resume the model from the latest checkpoint')
    parser.add_argument('--resume_from', help='Resume from a specified checkpoint')
    parser.add_argument('--non_deterministic', help='Do not use a fixed seed')
    parser.add_argument('--no_save', action='store_true', help='Do not save checkpoints')
    parser.add_argument('--max_epoch', type=int, default=200, help='Max epoch during training')
    parser.add_argument('--runs', type=int, default=1, help='Number of times to run')
    args = parser.parse_args()
    cfg, stem = parse_cfg(args.cfg)
    if not args.non_deterministic:
        set_seed(666)

    # work directory
    work_dir = join('work_dirs', stem)
    if not isdir(work_dir):
        makedirs(work_dir)

    # create logger
    logger = create_logger(work_dir)

    best_test_acc_runs = []

    # multiple runs
    if cfg['dataset']['type'] == 'ShapeNet':
        if cfg['dataset']['categories'] == 'All':
            run_args = ['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife', 'Lamp',
                        'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table']
        else:
            run_args = [cfg['dataset']['categories']]
    elif cfg['dataset']['type'] == 'S3DIS':
        if cfg['dataset']['test_area'] == 'All':
            run_args = [1, 2, 3, 4, 5, 6]
        else:
            run_args = [cfg['dataset']['test_area']]
    else:
        run_args = [None for i in range(args.runs)]

    for run_idx, run_arg in enumerate(run_args):
        if cfg['dataset']['type'] == 'ShapeNet':
            cfg['dataset']['categories'] = run_arg
            cfg['model']['categories'] = run_arg
        if cfg['dataset']['type'] == 'S3DIS':
            cfg['dataset']['test_area'] = run_arg

        train_dataset = DATASETS(cfg['dataset'], train=True)
        test_dataset = DATASETS(cfg['dataset'], train=False)
        train_loader = DATALOADERS(cfg.get('dataloader', None), train_dataset, shuffle=True, worker_init_fn=_init_fn)
        test_loader = DATALOADERS(cfg.get('dataloader', None), test_dataset, shuffle=False, worker_init_fn=_init_fn)
        model = MODELS(cfg['model'])
        if args.resume_last:
            checkpoint, last_epoch, best_epoch, best_test_acc = resume_last(model, work_dir, cfg['model'])
            logger.info('Resumed from ' + checkpoint)
        elif args.resume_best:
            checkpoint, last_epoch, best_epoch, best_test_acc = resume_best(model, work_dir, cfg['model'])
        elif args.resume_from:
            last_epoch, best_epoch, best_test_acc = resume_from(model, args.resume_from, cfg['model'])
            logger.info('Resumed from ' + args.resume_from)
        else:
            last_epoch = best_test_acc = best_epoch = 0

        from torch_geometric.data import Batch
        if cfg.get('scale_dynamic_init'):
            for data in train_loader:
                data = Batch.from_data_list(data)
                model(data)
                break

        model = apply_dataparallel(model, cfg['model'])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model_to_device(model, device, cfg['model'])

        # init trainer
        train = TRAINS(cfg.get('train', None))

        # init tester
        test = TESTS(cfg.get('test', None))

        # init optimizer
        optimizer = OPTIMIZERS(cfg.get('optimizer', None), model, cfg['model'])
        if cfg.get('lr_scheduler', False):
            scheduler = LR_SCHEDULER(cfg['lr_scheduler'], optimizer)

        # training and testing
        def Log_UP(epoch, K_min=1e-3, K_max=1e1):
            Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
            return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / 200 * epoch)], requires_grad=False).float().cuda()

        print('Run {} / {}, Start Training...'.format(run_idx+1, args.runs))
        if cfg['dataset']['type'] == 'ShapeNet':
            logger.info('vvv Part Seg Cat {}'.format(run_arg))

        for epoch in range(last_epoch + 1, args.max_epoch + 1):

            if 'IRNet' in cfg['model']['type']:
                t = Log_UP(epoch)
                k = 1 / t if t < 1 else torch.tensor([1], requires_grad=False).float().cuda()
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if '.k' in name:
                            param.copy_(k)
                        elif '.t' in name:
                            param.copy_(t)

            loss = train(model, train_loader, optimizer)
            test_acc = test(model, test_loader)
            test_acc_dict = None
            if isinstance(test_acc, dict):
                test_acc_dict = test_acc
                test_acc = test_acc_dict.pop('primary')

            if cfg.get('lr_scheduler', False):
                scheduler.step()

            # save the best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                last_best_epoch, best_epoch = best_epoch, epoch

                if cfg['dataset']['type'] == 'S3DIS':
                    with open(join(work_dir, '{}.json'.format(cfg['dataset']['test_area'])), 'w') as f:
                        json.dump(test_acc_dict, f, indent=4)

                if not args.no_save:
                    # remove the previous best epoch model
                    if last_best_epoch > 0:
                        try:
                            remove(join(work_dir, 'ckp_best_epoch_{:03d}.pth'.format(last_best_epoch)))
                        except:
                            print('Unable to remove', join(work_dir, 'ckp_best_epoch_{:03d}.pth'.format(last_best_epoch)))

                    best_save_pathname = join(work_dir, 'ckp_best_epoch_{:03d}.pth'.format(best_epoch))
                    save_model(model, best_save_pathname, cfg['model'])

            # save the latest model
            if not args.no_save:
                latest_save_pathname = join(work_dir, 'ckp_last_epoch_{:03d}.pth'.format(epoch))
                save_model(model, latest_save_pathname, cfg['model'])
                if epoch - 1 > 0:
                    try:
                        remove(join(work_dir, 'ckp_last_epoch_{:03d}.pth'.format(epoch - 1)))
                    except:
                        print('Unable to remove', join(work_dir, 'ckp_last_epoch_{:03d}.pth'.format(epoch - 1)))

            if isinstance(loss, dict):
                loss_info = ''.join(['{}: {:.4f}, '.format(loss_name, loss_value) for loss_name, loss_value in loss.items()])
            else:
                loss_info = 'Loss: {:.4f}, '.format(loss)

            if test_acc_dict is None:
                test_info = ''
            else:
                test_info = ''.join(['{}: {:.4f}, '.format(metric_name, test_value) for
                                     metric_name, test_value in test_acc_dict.items() if isinstance(test_value, float)])
            logger.info(
                'Epoch: {:03d}, {}Test: {:.4f}, Best Epoch: {:03d}, Best Test: {:.4f}, {}'.format(
                    epoch, loss_info, test_acc, best_epoch, best_test_acc, test_info))

        best_test_acc_runs.append(best_test_acc)

    print('--------------------')
    for run_idx, best_test_acc in enumerate(best_test_acc_runs):
        print('Run {} Best Test: {:.4f}'.format(run_idx+1, best_test_acc))
    print('Average Best Test of {} Runs: {:.4f}'.format(len(best_test_acc_runs),
                                                        sum(best_test_acc_runs)/len(best_test_acc_runs)))

    if cfg['dataset']['type'] == 'S3DIS':
        i_total = np.zeros((13, ))
        u_total = np.zeros((13, ))
        correct = 0
        total = 0
        for i in [1, 2, 3, 4, 5, 6]:
            with open(join(work_dir, '{}.json'.format(i)), 'r') as f:
                content = json.load(f)
            print('Area {}: mIoU = {}, accuracy = {}'.format(i, content['mean_IoU'], content['accuracy']))
            i_total += np.array(content['i_total'])
            u_total += np.array(content['u_total'])
            correct += content['correct']
            total += content['total']
        print('IoU = ', i_total / u_total)
        print('mean IoU = ', (i_total / u_total).sum() / 13.0)
        print('correct =', correct, 'total =', total)
        print('accuracy =', correct / total)

