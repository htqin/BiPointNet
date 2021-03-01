import torch
import torch.functional as F
from utils import FunctionRegistry
from torch_geometric.utils import intersection_and_union as i_and_u
import numpy as np

test_default_cfg = {
    'type': 'test'
}

TESTS = FunctionRegistry(default_cfg=test_default_cfg)


@TESTS.register
def test(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            pred = model(data)['out'].max(dim=1)[1]
        data_y = torch.cat([d.y for d in data]).to(pred.device)
        correct += pred.eq(data_y).sum().item()
    return correct / len(loader.dataset)


@TESTS.register
def test_part_seg(model, loader):

    model.eval()

    ious = [[] for _ in range(len(loader.dataset.categories))]

    for data in loader:
        out_dict = model(data)
        out = out_dict['out']
        pred = out.argmax(dim=1)

        # Adaptation to the datalistloader
        data_y = torch.cat([d.y for d in data]).to(out.device)
        data_batch = torch.cat([torch.ones(d.y.shape) * i for i, d in enumerate(data)]).type(torch.LongTensor).to(out.device)
        data_category = torch.cat([d.category for d in data])

        # loader.dataset.num_classes -> loader.dataset.y_mask.size(-1)
        # source code updated, see commint:
        # https://github.com/rusty1s/pytorch_geometric/commit/5c5509edbfd284e8c90b55faee9e68d7ae4f806a#diff-95045fdafc8ba001ebac808e65e54457
        # but the wheel is not updated to include the above changes in the ShapeNet class
        # (hence, missing overshadowing attribute num_classes)
        # this is a quick fix
        # update 2: we actually don't need y_mask in the computation
        # TODO: finalize the computation here
        # i, u = i_and_u(pred, data_y, loader.dataset.y_mask.size(-1), data_batch)
        # our modification
        data_y = data_y - data_y.min()  # part class id should start at 0
        # import pdb; pdb.set_trace()
        i, u = i_and_u(pred, data_y, data_y.max() + 1, data_batch)

        iou = i.cpu().to(torch.float) / u.cpu().to(torch.float)
        iou[torch.isnan(iou)] = 1

        # Find and filter the relevant classes for each category.
        for iou, category in zip(iou.unbind(), data_category):
            ious[category.item()].append(iou)

    # Compute mean IoU.
    ious = [torch.stack(iou).mean(0).mean(0) for iou in ious]
    return torch.tensor(ious).mean().item()


@TESTS.register
def test_sem_seg(model, loader):

    model.eval()

    correct = 0
    total = 0
    i_total = np.zeros((13, ))
    u_total = np.zeros((13, ))
    for data in loader:

        out_dict = model(data)
        out = out_dict['out']
        pred = out.argmax(dim=1)

        # Adaptation to the datalistloader
        data_y = torch.cat([d.y for d in data]).to(out.device)
        data_batch = torch.cat([torch.ones(d.y.shape) * i for i, d in enumerate(data)]).type(torch.LongTensor).to(out.device)

        correct += (pred == data_y).sum()
        total += pred.size(0)

        # S3DIS: 13 classes
        i, u = i_and_u(pred, data_y, 13, data_batch)
        i_total += i.cpu().to(torch.float).numpy().sum(axis=0)
        u_total += u.cpu().to(torch.float).numpy().sum(axis=0)

    # Compute point accuracy
    accuracy = correct.type(torch.FloatTensor) / total

    # Compute mean IoU

    ious = i_total / u_total
    mean_iou = ious.sum() / 13.0

    return {
        'primary': mean_iou.item(),
        'i_total': i_total.tolist(),
        'u_total': u_total.tolist(),
        'mean_IoU': mean_iou.item(),
        'correct': correct.item(),
        'total': total,
        'accuracy': accuracy.item()
    }
