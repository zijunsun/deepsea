# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: accuracy
@time: 2020/7/11 11:51

"""

import torch
from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.metrics.functional import auroc

class MaskedAccuracy(Metric):
    """
    Computes the accuracy classification score
    """

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        correct = (preds == target).float() * mask.float()
        self.correct += correct.sum()
        self.total += (mask.float().sum() + 1e-10)

    def compute(self):
        return self.correct.float() / self.total


def main():
    pred = torch.tensor([0, 0, 0, 0])
    target = torch.tensor([0, 0, 0, 0])
    mask = torch.tensor([1, 1, 1, 1])
    metric = MaskedAccuracy()
    print(metric(pred, target, mask))


if __name__ == '__main__':
    main()
