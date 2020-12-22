#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : deepsea_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/10/25 17:36
@version: 1.0
@desc  : 
"""

import os

import torch
from pandas import np
from torch.utils.data import Dataset, DataLoader
import random

class DeepseaDataset(Dataset):

    def __init__(self, directory, prefix, mode="conv"):
        super().__init__()
        self.mode = mode
        if mode == "conv":
            shape_dict = {
                "train": [(4400000, 4, 1000), (4400000, 919), 4400000],
                "valid": [(8000, 4, 1000), (8000, 919), 8000],
                "test": [(455024, 4, 1000), (455024, 919), 455024]
            }
        else:
            shape_dict = {
                "train": [(4400000, 1000), (4400000, 919), 4400000],
                "valid": [(8000, 1000), (8000, 919), 8000],
                "test": [(455024, 1000), (455024, 919), 455024]
            }
        data_file = os.path.join(directory, prefix + "_data.dat")
        label_file = os.path.join(directory, prefix + "_label.dat")
        self.data = np.memmap(data_file, mode='r', shape=shape_dict[prefix][0])
        self.label = np.memmap(label_file, mode='r', shape=shape_dict[prefix][1])

        self.len = shape_dict[prefix][2]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if self.mode == "conv":
            data = torch.FloatTensor(self.data[item])
            label = torch.FloatTensor(self.label[item])
        else:
            data = torch.LongTensor([4] + self.data[item].tolist())
            label = torch.FloatTensor(self.label[item])
        return data, label


def run():
    root_path = "/data/nfsdata2/sunzijun/media/deepsea/deepsea_train/trans_data"
    prefix = "test"
    dataset = DeepseaDataset(directory=root_path, prefix=prefix, mode="trans")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        num_workers=0,
        shuffle=False,
    )
    for data, label in dataloader:
        print(prefix, " batch shape: ", data.shape)
        print(prefix, " label shape: ", label.shape)
        print()


if __name__ == '__main__':
    run()
