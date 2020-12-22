#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : dynamic_deepsea_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/24 11:31
@version: 1.0
@desc  : 
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class DynamicDeepseaDataset(Dataset):
    """Dynamic Masked Language Model Dataset"""

    def __init__(self, directory, prefix, mask_prob=0.15):
        super().__init__()
        shape_dict = {
            "train": [(4400000, 1000), (4400000, 919), 4400000],
            "valid": [(8000, 1000), (8000, 919), 8000],
            "test": [(455024, 1000), (455024, 919), 455024]
        }
        self.mask_prob = mask_prob
        data_file = os.path.join(directory, prefix + "_data.dat")
        self.data = np.memmap(data_file, mode='r', shape=shape_dict[prefix][0])
        self.len = shape_dict[prefix][2]

        # 0,1,2,3 为 碱基对
        self.cls = 4
        self.sep = 5
        self.mask = 6

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        input_ids = torch.LongTensor([self.cls] + self.data[item].tolist() + [self.sep])
        masked_indices = self.char_mask(input_ids)
        labels = input_ids.clone()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(4, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels

    def char_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        random mask chars
        Args:
            input_ids: input ids [sent_len]
        Returns:
            masked_indices:[sent_len], if True, mask this token
        """
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        probability_matrix[0] = 0
        probability_matrix[-1] = 0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        return masked_indices


def unit_test():
    data_path = "/data/nfsdata2/sunzijun/media/deepsea/deepsea_train/trans_data"
    prefix = "valid"

    dataset = DynamicDeepseaDataset(data_path, prefix)
    print(len(dataset))
    from tqdm import tqdm
    for input_ids, labels in tqdm(dataset):
        output = [(input_id, label) for input_id, label in zip(input_ids.tolist()[:100], labels.tolist()[:100])]
        print(output)
        print()


if __name__ == '__main__':
    unit_test()
