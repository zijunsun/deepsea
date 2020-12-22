#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : transformer.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/10/24 19:01
@version: 1.0
@desc  : 
"""
import torch
from torch import nn
from transformers.modeling_bert import BertModel, BertConfig


class TransformerClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bert_config = BertConfig.from_pretrained(self.args.bert_config_dir, output_hidden_states=False)
        self.bert = BertModel(self.bert_config)
        self.linear = nn.Linear(self.bert_config.hidden_size*1001, 919)
        self.threshold = nn.Threshold(0, 1e-6)
        self.linear2 = nn.Linear(919, 919)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        bs, _ = input_ids.shape
        output, first_token = self.bert(input_ids)
        # -----------------------------
        # method 1: 取first token
        # linear_out = self.linear(first_token)
        # -----------------------------
        # method 2: 取所有 token
        output = output.view(bs, -1)
        linear_out = self.linear(output)
        # -----------------------------
        linear_out = self.linear2(self.threshold(linear_out))
        out = self.sigmoid(linear_out)
        return out


def main():
    class Args:
        bert_config_dir = "/data/nfsdata2/sunzijun/media/trans_configs/transformer_config_1"

    model = TransformerClassifier(Args)
    print(model)
    input_ids = torch.randint(0, 5, (16, 1001))
    model(input_ids)


if __name__ == '__main__':
    main()
