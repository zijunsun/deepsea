#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : convolution.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/10/24 19:01
@version: 1.0
@desc  : 
"""
import torch
from torch import nn


class ConvolutionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层卷积
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8)
        self.threshold1 = nn.Threshold(0, 1e-6)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout1 = nn.Dropout(0.2)
        # 第二层卷积
        self.conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8)
        self.threshold2 = nn.Threshold(0, 1e-6)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout2 = nn.Dropout(0.2)
        # 第三层卷积
        self.conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8)
        self.threshold3 = nn.Threshold(0, 1e-6)
        self.dropout3 = nn.Dropout(0.5)

        # 展开后过linear
        self.linear1 = nn.Linear(960 * 53, 919)
        self.threshold4 = nn.Threshold(0, 1e-6)
        self.linear2 = nn.Linear(919, 919)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        bs, _, _ = input_ids.shape
        # 第一层卷积
        conv1 = self.dropout1(self.pool1(self.threshold1(self.conv1(input_ids))))
        # 第二层卷积
        conv2 = self.dropout2(self.pool2(self.threshold2(self.conv2(conv1))))
        # 第三层卷积
        conv3 = self.dropout3(self.threshold3(self.conv3(conv2)))
        # reshape
        conv_out = conv3.reshape(bs, -1)
        # 过全连接层
        liner_out = self.linear2(self.threshold4(self.linear1(conv_out)))
        # 过sigmoid层
        out = self.sigmoid(liner_out)

        return out


def main():
    model = ConvolutionClassifier()
    # print(model)
    input_ids = torch.rand(16, 4, 1000)
    model(input_ids)


if __name__ == '__main__':
    main()
