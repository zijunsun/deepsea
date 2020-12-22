#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : conv_trainer.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/10/25 17:59
@version: 1.0
@desc  : 
"""

import argparse
import json
import os
import shutil

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy, auroc
from torch.nn import BCELoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from datasets.deepsea_dataset import DeepseaDataset
from task.convolution import ConvolutionClassifier


class ConvolutionClassificationModel(LightningModule):

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.model = ConvolutionClassifier()
        self.loss_fn = BCELoss()
        self.predict = [[] for i in range(919)]
        self.label = [[] for i in range(919)]

    def forward(self, inputs_ids):
        return self.model(inputs_ids)

    def get_dataloader(self, directory, prefix) -> DataLoader:
        """构造统一的dataloader方法"""
        dataset = DeepseaDataset(directory=directory, prefix=prefix)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            shuffle=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """训练过程的dataloader"""
        return self.get_dataloader(directory=self.args.root_path, prefix="valid")

    def training_step(self, batch, batch_idx):
        """训练过程"""
        input, label = batch
        y_hat = self.forward(input)
        loss = self.loss_fn(y_hat, label)
        acc = self.compute_metric(y_hat, label)
        auc = auroc(y_hat.view(-1), label.view(-1))
        tensorboard_logs = {'train_loss': loss, 'train_acc': acc, "train_auc": auc,
                            "lr": self.trainer.optimizers[0].param_groups[0]['lr']}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()],
            }
        ]
        optimizer = SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9,
                        weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=1 - self.args.lr_decay)
        return [optimizer], [scheduler]

    def compute_metric(self, y_hat, y):
        """
            计算准确率的函数
        Args:
            y_hat: 模型预测的y_hat
            y: 数据真实标签y
        """
        # 计算acc
        predict_labels = (y_hat > 0.5).int()
        acc = accuracy(predict_labels, y, 2)

        return acc

    def save_auc_data(self, y_hat, y):
        # 存储计算auc的数据
        y_hat_transpose = y_hat.transpose(1, 0)
        label_transpose = y.transpose(1, 0)
        y_hat_list = y_hat_transpose.tolist()
        label_list = label_transpose.tolist()

        for id, (y_hat_one, label_one) in enumerate(zip(y_hat_list, label_list)):
            self.predict[id].extend(y_hat_one)
            self.label[id].extend(label_one)

    def cal_auc(self):
        auc_list = []
        for id, (pred, label) in enumerate(zip(self.predict, self.label)):
            try:
                auc = auroc(torch.Tensor(pred), torch.Tensor(label))
            except:
                auc = torch.tensor(0)
            auc_list.append(auc)
        return auc_list

    def val_dataloader(self) -> DataLoader:
        """validation的dataloader"""
        return self.get_dataloader(directory=self.args.root_path, prefix="valid")

    def validation_step(self, batch, batch_idx):
        """对single batch进行validation"""
        input, label = batch
        y_hat = self.forward(input)
        loss = self.loss_fn(y_hat, label)
        acc = self.compute_metric(y_hat, label)
        self.save_auc_data(y_hat, label)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        """对all batch进行validation"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'].float() for x in outputs]).mean()

        auc_list = self.cal_auc()
        DNase_auc = torch.stack([auc for auc in auc_list[:125]]).sum() / 125
        TF_auc = torch.stack([auc for auc in auc_list[125:125 + 690]]).sum() / 690
        histone_auc = torch.stack([auc for auc in auc_list[125 + 690:]]).sum() / 104
        all_auc = torch.stack([auc for auc in auc_list]).sum() / 919

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc, "val_DNase_auc": DNase_auc,
                            "val_TF_auc": TF_auc, "val_histone_auc": histone_auc, "val_aucs": all_auc}
        # print("\n",tensorboard_logs)
        self.predict = [[] for i in range(919)]
        self.label = [[] for i in range(919)]
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_dataloader(self) -> DataLoader:
        """test的dataloader"""
        return self.get_dataloader(directory=self.args.root_path, prefix="valid")

    def test_step(self, batch, batch_idx):
        """对single batch 进行 test"""
        input, label = batch
        y_hat = self.forward(input)
        loss = self.loss_fn(y_hat, label)
        acc = self.compute_metric(y_hat, label)
        self.save_auc_data(y_hat, label)
        return {'test_loss': loss, 'test_acc': acc}

    def test_epoch_end(self, outputs):
        """对all batch进行test"""
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'].float() for x in outputs]).mean()

        auc_list = self.cal_auc()
        DNase_auc = torch.stack([auc for auc in auc_list[:125]]).sum() / 125
        TF_auc = torch.stack([auc for auc in auc_list[125:125 + 690]]).sum() / 690
        histone_auc = torch.stack([auc for auc in auc_list[125 + 690:]]).sum() / 104
        all_auc = torch.stack([auc for auc in auc_list]).sum() / 919

        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': avg_acc, "test_DNase_auc": DNase_auc,
                            "test_TF_auc": TF_auc, "test_histone_auc": histone_auc, "test_aucs": all_auc}

        name = "test_loss=" + str("%.4f" % avg_loss.item()) + "-acc=" + str("%.4f" % avg_acc.item()) + \
               "-DNase_auc=" + str("%.4f" % DNase_auc.item()) + "-TF_auc=" + str("%.4f" % TF_auc.item()) + \
               "-histone_auc=" + str("%.4f" % histone_auc.item()) + "-aucs=" + str("%.4f" % all_auc.item()) + '.txt'

        open(os.path.join(self.args.save_path, "checkpoints", name), 'a').close()
        return {'test_loss': avg_loss, 'log': tensorboard_logs}


def add_model_specific_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--batch_size", type=int, default=64, help="batch size大小")
    parser.add_argument("--save_topk", type=int, default=5, help="save checkpoints个数")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--lr_decay", type=float, default=8e-7, help="learning rate decay per evaluation")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="learning rate")
    # parser.add_argument("--epoch_size", type=int, default=8000, help="epoch size")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--patience", default=10, type=int, help="patience to early stop")

    parser.add_argument("--root_path", required=True, type=str, help="该实验整体的根目录")
    parser.add_argument("--save_path", required=True, type=str, help="存checkpoint的路径")

    return parser


def find_best_checkpoint(path: str):
    checkpoints = []
    for file in os.listdir(path):
        if file.__contains__('tmp') or file.__contains__('.txt'):
            continue
        acc = float(file.split('-')[3].split('=')[1].replace(".ckpt", ""))
        checkpoints.append((acc, file))
    orderd_checkpoints = sorted(checkpoints, key=lambda k: k[0], reverse=True)
    bert_checkpoint = os.path.join(path, orderd_checkpoints[0][1])

    return bert_checkpoint


def main():
    parser = add_model_specific_args()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    model = ConvolutionClassificationModel(args)

    # 如果save path不存在，则创建
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    # 保存args到文件
    with open(os.path.join(args.save_path, "args.json"), 'w') as f:
        args_dict = args.__dict__
        del args_dict['tpu_cores']
        json.dump(args_dict, f, indent=4)
    # 构造存储的路径
    checkpoint_path = os.path.join(args.save_path, "checkpoints")
    log_path = os.path.join(args.save_path, "log")
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, '{epoch}-{val_loss:.4f}-{val_acc:.4f}-{val_aucs:.4f}'),
        save_top_k=args.save_topk,
        monitor="val_aucs",
        mode='max',
        period=1
    )
    logger = TensorBoardLogger(
        save_dir=args.save_path,
        name="log"
    )
    early_stop = EarlyStopping(
        monitor='val_aucs',
        patience=args.patience,
        strict=False,
        verbose=False,
        mode='max'
    )
    trainer = Trainer.from_argparse_args(args,
                                         checkpoint_callback=checkpoint_callback,
                                         early_stop_callback=early_stop,
                                         logger=logger)

    trainer.fit(model)

    # 选取最优的val做测试
    best_checkpoint_path = find_best_checkpoint(checkpoint_path)
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    trainer.test(model)


def inference():
    parser = add_model_specific_args()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    model = ConvolutionClassificationModel(args)
    trainer = Trainer.from_argparse_args(args)
    # 选取最优的val做测试
    checkpoint_path = os.path.join(args.save_path, "checkpoints")
    best_checkpoint_path = find_best_checkpoint(checkpoint_path)
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    trainer.test(model)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    # main()
    inference()
