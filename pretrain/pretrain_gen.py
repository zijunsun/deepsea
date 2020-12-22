#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : pretrain_gen.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/24 11:17
@version: 1.0
@desc  : 
"""
import argparse
import json
import os

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.modeling_bert import BertForMaskedLM, BertConfig

from datasets.dynamic_deepsea_dataset import DynamicDeepseaDataset
from metric.classification import MaskedAccuracy
from utils.random_seed import set_random_seed

set_random_seed(0)


class RobertaPretrainGenModel(LightningModule):

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.bert_config = BertConfig.from_pretrained(self.args.bert_path)
        self.model = BertForMaskedLM(self.bert_config)
        self.loss_fn = CrossEntropyLoss(reduction="none")

        self.train_acc = MaskedAccuracy()
        self.valid_acc = MaskedAccuracy()

    def forward(self, inputs_ids):
        return self.model(inputs_ids)

    def get_dataloader(self, directory, prefix) -> DataLoader:
        """构造统一的dataloader方法"""
        dataset = DynamicDeepseaDataset(directory=directory, prefix=prefix)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            shuffle=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """训练过程的dataloader"""
        return self.get_dataloader(directory=self.args.data_path, prefix="valid")

    def training_step(self, batch, batch_idx):
        """训练过程"""
        loss, acc = self.compute_loss_and_acc(batch)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log('train_acc', acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),  # according to RoBERTa paper
                          lr=self.args.lr,
                          eps=self.args.adam_epsilon)
        t_total = len(self.train_dataloader()) // self.args.accumulate_grad_batches * self.args.max_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def compute_loss_and_acc(self, batch, mode='train'):
        """"""
        epsilon = 1e-10
        masked_lms = batch[1].view(-1)
        outputs = self(inputs_ids=batch[0])
        prediction_scores = outputs[0]
        label_mask = (masked_lms >= 0)
        predict_labels = torch.argmax(prediction_scores.view(-1, self.bert_config.vocab_size), dim=-1)
        if mode == 'train':
            acc = self.train_acc(preds=predict_labels, target=masked_lms, mask=label_mask.long())
        else:
            acc = self.valid_acc(preds=predict_labels, target=masked_lms, mask=label_mask.long())

        loss = self.loss_fn(prediction_scores.view(-1, self.bert_config.vocab_size), masked_lms)
        label_mask = label_mask.float()
        loss *= label_mask
        loss = loss.sum() / (label_mask.sum() + epsilon)
        return loss, acc

    def val_dataloader(self) -> DataLoader:
        """validation的dataloader"""
        return self.get_dataloader(directory=self.args.data_path, prefix="valid")

    def validation_step(self, batch, batch_idx):
        """对single batch进行validation"""
        loss, acc = self.compute_loss_and_acc(batch, mode='test')
        self.log('valid_acc', acc, on_step=False, on_epoch=True)
        self.log('valid_loss', loss)
        return loss


def add_model_specific_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--batch_size", type=int, default=2, help="batch size大小")
    parser.add_argument("--save_topk", type=int, default=5, help="save checkpoints个数")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=100, type=int, help="warmup steps")
    parser.add_argument("--lr_decay", type=float, default=0, help="learning rate decay per evaluation")
    parser.add_argument("--weight_decay", type=float, default=0, help="learning rate")
    parser.add_argument("--eps", type=float, default=1e-9, help="eps")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--data_path", required=True, type=str, help="该实验整体的根目录")
    parser.add_argument("--save_path", required=True, type=str, help="存checkpoint的路径")
    parser.add_argument("--bert_path", required=True, type=str, help="bert配置路径")

    return parser


def main():
    parser = add_model_specific_args()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    model = RobertaPretrainGenModel(args)
    # 如果save path不存在，则创建
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    # 保存args到文件
    with open(os.path.join(args.save_path, "args.json"), 'w') as f:
        args_dict = args.__dict__
        del args_dict['tpu_cores']
        json.dump(args_dict, f, indent=4)
    # 构造存储的路径
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_path, '{epoch}-{valid_loss:.4f}-{valid_acc:.4f}'),
        save_top_k=args.save_topk,
        monitor="train_loss",
        mode='min'
    )
    logger = TensorBoardLogger(
        save_dir=args.save_path,
        name="log"
    )
    trainer = Trainer.from_argparse_args(args,
                                         distributed_backend="ddp",
                                         checkpoint_callback=checkpoint_callback,
                                         logger=logger)

    trainer.fit(model)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    main()
