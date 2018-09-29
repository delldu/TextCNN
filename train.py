#! /usr/bin/env python
# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018年 09月 21日 星期五 10:25:44 CST
# ***
# ************************************************************************************/


import os
import argparse

import torch
import model
import data
from config import Config

parser = argparse.ArgumentParser(description='Train Text CNN classificer')
parser.add_argument(
    '-model',
    type=str,
    default="logs/model.pth",
    help='file name of pre-trained model [logs/model.pth]')


if __name__ == '__main__':
    conf = Config()
    conf.dump()
    args = parser.parse_args()

    if not os.path.isdir("logs"):
        os.mkdir("logs")

    print("Loading data...")
    train_iter, text_field, label_field = data.fasttext_dataloader(
        "data/train.txt", conf.batch_size)
    data.save_vocab(text_field.vocab, "logs/text.vocab")
    data.save_vocab(label_field.vocab, "logs/label.vocab")

    # Update configurations
    conf.embed_num = len(text_field.vocab)
    conf.class_num = len(label_field.vocab) - 1
    conf.kernel_sizes = [int(k) for k in conf.kernel_sizes.split(',')]

    # model
    if os.path.exists(args.model):
        print('Loading model from {}...'.format(args.model))
        cnn = torch.load(args.model)
    else:
        cnn = model.TextCNN(conf)

    print(cnn)
    try:
        model.train(train_iter, cnn, conf)
    except KeyboardInterrupt:
        print('-' * 80)
        print('Exiting from training early')
