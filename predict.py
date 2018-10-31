#! /usr/bin/env python
# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018年 09月 21日 星期五 10:25:44 CST
# ***
# ************************************************************************************/
#

import os
import sys
import argparse
import torch
import model
import data
from config import Config

parser = argparse.ArgumentParser(description='Text CNN classificer Predictor')
# model
parser.add_argument(
    '-model',
    type=str,
    default="model/textcnn.model",
    help='file name of the pre-trained model[model/textcnn.model]')
parser.add_argument(
    'predict', type=str, default=None, help='predict the sentence')

if __name__ == '__main__':
    conf = Config()
    args = parser.parse_args()

    text_field = data.FastTextTEXT
    label_field = data.FastTextLABEL

    text_field.vocab = data.load_vocab("model/text.vocab")
    label_field.vocab = data.load_vocab("model/label.vocab")

    # model
    if os.path.exists(args.model):
        print('Loading model from {}...'.format(args.model))
        cnn = torch.load(args.model)
    else:
        print("Model doesn't exist.")
        sys.exit(-1)

    if args.predict is not None:
        label = model.predict(args.predict, cnn, text_field, label_field, conf.cuda)
        print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
