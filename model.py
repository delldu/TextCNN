# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018年 09月 21日 星期五 10:25:44 CST
# ***
# ************************************************************************************/

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3)
             for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2)
             for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit


def train(train_iter, model, args):
    """
    Train Text CNN Model
    """
    def save_model(model, steps):
        if not os.path.isdir("logs"):
            os.makedirs("logs")
        save_path = 'logs/textcnn.model-{}'.format(steps)
        torch.save(model, save_path)

    def save_steps(epochs):
        n = int((epochs + 1) / 10)
        if n < 10:
            n = 10
        n = 10 * int((n + 9) / 10)  # round to 10x times
        return n

    print("Start training ...")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    if args.cuda:
        model.cuda()

    save_interval = save_steps(args.epochs)

    for epoch in range(1, args.epochs+1):
        training_loss = 0.0
        training_acc = 0.0
        training_count = 0.0

        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  # batch first, index align

            # print("-"*80)
            # print("feature: ", feature, feature.size())
            # print("target: ", target, target.size())

            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            # print("logit:", logit, logit.size())

            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

            training_loss += loss.item()
            training_acc += corrects.item()
            training_count += batch.batch_size

        training_loss /= training_count
        training_acc /= training_count
        accuracy = 100.0 * training_acc
        print('Training epoch [{}/{}] - loss: {:.6f}  acc: {:.2f}%'.format(
            epoch, args.epochs, training_loss, accuracy))

        if epoch % save_interval == 0:
            save_model(model, epoch)
    print("Training finished.")


def eval(data_iter, model, args):
    print("Start evaluating ...")
    model.eval()
    if args.cuda:
        model.cuda()

    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('Evaluation - loss: {:.6f}  acc: {:.4f}%'.format(avg_loss, accuracy))
    print("Evaluating finished.")
    return accuracy


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    if cuda_flag:
        model.cuda()

    text = text_field.preprocess(text)

    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.LongTensor(text)
    if cuda_flag:
        x = x.cuda()
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0] + 1]
