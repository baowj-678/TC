""" 训练文件
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/31
"""
import pandas as pd
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
import torch
import sys
import os
abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)
from dataset.vocab import Vocab
from dataset.dataset import DataSet
from dataset.dataloader import DataLoader
from model import Model
from seed_all import seed_all



if __name__ == '__main__':
    # 设置随机种子
    seed_all(42)
    data_dir = '/home/baowenjie/TC/dataset/SST-1'
    # data_dir = 'D:/NLP/TC/dataset/SST-1'
    train_data_path = os.path.join(data_dir, 'train.tsv')
    dev_data_path = os.path.join(data_dir, 'dev.tsv')
    test_data_path = os.path.join(data_dir, 'test.tsv')
    vocab_path = os.path.join(data_dir, 'vocab.txt')
    save_path = 'output/MT_LSTM.pkl'

    BATCH_SIZE = 32
    max_length = 56
    embedding_size = 128
    hidden_size = 64
    lr = 0.003
    output_per_batchs = 10
    test_per_batchs = 60
    test_batchs = 10
    groups = 3
    p=0.1
    # 加载字典
    vocab = Vocab(vocab_path)
    # 创建数据集
    train_data_set = DataSet(train_data_path, vocab, max_length)
    test_data_set = DataSet(test_data_path, vocab, max_length)
    # 创建加载器
    train_data_loader = DataLoader(train_data_set, shuffle=True, batch_size=BATCH_SIZE)
    test_data_loader = DataLoader(test_data_set, shuffle=True, batch_size=BATCH_SIZE)
    # 是否用GPU
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    # 模型初始化
    model = Model(vocab=vocab,
                  hidden_size=hidden_size,
                  device=device,
                  ave_length=19,
                  embedding_size=embedding_size,
                  num_class=5,
                  g=groups,
                  p=0.1)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=1e-5)
    # 开始训练
    for i in range(100):
        print('='*8 + '开始训练' + '='*8)
        model.train()
        loss_sum = 0
        for epoch, data in enumerate(train_data_loader):
            X, Y = data
            optimizer.zero_grad()
            loss = model(X, Y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach()
            # 打印训练情况
            if((epoch + 1) % output_per_batchs == 0):
                print('itor: {}: epoch: {}/{}  loss: {}'.format(i + 1, epoch + 1, len(train_data_set), loss_sum / output_per_batchs))
                loss_sum = 0
            ############################### 测试 ######################################
            if (epoch + 1) % test_per_batchs == 0:
                print('-'*8 + '开始测试' + '-'*8)
                with torch.no_grad():
                    accuracy = 0
                    model.eval()
                    for epoch, data in enumerate(test_data_loader):
                        X, Y = data
                        Y = Y.to(device=device).squeeze(dim=1)
                        y = model(X).detach()
                        accuracy += torch.sum(y == Y).cpu()
                        if (epoch + 1) % test_batchs == 0:
                            break   
                    print('正确个数:{}, 总数:{}, 测试结果accu: {}'.format(accuracy, len(test_data_set), float(accuracy) / len(test_data_set)))
                    torch.save(model.state_dict(), save_path)
                model.train()
######################################## 最终测试 #############################
    print('-'*8 + '开始测试' + '-'*8)
    with torch.no_grad():
        accuracy = 0
        model.eval()
        for epoch, data in enumerate(test_data_loader):
            X, Y = data
            Y = Y.to(device=device).squeeze(dim=1)
            y = model(X).detach()
            accuracy += torch.sum(y == Y).cpu()
        print('正确个数:{}, 总数:{}, 测试结果accu: {}'.format(accuracy, len(test_data_set), float(accuracy) / len(test_data_set)))
        torch.save(model.state_dict(), save_path)

