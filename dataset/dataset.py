""" 数据集
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/31
"""
import sys
import os
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
import pandas as pd
from torch.utils.data import Dataset
from vocab import Vocab
import torch


class DataSet(Dataset):
    def __init__(self, path, vocab, max_length):
        """ 初始化
        @param:
        :path: 文件路径
        :vocab: 字典
        :max_length: 句子长度
        """
        super(DataSet, self).__init__()
        self.vocab = vocab
        self.data_Y = None
        self.data_X = None
        self.length_X = None
        self.max_length = max_length
        self.load_data(path)

    def __getitem__(self, index):
        return (self.data_X[index],     # 句子
                self.length_X[index],   # 句子长度
                self.data_Y[index])     # label

    def __len__(self):
        return len(self.data_Y)

    def load_data(self, path, sep="\t", x_head='sentence', label_head='label'):
        """ 加载数据
        @param:
        :path: (str) 文件路径
        :vocab: (Vocab) 字典
        :sep: 分隔符
        @return:
        :data_X: (list) 处理（pad， to index）后的句子
        :length_X: (list) 句子原长度
        :data_Y: (list) labels
        """
        # 读取数据
        data = pd.read_csv(path, sep=sep)
        # 获取Y
        self.data_Y = data[label_head].values
        # 获取X
        data_X = data[x_head].values
        # word映射到int
        self.data_X, self.length_X = self.vocab.sents2indexs(data_X, self.max_length)
        return (self.data_X, self.length_X, self.data_Y)


if __name__ == '__main__':
    vocab = Vocab('D:/NLP/TC/dataset/SST-1/vocab.txt')
    dataset = DataSet('D:/NLP/TC/dataset/SST-1/test.tsv',
                      vocab,
                      56)
    print(dataset[9])