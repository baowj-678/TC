""" 数据集
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/2/3
"""
import pandas as pd
from torch.utils.data import Dataset
from dataset.vocab.vocab import Vocab


class MyDataSet(Dataset):
    def __init__(self, path, vocab, max_length):
        """ 初始化
        @param:
        :path: 文件路径
        :vocab: 字典
        :max_length: 句子长度
        """
        super(MyDataSet, self).__init__()
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
        self.data_X = []
        self.length_X = []
        for line in data_X:
            ids = self.vocab.build_inputs_with_special_tokens(data_X)
            self.data_X.append(ids)
            self.length_X.append(len(ids))
        return (self.data_X, self.length_X, self.data_Y)

