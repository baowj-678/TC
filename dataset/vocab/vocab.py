""" 字典的文件
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/30
"""
import os
import nltk


class Vocab:
    def __init__(self, vocab_path):
        """ 字典初始化
        @param:
        :vocab_path: 字典文件路径
        """
        super().__init__()
        self._stoi_ = {}
        self._itos_ = {}
        self._pad_s_ = '_pad_'
        self._pad_i_ = 0
        self._stoi_['_pad_'] = self._pad_i_
        self._itos_[self._pad_i_] = self._pad_s_
        print('-'*8, 'load vocab', '-'*8)
        self.load_file(path=vocab_path)
        print('-'*8, 'load successfully', '-'*8)
        self.max_l = 0

    def load_file(self, path):
        """ 加载字典文件 """
        if not os.path.exists(path):
            raise Exception("文件不存在")
        with open(path, mode='r') as file:
            data = file.readlines()
        begin = 1
        for word in data:
            word = word.strip()
            if word not in self._stoi_:
                self._stoi_[word] = begin
                self._itos_[begin] = word
                begin += 1
        return self._stoi_

    def sents2indexs(self, sents, max_length=None):
        """ 句子转index
        @param:
        :sents: (list) 句子s
        :max_length: (int) 最大句长
        @return:
        :indexs: (list) padding好的句子s
        :lengths: (list) 原句子s长度
        """
        indexs = []
        lengths = []
        for sent in sents:
            index, length = self.sent2index(sent, max_length)
            indexs.append(index)
            lengths.append(length)
        return (indexs, lengths)

    def sent2index(self, sent: str, max_length=None):
        """ 单个句子(str)转index(max_length)
        @param:
        :sent: (str)句子
        :max_length: (int) 最大句长
        @return:
        :sent: (list) padding好的indexs
        :length: int 原句长
        """
        sent = nltk.word_tokenize(sent)
        length = len(sent)
        if max_length is None:
            # 不padding
            sent = [self.__getitem__(word) for word in sent]
        else:
            # padding
            if len(sent) > max_length:
                sent = [self.__getitem__(word) for word in sent[:max_length]]
                length = max_length
            else:
                sent = [self.__getitem__(word) for word in sent] + [self._pad_i_] * (max_length - len(sent))
        return (sent, length)

    def __len__(self):
        return len(self._stoi_)

    def __getitem__(self, word):
        """ 根据word查询index """
        return self._stoi_.get(word, self._pad_i_)

    @property
    def pad_i(self):
        """ 返回pad index """
        return self._pad_i_

    @property
    def pad_s(self):
        """ 返回pad str"""
        return self._pad_s_

    def itos(self, index):
        """ 根据index查询word """
        return self._itos_.get(index, None)

if __name__ == '__main__':
    vocab = Vocab('D:/NLP/TC/dataset/SST-1/vocab.txt')
    print(len(vocab))
    print(vocab.sent2index('Yet the act is still charming here .'))
    print(vocab.pad_s)