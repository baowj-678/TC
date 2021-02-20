""" 根据原文件生成字典文件
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/30
"""
import nltk
import os
import sys

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)

from dataset.utils.data_deal import *


class MakeVocab():
    def __init__(self):
        pass

    @staticmethod
    def make_vocab_csv(path, tgt_path, seq_index, delimiter='\t'):
        """ 生成字典文件
        :path: 原文件
        :tgt_path: 目标文件存放位置
        :head: 句子所在的列
        :delimiter: 分割符
        """
        vocab = set()
        print('-'*8, 'tokenlize', '-'*8)
        for path_ in path:
            data = DataDeal.load_data(path=path_, delimiter=delimiter)
            for line in data:
                sent = line[seq_index]
                words = nltk.word_tokenize(sent)
                for word in words:
                    vocab.add(word)
        print("-"*8, "共{}词".format(len(vocab)), '-'*8)
        vocab = list(vocab)
        DataDeal.save_single(data=vocab, path=tgt_path)
        return

    @staticmethod
    def make_raw_file(path, tgt_path, seq_index, delimiter='\t'):
        """ 生成字典文件
        :path: 原文件
        :tgt_path: 目标文件存放位置
        :head: 句子所在的列
        :delimiter: 分割符
        """
        vocab = set()
        all_data = []
        for path_ in path:
            data = DataDeal.load_data(path=path_, delimiter=delimiter)
            data = [line[seq_index] for line in data]
            all_data += data
        DataDeal.save_single(data=all_data, path=tgt_path)
        return
    
if __name__ == '__main__':
    MakeVocab.make_raw_file(['dataset/SST-1/train.tsv', 'dataset/SST-1/test.tsv', 'dataset/SST-1/dev.tsv'],
                             tgt_path='dataset/SST-1/raw.raw',
                             seq_index=1,
                             delimiter='\t')
    # print(nltk.word_tokenize("The Rock is destined to be the 21st Century s newConan '' and that he s going to make a splash even greater than; it's Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal"))