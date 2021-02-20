""" 数据集分析
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/31
"""

import matplotlib.pyplot as plt
import nltk
from utils.data_deal import DataDeal



class Analyse:
    def __init__(self, path):
        """ 初始化
        @param:
        :path: (str) 文件路径
        """

    @staticmethod
    def analyse(path,
                delimiter='\t',
                sent_index=1,
                label_index=-1,
                has_head=True,
                length_analyse=True,
                word_analyse=True,
                label_analyse=True,
                save_path=None):
        """ 生成字典文件
        :path: (list/str) 原文件
        :tgt_path: 目标文件存放位置
        :head: 句子所在的列
        :delimiter: 分割符
        """
        vocab = dict()
        label = dict()
        max_length = 0
        sum_length = 0
        sum_lines = 0
        lengths = dict()
        print('-' * 8, 'analyse', '-' * 8)
        if isinstance(path, list):
            for _ in path:
                data = DataDeal.load_data(path=_, delimiter=delimiter)
                if has_head:
                    data = data[1:]
                for line in data:
                    if word_analyse or length_analyse:
                        sent = line[sent_index]
                        words = nltk.word_tokenize(sent)
                        max_length = max(len(words), max_length)
                        sum_length += len(words)
                        sum_lines += 1
                        if len(words) in lengths:
                            lengths[len(words)] += 1
                        else:
                            lengths[len(words)] = 1
                        if word_analyse:
                            for word in words:
                                if word in vocab:
                                    vocab[word] += 1
                                else:
                                    vocab[word] = 1
                    if label_analyse:
                        label_ = line[label_index]
                        if label_ in label:
                            label[label_] += 1
                        else:
                            label[label_] = 1
        else:
            data = DataDeal.load_data(path=path, delimiter=delimiter)
            if has_head:
                data = data[1:]
            for line in data:
                if word_analyse or length_analyse:
                    sent = line[sent_index]
                    words = nltk.word_tokenize(sent)
                    max_length = max(len(words), max_length)
                    sum_length += len(words)
                    sum_lines += 1
                    if len(words) in lengths:
                        lengths[len(words)] += 1
                    else:
                        lengths[len(words)] = 1
                    if word_analyse:
                        for word in words:
                            if word in vocab:
                                vocab[word] += 1
                            else:
                                vocab[word] = 1
                if label_analyse:
                    label_ = line[label_index]
                    if label_ in label:
                        label[label_] += 1
                    else:
                        label[label_] = 1
        print("-" * 8, "result", '-' * 8)
        font = {'family': 'SimHei',
                'style': 'italic',
                'weight': 'normal',
                'color': 'black',
                'size': 20
                }
        if length_analyse:
            print('max sentence length:{}'.format(max_length))
            print('average sentence length:{}'.format(sum_length/sum_lines))
            plt.scatter(list(lengths.keys()), list(lengths.values()))
            plt.title('sentence length distribution', font)
            plt.ylabel('length', font)
            plt.xlabel('nums', font)
            plt.grid()
            plt.show()
        if word_analyse:
            word_time = dict()
            for time in vocab.values():
                if time in word_time:
                    word_time[time] += 1
                else:
                    word_time[time] = 1
            plt.scatter(list(word_time.keys()), list(word_time.values()))
            plt.title('word show times distribution', font)
            plt.ylabel('nums', font)
            plt.xlabel('show-times', font)
            plt.grid()
            plt.show()
            DataDeal.save_dict_json(path=save_path, dict_=vocab)
        return

    @staticmethod
    def word_analyse():
        pass


if __name__ == '__main__':
    Analyse.analyse(path=['../SST-1/test.tsv', '../SST-1/train.tsv', '../SST-1/dev.tsv'],
                    save_path='../SST-1/vocab_analyse.json')