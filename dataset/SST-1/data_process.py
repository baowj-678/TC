""" SST-1数据处理
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/29
"""
import sys
import os
sys.path.append('..')
from utils import *

def combine_data(seq_path, dict_path, label_path, save_path):
    """ 将句子和snetiment-value合并起来
        'sentence_index	sentence	sentiment values'
    :seq_path: delimiter  :label_path: label路径 
    """
    data = DataDeal.load_data(seq_path, delimiter='\t')
    label = DataDeal.load_data(label_path, delimiter='|')
    dicto = DataDeal.load_data(dict_path, delimiter='|')
    head = data[0] + [label[0][-1]]
    data = data[1:]
    label = label[1:]
    seq_phrase = {_[0] : int(_[1]) for _ in dicto}
    print('seq_size', len(data))
    print('label', len(label))
    print('seq-phrase', len(seq_phrase))
    del_lines = []
    for i, line in enumerate(data):
        index = seq_phrase.get(line[1], None)
        if index is None:
            del_lines.append(i)
            continue
        line.append(label[index][-1])
    for i in range(len(del_lines) - 1, -1, -1):
        del(data[del_lines[i]])
    print('nums', len(data))
    DataDeal.save_data(data=data, head=head, path=save_path)
    pass

def split_label(src_path, tgt_path):
    """ 将sentiment value划为label
        [0, 0.2]        (0.2, 0.4]  (0.4, 0.6]  (0.6, 0.8]  (0.8, 1.0]
        very negative   negative    neutral     positive    very positive
    @param:
    :src_path: 源文件
    :tgt_path: 保存路径
    """ 
    data = DataDeal.load_data(src_path, delimiter='\t')
    for line in data[1:]:
        value = float(line[-1])
        if value <= 0.2:
            value = 0
        elif value <= 0.4:
            value = 1
        elif value <= 0.6:
            value = 2
        elif value <= 0.8:
            value = 3
        elif value <= 1.0:
            value = 4
        line[-1] = value
    DataDeal.save_data(data=data, path=tgt_path)

def split_dataset(src_path, seq_split, train_path, test_path, dev_path):
    """ 将数据切分为train/test/dev
    @param:
    :src_path: 源文件目录
    :seq_split: 切分文件
    :train_path: 训练集保存路径
    :test_path: 测试集保存路径
    :dev_path: 验证集保存路径
    """
    data = DataDeal.load_data(src_path, delimiter='\t')
    split = DataDeal.load_data(seq_split, delimiter=',')
    head = data[0]
    data = data[1:]
    train = []
    test = []
    dev = []
    for line in data:
        index = int(split[int(line[0])][1])
        if index == 1:
            train.append(line)
        elif index == 2:
            test.append(line)
        elif index == 3:
            dev.append(line)
    print(train[:4])
    DataDeal.save_data(data=train,path=train_path, head=head)
    DataDeal.save_data(data=test,path=test_path, head=head)
    DataDeal.save_data(data=dev,path=dev_path, head=head)



if __name__ == "__main__":
    # seq_path = 'stanfordSentimentTreebank\datasetSentences.txt'
    # dict_path = 'stanfordSentimentTreebank\dictionary.txt'
    # label_path = 'stanfordSentimentTreebank\sentiment_labels.txt'
    # save_path = 'stanfordSentimentTreebank\dataSeqLabel.txt'
    # combine_data(seq_path, dict_path, label_path, save_path)
    # split_label('stanfordSentimentTreebank\dataSeqLabel.txt', 
    #              'stanfordSentimentTreebank\data.txt')
    split_dataset('stanfordSentimentTreebank\data.txt',
                  'stanfordSentimentTreebank\datasetSplit.txt',
                  'train.tsv',
                  'test.tsv',
                  'dev.tsv')