""" 根据原文件生成字典文件
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/30
"""
import nltk
from utils.data_deal import *
class MakeVocab():
    def __init__(self):
        pass

    @staticmethod
    def make_vocab_csv(path, tgt_path, seq_index, delimiter='\t'):
        """ 生成字典文件
        :path: 原文件(列表，多个)
        :tgt_path: 目标文件存放位置
        :seq_index: 句子所在的列
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
                    word = word.lower()
                    vocab.add(word)
        print("-"*8, "共{}词".format(len(vocab)), '-'*8)
        vocab = list(vocab)
        DataDeal.save_single(data=vocab, path=tgt_path)
        return

if __name__ == '__main__':
    MakeVocab.make_vocab_csv(['IMDB/test.csv', 'IMDB/test.csv'],
                             tgt_path='IMDB/vocab_.txt',
                             seq_index=0,
                             delimiter='\t')
    # print(nltk.word_tokenize("The Rock is destined to be the 21st Century s newConan '' and that he s going to make a splash even greater than; it's Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal"))