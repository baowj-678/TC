"""
@Description: 加载数据，保存数据
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/27
"""
import csv
import sys
import os
import json

relative_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(relative_path)
import my_csv


class DataDeal:
    def __init__(self):
        pass
    
    @classmethod
    def load_data(cls, path, delimiter='\t', encoding="utf-8-sig", quotechar=None):
        """ 从文件读取数据\n
        @param:\n
        :path: 路径\n
        :cls: 分隔符（一行内）\n
        :encoding: 编码方式\n
        :quotechar: 引用符\n
        @return:\n
        :lines: list(),读取的数据\n
        """
        with open(path, "r", encoding=encoding) as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
        return lines
    
    @classmethod
    def save_data(cls, data, path, head=None, encoding='utf-8', dialect='tsv'):
        """ 保存多列数据到文件 [[str, ...], [str,...]]
        @param:\n
        :data: 数据
        :head: 数据头
        :path: 保存路径
        :cls: 分隔符
        :encoding: 编码方式
        """
        with open(path, mode='w+', encoding=encoding) as file:
            writer = csv.writer(file, dialect=dialect)
            if head is not None:
                writer.writerow(head)
            writer.writerows(data)
        return

    @classmethod
    def save_single(cls, data, path, encoding='utf-8'):
        """ 保存单列数据 [str]
        @param:
        :data: 数据
        :path: 保存路径
        :encoding: 编码方式
        """
        if not os.path.exists(os.path.dirname(path)):
            raise Exception("该路径不存在")
        with open(path, mode='w+', encoding=encoding) as file:
            for line in data:
                file.write(line)
                file.write('\n')
        print('-'*8, '写入成功', '-'*8)
        return

    @staticmethod
    def save_dict_json(path, dict_, encoding='utf-8'):
        """ 保存dict到json文件
        @param:
        :path: (str) 保存路径
        :dict_: (dict) 字典
        """
        # if not os.path.exists(os.path.dirname(path)):
        #     raise Exception("路径不存在")
        with open(path, 'w+', encoding=encoding) as json_file:
            json.dump(dict_, json_file, indent=2)

if __name__ == "__main__":
    # data = DataDeal.load_data(path='dataset/CONLL2003/test.txt', cls=' ')
    # print(data[:9])
    data = {2:3, 4:5}
    DataDeal.save_dict_json(dict_=data, path='a.out')