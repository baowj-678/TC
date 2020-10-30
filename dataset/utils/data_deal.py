"""
@Description: 加载数据，保存数据
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/27
"""
import csv
import utils.my_csv

class DataDeal():
    def __init__(self):
        super().__init__()
    
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
        """ 保存数据到文件
        @param:\n
        :data: 数据
        :head: 数据头
        :path: 保存路径
        :cls: 分隔符
        :encoding: 编码方式
        """
        with open(path, mode='w+', encoding=encoding) as file:
            writer = csv.writer(file, dialect='tsv')
            if head is not None:
                writer.writerow(head)
            writer.writerows(data)
        return


if __name__ == "__main__":
    # data = DataDeal.load_data(path='dataset/CONLL2003/test.txt', cls=' ')
    # print(data[:9])
    data = [['wfereer', 'wce'], ['wcer', 'er']]
    head = ['we', 'e']
    DataDeal.save_data(data=data, head=head, path='a.out')
