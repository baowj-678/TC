""" 自定义CSV格式
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/29
"""
import csv
from csv import excel

class tsv(excel):
    """ \t分隔符 """
    delimiter = '\t'
    lineterminator = '\n'

csv.register_dialect('tsv', tsv)