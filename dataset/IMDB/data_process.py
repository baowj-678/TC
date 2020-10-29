import os
import csv

def combine_file(src_dir, tgt_path):
    """ 将源目录下所有文件合并成一个文件\n
    :src_dir :源目录\n
    :tgt_path :生成文件的存放位置\n
    """
    data = []
    for root, dirs, files in os.walk(src_dir):
        """ 
        :root: 当前目录
        :dirs: 当前目录下所有子目录
        :files: 当前目录下所有文件
        """
        for f in files:
            path = os.path.join(root, f)
            with open(path, mode='r', encoding='utf-8') as file:
                data.append(file.read() + '\t0\n')
    with open(tgt_path, mode='w+', encoding='utf-8') as file:
        file.writelines(data)
    return

def combine_two_file(src_paths, tgt_path):
    """ 将src_path中的文件合并
    :src_paths: 源文件
    :tgt_path: 保存位置
    """
    data = []
    for path in src_paths:
        with open(path, mode='r', encoding='utf-8') as file:
            tmp = file.readlines()
            data += tmp
    with open(tgt_path, mode='w+', encoding='utf-8') as file:
        file.writelines(data)
    return

if __name__ == "__main__":
    src_dir = ['dataset/IMDB/aclImdb_v1/aclImdb/train/train_neg.txt',
               'dataset/IMDB/aclImdb_v1/aclImdb/train/train_pos.txt']
    tgt_path = 'dataset/IMDB/aclImdb_v1/aclImdb/train.csv'
    # combine_file(src_dir, tgt_path)
    combine_two_file(src_dir,tgt_path)