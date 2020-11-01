""" 数据集加载器
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/31
"""
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import torch

def collate_func(X):
    """ batch数据处理 (snntence, length, label)
    @param:
    :pre_X (tensor):从大到小排列的pre
    :pre_length (tensor): pre长度
    :hyp_X (tensor): 从大到小排列的hyp
    :hyp_length (tensor): hyp长度
    :Y (tensor):target和pre对应
    :pre_indices (tensor): hyp to pre
    """
    x = []
    x_length = []
    label = []
    for i in X:
        x.append(i[0])
        x_length.append(i[1])
        label.append([i[2]])
    x_length = torch.tensor(x_length)
    x = torch.tensor(x)
    label = torch.tensor(label)
    # Sort
    x_length, x_indices = torch.sort(x_length, descending=True)
    x = x[x_indices]
    label = label[x_indices]
    # Pack
    x = pack_padded_sequence(x, x_length, batch_first=True)
    return (x, label)


class DataLoader(DataLoader):
    def __init__(self, dataset, batch_size=16,shuffle=True):
        super(DataLoader, self).__init__(dataset=dataset, # 数据集
                                         batch_size=batch_size, # batch_size
                                         shuffle=shuffle, # 打乱
                                         sampler=None,
                                         batch_sampler=None,
                                         num_workers=0,
                                         collate_fn=collate_func,
                                         pin_memory=False,
                                         drop_last=False,
                                         timeout=0,
                                         worker_init_fn=None,
                                         multiprocessing_context=None)


if __name__ == '__main__':
    pass