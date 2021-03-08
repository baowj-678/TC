"""
Transformer 实现
Author: Bao Wenjie
Date: 2021/3/7
"""

import math
from numpy.core.numeric import outer
import torch
import torch.nn as NN
from torch.autograd import Variable
import torch.nn.functional as F

class Transformer(NN.Module):
    def __init__(self, d_model, max_len):
        super(Transformer, self).__init__()
        self.PosEnc = PositionalEncoding(d_model, max_len)
        pass

    def forward(self, X):
        pass


class PositionalEncoding(NN.Module):
    """ 位置编码 """
    def __init__(self, d_model, max_len=50000):
        """
        Param
        -----
        :d_model 词向量的维度
        :max_len 句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        assert d_model % 2 == 0
        pos_enc = torch.zeros(max_len, d_model)
        # (max_len, d_model)
        pos = torch.arange(0, max_len, 1).unsqueeze(1)
        # (max_len, 1)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2) / d_model)
        # (d_model/2)
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pos_enc', pos_enc)
        # (1, max_len, d_model)
    
    def forward(self, X:torch.tensor):
        """
        Param
        -----
        :X [batch_size, length, d_model]
        
        Return
        ------
        :X [batch_size, length, d_model]
        """
        X =  X + Variable(self.pos_enc[:, 0:X.shape[1]], requires_grad=False)
        return X

def attention(query, key, value, mask=None, dropout=None):
    """
    注意力机制
    Param
    -----
    :query (batch_size, seq_len, d_k)
    :key (batch_size, seq_len, d_k)
    :value (batch_size, seq_len, d_v)
    :mask (batch_size, seq_len, seq_len)
    :dropout function

    Return
    ------
    :output (batch_size, seq_len, d_v)
    :p_attn (batch_size, seq_len, d_k)
    """
    d_k = query.shape[-1]
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # (batch_size, seq_len, seq_len)
    if mask is not None:
        score = score.masked_fill(mask, -100)
    p_attn = F.softmax(score, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(NN.Module):
    """ 多头注意力 """
    def __init__(self, h, d_model, dropout=0.1) -> None:
        """
        Param
        -----
        :h 头的个数
        :d_model 模型维度
        :dropout 
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.W_O = NN.Linear(self.d_k * h, d_model)
        self.W_Q = [NN.Linear(d_model, self.d_k) for i in range(h)]
        self.W_K = [NN.Linear(d_model, self.d_k) for i in range(h)]
        self.W_V = [NN.Linear(d_model, self.d_k) for i in range(h)]
        self.dropout = NN.Dropout(dropout)
        self.attn = None
    
    def forward(self, query, key, value, mask=None):
        """
        Param
        -----
        :query (batch_size, seq_len, d_model)
        :key (batch_size, seq_len, d_model)
        :value (batch_size, seq_len, d_model)
        :mask (batch_size, seq_len, seq_len)
        
        Return
        ------
        :x (batch_size, seq_len, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(dim=0)
            # (1, batch_size, seq_len, seq_len)
        batch_size = query.shape[0]
        query = [network(query) for network in self.W_Q]
        key = [network(key) for network in self.W_K]
        value = [network(value) for network in self.W_V]
        # (h, batch_size, seq_len, d_k/d_v)
        query = torch.stack(query)
        key = torch.stack(key)
        value = torch.stack(value)
        # (h, batch_size, seq_len, d_k/d_v)
        x, self.attn = attention(query, key, value, mask)
        # (h, batch_size, seq_len, d_v)
        x = x.permute([1, 2, 0, 3])
        # (batch_size, seq_len, n, d_v)
        x = x.reshape(shape=(batch_size, -1, self.h * self.d_k))
        x = self.W_O(x)
        x = self.dropout(x)
        return x
