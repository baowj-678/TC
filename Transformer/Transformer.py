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
from torch.nn.modules.dropout import Dropout

class TransformerGenerate(NN.Module):
    """ Transformer """
    def __init__(self, d_model, d_ff, max_len, src_seq_len, tgt_seq_len, vocab_size, h=8, dropout=0.1, n_layers=6):
        super(TransformerGenerate, self).__init__()
        self.PosEnc = PositionalEncoding(d_model, max_len)
        self.encoder = Encoder(src_seq_len, d_model, h, dropout, d_ff, n_layers)
        self.decoder = Decoder(tgt_seq_len, d_model, h, dropout, d_ff, n_layers)
        self.linear = NN.Linear(d_model, vocab_size)
        self.softmax = NN.Softmax(dim=-1)

    def forward(self, X, src_mask, tgt_src_mask, tgt_mask, Y=None):
        """
        :X (batch_size, src_seq_len, d_model)
        :src_mask (batch_size, src_seq_len, src_seq_len)
        :tgt_src_mask (batch_size, tgt_seq_len, src_seq_len)
        :tgt_seq_mask (batch_size, src_seq_len, src_seq_len)
        :Y (batch_size, tgt_seq_len, d_model)
        """
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

class PositionwiseFeedForward(NN.Module):
    """ FeedForward"""
    def __init__(self, d_model, d_ff, dropout=0) -> None:
        """
        Param
        -----
        :d_model 模型(输入)维度
        :d_ff 内部参数维度
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = NN.Linear(d_model, d_ff)
        self.relu = NN.ReLU()
        self.w_2 = NN.Linear(d_ff, d_model)
        self.dropout = NN.Dropout(dropout)
    
    def forward(self, X):
        """
        Param
        -----
        :X (batch_size, seq_len, d_model)

        Return
        ------
        :X (batch_size, seq_len, d_model)
        """
        X = self.relu(self.w_1(X))
        X = self.dropout(X)
        X = self.w_2(X)
        return X

class Embedding(NN.Module):
    def __init__(self, vocab_size, d_model) -> None:
        """
        Param
        -----
        :vocab_size 词典大小(int)
        :d_model 模型维度(int)
        """
        super(Embedding, self).__init__()
        self.embeddings = NN.Embedding(vocab_size, d_model)
        self.sqrt_d_model = math.sqrt(d_model)
    
    def forward(self, X):
        """
        词向量编码
        Param
        -----
        :X [torch.tensor](batch_size, max_seq_len)

        Return
        ------
        :embed [torch.tensor](batch_size, max_seq_len, d_model)
        """
        embed = self.embeddings(X) * self.sqrt_d_model
        return embed

class AddNorm(NN.Module):
    """ 残差连接 """
    def __init__(self, seq_len, d_model, dropout=0.1) -> None:
        super(AddNorm, self).__init__()
        self.layernorm = NN.LayerNorm((seq_len, d_model))
        self.dropout = NN.Dropout(dropout)
    
    def forward(self, X, sub_X):
        sub_X = self.dropout(sub_X)
        X = X + sub_X
        X = self.layernorm(X)
        return X

class EncoderLayer(NN.Module):
    def __init__(self, seq_len, d_model, h, dropout, d_ff) -> None:
        """
        一个编码层
        Param
        -----
        :seq_len 句子长度
        :d_model 模型维度
        :h 头数
        :dropout
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(h, d_model, dropout)
        self.addnorm_1  = AddNorm(seq_len, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.addnorm_2  = AddNorm(seq_len, d_model, dropout)

    
    def forward(self, X, mask):
        """
        Param
        -----
        :X (batch_size, seq_len, d_model)
        :mask (batch_size, seq_len, seq_len)

        Return
        ------
        :X (batch_size, seq_len, d_model)
        """
        sub_X = self.self_attn(X, X, X, mask)
        X = self.addnorm_1(X, sub_X)
        sub_X = self.feed_forward(X)
        X = self.addnorm_2(X, sub_X)
        return X


class Encoder(NN.Module):
    def __init__(self, seq_len, d_model, h, dropout, d_ff, n_layer=6) -> None:
        """
        编码器
        Param
        -----
        :seq_len 句子长度
        :d_model 模型维度
        :h 头数
        :dropout
        :d_ff 前馈层的内部向量维度
        :n_layer 编码器包含的层数
        """
        super(Encoder, self).__init__()
        self.encoder_layers = [EncoderLayer(seq_len, d_model, h, dropout, d_ff) for i in range(n_layer)]
    
    def forward(self, X, mask):
        """
        Param
        -----
        :X (batch_size, seq_len, d_model)
        :mask (batch_size, seq_len, seq_len)

        Return
        ------
        :X (batch_size, seq_len, d_model)
        """
        for encoderlayer in self.encoder_layers:
            X = encoderlayer(X, mask)
        return X


class DecoderLayer(NN.Module):
    """ 解码器的子层 """
    def __init__(self, tgt_len, d_model, h, dropout, d_ff) -> None:
        """
        Param
        -----
        :seq_len 句子长度
        :d_model 模型维度
        :h 头数
        :dropout
        :d_ff
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(h, d_model, dropout)
        self.addnorm_1  = AddNorm(tgt_len, d_model, dropout)
        self.src_attn = MultiHeadAttention(h, d_model, dropout)
        self.addnorm_2  = AddNorm(tgt_len, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.addnorm_3  = AddNorm(tgt_len, d_model, dropout)


    def forward(self, X, M, src_mask, tgt_mask):
        """
        Param
        -----
        :X 训练时的目标语句 (batch_size, tgt_len, d_model)
        :M encoder得到的数据 (batch_size, src_len, d_model)
        :src_mask 对XM进行attention时的mask (batch_size, src_len, src_len)
        :tgt_mask 生成目标语句的mask (batch_size, tgt_len, src_len)

        Return
        ------
        :X (batch_size, tgt_len, d_model)
        """
        sub_X = self.self_attn(X, X, X, tgt_mask)
        X = self.addnorm_1(X, sub_X)
        sub_X = self.src_attn(X, M, M, src_mask)
        X = self.addnorm_2(X, sub_X)
        sub_X = self.feed_forward(X)
        X = self.addnorm_3(X, sub_X)
        return X


class Decoder(NN.Module):
    """ 解码器 """
    def __init__(self, tgt_len, d_model, h, dropout, d_ff, n_layers) -> None:
        super(Decoder, self).__init__()
        self.decoderlayers = [DecoderLayer(tgt_len, d_model, h, dropout, d_ff) for i in range(n_layers)]
    
    def forward(self, X, M, src_mask, tgt_mask):
        """
        Param
        -----
        :X 训练时的目标语句 (batch_size, tgt_len, d_model)
        :M encoder得到的数据 (batch_size, src_len, d_model)
        :src_mask 对XM进行attention时的mask (batch_size, src_len, src_len)
        :tgt_mask 生成目标语句的mask (batch_size, tgt_len, src_len)

        Return
        ------
        :X (batch_size, tgt_len, d_model)
        """
        for decoderlayer in self.decoderlayers:
            X = decoderlayer(X, M, src_mask, tgt_mask)
        return X


def en_seq_mask(seq_len, tgt_len, src_len):
    """
    encoder的句子mask
    Param
    -----
    :seq_len (list) (batch_size)
    :max_len (int)

    Return
    ------
    :mask (torch.ByteTensor) (batch_size, max_len, max_len)
    """
    batch_size = len(seq_len)
    mask = torch.ones(size=(batch_size, tgt_len, src_len), dtype=torch.bool)
    for mask_i, length in zip(mask, seq_len):
        mask_i[0:min(length, tgt_len), 0:min(length, src_len)] = torch.zeros((min(length, tgt_len), min(length, src_len)), dtype=torch.bool)
    return mask

def de_seq_mask(tgt_len):
    """
    生成上三角的mask

    Param
    -----
    :tgt_len 方阵边长 (int)

    Return
    ------
    :mask (1, tgt_len, tgt_len)
    """
    mask = torch.ones((1, tgt_len, tgt_len), dtype=torch.bool)
    mask = torch.triu(mask, diagonal=1)
    return mask

# tgt_len = 25
# src_len = 23
# d_model = 512
# h = 8
# dropout = 0.1
# d_ff = 2048
# batch_size = 13
# seq_lens = [22, 21, 21, 20, 18, 15, 17, 19, 13, 12, 11, 11, 10]
# M = torch.randn((batch_size, src_len, d_model))
# X = torch.randn((batch_size, tgt_len, d_model))
# src_mask = en_seq_mask(seq_lens, tgt_len, src_len)
# tgt_mask = de_seq_mask(tgt_len)
# model = Decoder(tgt_len, d_model, h, dropout, d_ff, 6)
# model(X, M, src_mask, tgt_mask)
