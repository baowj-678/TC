"""
@Description: paper: Mutil-Timescale Long Short-Term Memory Neural Network
                     for Modeling Sentences and Documents
              link: https://www.aclweb.org/anthology/D15-1280/
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/27
"""

import torch
import torch.nn as NN
import torch.distributions as tdist
from torch.nn.utils.rnn import *


class MT_LSTM(NN.Module):
    def __init__(self, 
                 ave_length,
                 vocab,
                 embedding=None,
                 embedding_size=128,
                 hidden_size=128,
                 g=None,
                 class_num=2,
                 device=torch.device('cpu')):
        super(MT_LSTM, self).__init__()
        """
        @param:
        :g: 类别数
        :device: 训练设备
        :max_length: 句子最大长度
        """
        if g is None:
            g = torch.log2(ave_length) - 1
        if embedding is None:
            self.embedding = NN.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_size, padding_idx=0)
        else:
            self.embedding = embedding
        # 参数初始化(mu=0, sigma=1.0)
        Sampler = tdist.Normal(torch.tensor(0.0), torch.tensor(1.0))
        # Parameters
        # input gate
        self.W_i = NN.Parameter(Sampler.sample((1, hidden_size, embedding_size)), requires_grad=True)
        self.U_i = NN.Parameter(Sampler.sample((g, hidden_size, hidden_size)), requires_grad=True)
        self.V_i = NN.Parameter(Sampler.sample((g, hidden_size, hidden_size)), requires_grad=True)
        # forget gate
        self.W_f = NN.Parameter(Sampler.sample((1, hidden_size, embedding_size)), requires_grad=True)
        self.U_f = NN.Parameter(Sampler.sample((g, hidden_size, hidden_size)), requires_grad=True)
        self.V_f = NN.Parameter(Sampler.sample((g, hidden_size, hidden_size)), requires_grad=True)
        # output gate
        self.W_o = NN.Parameter(Sampler.sample((1, hidden_size, embedding_size)), requires_grad=True)
        self.U_o = NN.Parameter(Sampler.sample((g, hidden_size, hidden_size)), requires_grad=True)
        self.V_o = NN.Parameter(Sampler.sample((g, hidden_size, hidden_size)), requires_grad=True)
        # cell gate
        self.W_c = NN.Parameter(Sampler.sample((1, hidden_size, embedding_size)), requires_grad=True)
        self.U_c = NN.Parameter(Sampler.sample((g, hidden_size, hidden_size)), requires_grad=True)
        # Activate Func
        self.sigmoid = NN.Sigmoid()
        self.tanh = NN.Tanh()

    def forward(self, input, hidden_state=None):
        """ 前向传播
        @param:\n
        :input: padding的句子(batch_size, seq_len, input_size)/packed variable length sequence\n
        :hidden_state: (h_0, c_0),(batch_size, hidden_size, hidden_size)隐状态,如果为None则初始化为0
        @return:\n
        :output: (batch_size, seq_length, hidden_size)\n
        :h_n: (batch_size, hidden_size)\n
        :c_n: (batch_size, hidden_size)\n
        """
        # get batch_size
        batch_size = input.shape[0]
        if hidden_state is None:
            hidden_state = self.init_hidden_state(batch_size)
        h_0, c_0 = hidden_state
        # pad if packed
        if isinstance(input, PackedSequence):
            input, input_length = pad_packed_sequence(input)
        
        pass

    def cell(self, hidden_state, X: torch, U_index, V_index):
        """ 计算LSTM一个步骤\n
        @param:\n
        :hidden_state: (hidden, cell),上一cell的隐状态\n
        :X : 输入 (batch_size, input_size)\n
        :U_index : torch()\n
        :V_index : torch()\n
        @return:\n
        :(h_new, c_new): (batch_size, hidden_size)\n
        """
        h_t, c_t = hidden_state # (batch_size, hidden_size)
        h_t = h_t.unsqueeze(dim=1).unsqueeze(dim=3) # (batch_size, 1, hidden_size, 1)
        c_t = c_t.unsqueeze(dim=1).unsqueeze(dim=3)
        X = X.unsqueeze(dim=2) # (batch_size, input_size, 1)
        # input gate
        i_x = torch.matmul(self.W_i, X) # (batch_size, hidden_size, 1)
        U_i = self.U_i[U_index].unsqueeze(dim=0) # (1, k, hidden_size, hidden_size)
        V_i = self.V_i[V_index].unsqueeze(dim=0) # (1, k, hidden_size, hidden_size)
        i_h = torch.matmul(U_i, h_t) # (bacth_size, k, hidden_size, 1)
        i_h = torch.sum(i_h, dim=1) # (batch_size, hidden_size, 1)
        i_c = torch.matmul(V_i, c_t) # (batch_size, k, hidden_size, 1)
        i_c = torch.sum(i_c, dim=1) # (batch_size, hidden_size, 1)
        i = self.sigmoid(i_x, i_h, i_c).squeeze(dim=2) # (batch_size, hidden_size)
        # forget gate
        f_x = torch.matmul(self.W_f, X)
        U_f = self.U_f[U_index].unsqueeze(dim=0)
        V_f = self.V_f[V_index].unsqueeze(dim=0)
        f_h = torch.matmul(U_f, h_t)
        f_h = torch.sum(f_h, dim=1)
        f_c = torch.matmul(V_f, c_t)
        f_c = torch.sum(f_c, dim=1)
        f = self.sigmoid(f_x, f_h, f_c).squeeze(dim=2)
        # output gate
        o_x = torch.matmul(self.W_o, X)
        U_o = self.U_o[U_index].unsqueeze(dim=0)
        V_o = self.V_o[V_index].unsqueeze(dim=0)
        o_h = torch.matmul(U_o, h_t)
        o_h = torch.sum(o_h, dim=1)
        o_c = torch.matmul(V_o, c_t)
        o_c = torch.sum(o_c, dim=1)
        o = self.sigmoid(o_x, o_h, o_c).squeeze(dim=2)
        # next cell
        c_tilde_x = torch.matmul(self.W_c, X)
        U_c = self.U_c[U_index].unsqueeze(dim=0)
        c_tilde_h = torch.matmul(U_c, h_t)
        c_tilde_h = torch.sum(c_tilde_h, dim=1)
        c_tilde = self.tanh(c_tilde_x, c_tilde_h).squeeze(dim=2)
        c_new = torch.mul(f, c_t) + torch.mul(i, c_tilde)
        # next hidden
        h_new = torch.mul(o, self.tanh(c_new))
        return (h_new, c_new)

    def init_hidden_state(self, batch_size):
        """ 初始化开始的隐状态为0
        """
        return (torch.zeros(size=[batch_size, self.hidden_size, self.hidden_size], dtype=torch.float32),
                torch.zeros(size=[batch_size, self.hidden_size, self.hidden_size], dtype=torch.float32))
        
