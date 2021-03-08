"""
@Description: paper: Mutil-Timescale Long Short-Term Memory Neural Network
                     for Modeling Sentences and Documents
              link: https://www.aclweb.org/anthology/D15-1280/
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Github: https://github.com/baowj-678
@Date: 2020/10/27
"""

import torch
import torch.nn as NN
import torch.distributions as tdist
from torch.nn import init
from torch.nn.utils.rnn import *
import numpy as np


class MT_LSTM(NN.Module):
    def __init__(self, 
                 ave_length,
                 input_size=128,
                 hidden_size=128,
                 g=None,
                 batch_first=True,
                 device=torch.device('cpu'),
                 p=0.1):
        super(MT_LSTM, self).__init__()
        """
        @param:
        :g: 类别数
        :device: 训练设备
        :max_length: 句子最大长度
        """
        if g is None:
            self.g = int(np.math.log2(ave_length) - 1)
        else:
            self.g = g
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.device = device
        # Parameters
        # input gate
        self.W_i = NN.Parameter(torch.zeros((self.g, hidden_size, input_size), requires_grad=True)).to(device)
        self.U_i = NN.Parameter(torch.zeros((self.g, self.g, hidden_size, hidden_size), requires_grad=True)).to(device)
        self.V_i = NN.Parameter(torch.zeros((self.g, self.g, hidden_size, hidden_size), requires_grad=True)).to(device)
        NN.init.uniform_(self.W_i, 0, 0.1)
        NN.init.uniform_(self.U_i, 0, 0.1)
        NN.init.uniform_(self.V_i, 0, 0.1)
        # forget gate
        self.W_f = NN.Parameter(torch.zeros((self.g, hidden_size, input_size), requires_grad=True)).to(device)
        self.U_f = NN.Parameter(torch.zeros((self.g, self.g, hidden_size, hidden_size), requires_grad=True)).to(device)
        self.V_f = NN.Parameter(torch.zeros((self.g, self.g, hidden_size, hidden_size), requires_grad=True)).to(device)
        NN.init.uniform_(self.W_f, 0, 0.1)
        NN.init.uniform_(self.U_f, 0, 0.1)
        NN.init.uniform_(self.V_f, 0, 0.1)
        # output gate
        self.W_o = NN.Parameter(torch.zeros((self.g, hidden_size, input_size), requires_grad=True)).to(device)
        self.U_o = NN.Parameter(torch.zeros((self.g, self.g, hidden_size, hidden_size), requires_grad=True)).to(device)
        self.V_o = NN.Parameter(torch.zeros((self.g, self.g, hidden_size, hidden_size), requires_grad=True)).to(device)
        NN.init.uniform_(self.W_o, 0, 0.1)
        NN.init.uniform_(self.U_o, 0, 0.1)
        NN.init.uniform_(self.V_o, 0, 0.1)
        # cell gate
        self.W_c = NN.Parameter(torch.zeros((self.g, hidden_size, input_size), requires_grad=True)).to(device)
        self.U_c = NN.Parameter(torch.zeros((self.g, self.g, hidden_size, hidden_size), requires_grad=True)).to(device)
        NN.init.uniform_(self.W_c, 0, 0.1)
        NN.init.uniform_(self.U_c, 0, 0.1)
        # Dropout
        self.dropout = NN.Dropout(p=p)
        # Activate Func
        self.sigmoid = NN.Sigmoid()
        self.tanh = NN.Tanh()

    def forward(self, input, hidden_state=None):
        """ 前向传播
        @param:\n
        :input: padding的句子(batch_size, seq_len, input_size)/packed variable length sequence\n
        :hidden_state: (h_0, c_0),(batch_size, group, hidden_size)隐状态,如果为None则初始化为0
        @return:\n
        :output: (batch_size, seq_length, hidden_size)\n
        :hidden_state: (batch_size, group, hidden_size)\n
        """
        # pad if packed
        is_packed = False
        input_length = None
        if isinstance(input, PackedSequence):
            input, input_length = pad_packed_sequence(input, batch_first=self.batch_first)
            is_packed = True
        # get sent length
        if input_length is None:
            input_length = torch.ones(size=(len(input)))*len(input[0])
        # batch_first
        if self.batch_first:
            batch_size = input.shape[0]
            input = input.permute([1, 0, 2]) #(seq_lenght, batch_size, hidden_size)
        else:
            batch_size = input.shape[1]
        # init hidden_state
        hidden_state_t = None
        if hidden_state is None:
            hidden_state_t = self.init_hidden_state(batch_size)
        else:
            hidden_state_t = hidden_state
        # get senetence max_length
        seq_length = input.shape[0]
        # MT-LSTM
        output = []
        h_t = None
        c_t = None
        hidden_states = []
        for t in range(1, seq_length + 1, 1):
            input_t = input[t - 1] 
            # (batch_size, hidden_size)
            UV_index = self.get_UV(t=t) 
            # 获得参与运算的Groups
            h_t, c_t = self.cell(hidden_state_t, input_t, UV_index=UV_index) 
            # (batch_size, k, hidden_size)
            output.append(h_t)
            hidden_state_t = (h_t, c_t)
            hidden_states.append(hidden_state_t)
        output = torch.stack(output, dim=0)
        h_ns = []
        c_ns = []
        for batch, length in enumerate(input_length):
            h_ns.append(hidden_states[length - 1][0][batch, ::])
            c_ns.append(hidden_states[length - 1][1][batch, ::])
        h_ns = torch.stack(h_ns, dim=0) 
        # (batch_size, group, hidden_size)
        c_ns = torch.stack(c_ns, dim=0)
        hidden_state_n  = (h_ns, c_ns)
        if self.batch_first:
            output = output.permute([1, 0, 2, 3])
            output = output.flatten(start_dim=2)
            # (batch_size, seq_length, k*hidden_size) 
        # output_packed if input packed
        if is_packed:
            output = pack_padded_sequence(output, input_length, batch_first=self.batch_first)
        return output, hidden_state_n

    def cell(self, hidden_state, X: torch, UV_index):
        """ 计算MT-LSTM一个步骤\n
        @param:\n
        :hidden_state: (batch_size, group, hidden_size),上一cell的隐状态\n
        :X: 输入 (batch_size, input_size)\n
        :UV_index: (n)\n
        @return:\n
        :(h_new, c_new): (batch_size, group, hidden_size)\n
        """
        h_t, c_t = hidden_state
        h_new = h_t.clone()
        c_new = c_t.clone()
        # (batch_size, group, hidden_size)
        X = X.unsqueeze(dim=2)
        # (batch_size, input_size, 1)
        for length in range(len(UV_index), 0, -1):
            sub_index = UV_index[:length]
            g_ = sub_index[-1]
            """ input gate """
            i_x = torch.matmul(self.W_i[g_], X) 
            # (batch_size, hidden_size, 1)
            U_i = self.U_i[sub_index, g_, ::].unsqueeze(dim=0)
            # (1, k, hidden_size, hidden_size)
            V_i = self.V_i[sub_index, g_, ::].unsqueeze(dim=0)
            h_k = h_t[:, g_, :].unsqueeze(dim=2).unsqueeze(dim=1) # (batch_size, 1, hidden_size, 1)
            i_h = torch.matmul(U_i, h_k)
            # (bacth_size, k, hidden_size, 1)
            i_h = torch.sum(i_h, dim=1) # (batch_size, hidden_size, 1)
            c_k = c_t[:, g_, :].unsqueeze(dim=2).unsqueeze(dim=1) # (batch_size, 1, hidden_size, 1)
            i_c = torch.matmul(V_i, c_k) # (batch_size, k, hidden_size, 1)
            i_c = torch.sum(i_c, dim=1) # (batch_size, hidden_size, 1)
            i = self.sigmoid(i_x + i_h + i_c).squeeze(dim=2) # (batch_size, hidden_size)
            """ forget gate """
            f_x = torch.matmul(self.W_f[g_], X)     # (batch_size, hidden_size, 1)
            U_f = self.U_f[sub_index, g_, ::].unsqueeze(dim=0) # (1, k, hidden_size, hidden_size)
            V_f = self.V_f[sub_index, g_, ::].unsqueeze(dim=0)
            f_h = torch.matmul(U_f, h_k)
            f_h = torch.sum(f_h, dim=1) # (batch_size, hidden_size, 1)
            f_c = torch.matmul(V_f, c_k)
            f_c = torch.sum(f_c, dim=1)
            f = self.sigmoid(f_x + f_h + f_c).squeeze(dim=2) # (batch_size, hidden_size)
            """ output gate """
            o_x = torch.matmul(self.W_o[g_], X) # (batch_size, hidden_size, 1)
            U_o = self.U_o[sub_index, g_, ::].unsqueeze(dim=0) # (1, k, hidden_size, hidden_size)
            V_o = self.V_o[sub_index, g_, ::].unsqueeze(dim=0)
            o_h = torch.matmul(U_o, h_k)
            o_h = torch.sum(o_h, dim=1) # (batch_size, hidden_size, 1)
            o_c = torch.matmul(V_o, c_k)
            o_c = torch.sum(o_c, dim=1)
            o = self.sigmoid(o_x + o_h + o_c).squeeze(dim=2) # (batch_size, hidden_size)
            """ next cell """
            c_tilde_x = torch.matmul(self.W_c[g_], X) # (batch_size, hidden_size, 1)
            U_c = self.U_c[sub_index, g_, ::].unsqueeze(dim=0) # (1, k, hidden_size, hidden_size)
            c_tilde_h = torch.matmul(U_c, h_k)
            c_tilde_h = torch.sum(c_tilde_h, dim=1) # (batch_size, hidden_size, 1)
            c_tilde = self.tanh(c_tilde_x + c_tilde_h).squeeze(dim=2)
            # (batch_size, hidden_size)
            c_new[:, g_, :] = torch.mul(f, c_t[:, g_, :]) + torch.mul(i, c_tilde)
            c_new = self.dropout(c_new)
            """ next hidden """
            h_new[:, g_, :] = torch.mul(o, self.tanh(c_new[:, g_, :]))
        return (h_new, c_new)

    def init_hidden_state(self, batch_size):
        """ 初始化开始的隐状态为0
        """
        return (torch.zeros(size=[batch_size, self.g, self.hidden_size], dtype=torch.float32).to(self.device),
                torch.zeros(size=[batch_size, self.g, self.hidden_size], dtype=torch.float32).to(self.device))

    def get_UV(self, t):
        """ 根据t，返回这一步骤参与的group,
        1 -> [0]
        2 -> [0, 1]
        3 -> [0]
        4 -> [0, 1, 2]
        5 -> [0]
        6 -> [0, 1]
        7 -> [0]
        8 -> [0, 1, 2, 3]
        :t: int时间步
        """
        max_index = np.math.log2(t) + 1
        if max_index == int(max_index):
            indexs = torch.arange(0, min(self.g, max_index), 1, dtype=torch.long)
        else:
            indexs = torch.arange(0, 1, 1, dtype=torch.long)
        return indexs


if __name__ == "__main__":
    batch_size = 8
    max_length = 30
    hidden_size = 128
    lstm = MT_LSTM(ave_length=24,
                   hidden_size=hidden_size)
    x = torch.randn(size=(batch_size, max_length, hidden_size))
    length = [30, 29,29, 28,27, 26, 25, 25]
    x = pack_padded_sequence(x, torch.tensor(length), batch_first=True)
    output, hidden_state = lstm(x)
    h_n, c_n = hidden_state
    output, length = pad_packed_sequence(output, batch_first=True)
    print(output.shape, length)
    print(output[-1][24])
    print(h_n[-1])