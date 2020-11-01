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
import numpy as np


class MT_LSTM(NN.Module):
    def __init__(self, 
                 ave_length,
                 input_size=128,
                 hidden_size=128,
                 g=None,
                 batch_first=True,
                 device=torch.device('cpu')):
        super(MT_LSTM, self).__init__()
        """
        @param:
        :g: 类别数
        :device: 训练设备
        :max_length: 句子最大长度
        """
        if g is None:
            self.g = int(np.math.log2(ave_length) - 1)
            
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.device = device
        # 参数初始化(mu=0, sigma=1.0)
        Sampler = tdist.Normal(torch.tensor(0.0), torch.tensor(1.0))
        # Parameters
        # input gate
        self.W_i = NN.Parameter(Sampler.sample((1, hidden_size, input_size)), requires_grad=True).to(device)
        self.U_i = NN.Parameter(Sampler.sample((self.g, hidden_size, hidden_size)), requires_grad=True).to(device)
        self.V_i = NN.Parameter(Sampler.sample((self.g, hidden_size, hidden_size)), requires_grad=True).to(device)
        # forget gate
        self.W_f = NN.Parameter(Sampler.sample((1, hidden_size, input_size)), requires_grad=True).to(device)
        self.U_f = NN.Parameter(Sampler.sample((self.g, hidden_size, hidden_size)), requires_grad=True).to(device)
        self.V_f = NN.Parameter(Sampler.sample((self.g, hidden_size, hidden_size)), requires_grad=True).to(device)
        # output gate
        self.W_o = NN.Parameter(Sampler.sample((1, hidden_size, input_size)), requires_grad=True).to(device)
        self.U_o = NN.Parameter(Sampler.sample((self.g, hidden_size, hidden_size)), requires_grad=True).to(device)
        self.V_o = NN.Parameter(Sampler.sample((self.g, hidden_size, hidden_size)), requires_grad=True).to(device)
        # cell gate
        self.W_c = NN.Parameter(Sampler.sample((1, hidden_size, input_size)), requires_grad=True).to(device)
        self.U_c = NN.Parameter(Sampler.sample((self.g, hidden_size, hidden_size)), requires_grad=True).to(device)
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
        # pad if packed
        is_packed = False
        input_length = None
        if isinstance(input, PackedSequence):
            input, input_length = pad_packed_sequence(input, batch_first=self.batch_first)
            is_packed = True
        # get sent length
        if input_length is not None:
            input_length_ = input_length.numpy().tolist()
        else:
            input_length_ = None
        # batch_first
        if self.batch_first:
            batch_size = input.shape[0]
            input = input.permute([1, 0, 2]) #(seq_lenght, batch_size, hidden_size)
        else:
            batch_size = input.shape[1]
        # init hidden_state
        if hidden_state is None:
            hidden_state = self.init_hidden_state(batch_size)
        h_0, c_0 = hidden_state
        # get senetence max_length
        seq_length = input.shape[0]
        # MT-LSTM
        output = torch.zeros(size=(seq_length, batch_size, self.hidden_size), requires_grad=False).to(self.device)
        h_n = torch.zeros(size=(batch_size, self.hidden_size), requires_grad=False).to(self.device)
        c_n= torch.zeros(size=(batch_size, self.hidden_size), requires_grad=False).to(self.device)
        h_t = None
        c_t = None
        for t in range(1, seq_length, 1):
            input_t = input[t - 1] # (batch_size, hidden_size)
            if input_length_ is not None:
                input_t = input_t[torch.arange(0, len(input_length_), 1)] # (k, hidden_size)
            UV_index = self.get_UV(t=t) # 参与运算的groupi id
            h_t, c_t = self.cell(hidden_state, input_t, UV_index=UV_index) # (k, hidden_size)
            if input_length_ is not None:
                output[t - 1, torch.arange(0, len(input_length_), 1), :] = h_t
            else:
                output[t - 1] = h_t
            index = -1
            if input_length_ is not None:
                while t == input_length_[-1]:
                    h_n[len(input_length_) - 1] = h_t[index]
                    c_n[len(input_length_) - 1] = c_t[index]
                    index -= 1
                    input_length_.pop()
                h_t = h_t[torch.arange(0, len(input_length_), 1)]
                c_t = c_t[torch.arange(0, len(input_length_), 1)]
            hidden_state = (h_t, c_t)
        if input_length_ is None:
            h_n, c_n = h_t, c_t
        if self.batch_first:
            output = output.permute([1, 0, 2]) # (batch_size, seq_length, hidden_size) 
        # output_packed if input packed
        if is_packed:
            output = pack_padded_sequence(output, input_length, batch_first=self.batch_first)
        return output, (h_n, c_n)

    def cell(self, hidden_state, X: torch, UV_index):
        """ 计算LSTM一个步骤\n
        @param:\n
        :hidden_state: (hidden, cell),上一cell的隐状态\n
        :X : 输入 (batch_size, input_size)\n
        :UV_index : torch()\n
        @return:\n
        :(h_new, c_new): (batch_size, hidden_size)\n
        """
        h_t, c_t = hidden_state # (batch_size, hidden_size)
        h_t = h_t.unsqueeze(dim=1).unsqueeze(dim=3) # (batch_size, 1, hidden_size, 1)
        c_t = c_t.unsqueeze(dim=1).unsqueeze(dim=3)
        X = X.unsqueeze(dim=2) # (batch_size, input_size, 1)
        # input gate
        i_x = torch.matmul(self.W_i, X) # (batch_size, hidden_size, 1)
        U_i = self.U_i[UV_index].unsqueeze(dim=0) # (1, k, hidden_size, hidden_size)
        V_i = self.V_i[UV_index].unsqueeze(dim=0) # (1, k, hidden_size, hidden_size)
        i_h = torch.matmul(U_i, h_t) # (bacth_size, k, hidden_size, 1)
        i_h = torch.sum(i_h, dim=1) # (batch_size, hidden_size, 1)
        i_c = torch.matmul(V_i, c_t) # (batch_size, k, hidden_size, 1)
        i_c = torch.sum(i_c, dim=1) # (batch_size, hidden_size, 1)
        i = self.sigmoid(i_x + i_h + i_c).squeeze(dim=2) # (batch_size, hidden_size)
        # forget gate
        f_x = torch.matmul(self.W_f, X)
        U_f = self.U_f[UV_index].unsqueeze(dim=0)
        V_f = self.V_f[UV_index].unsqueeze(dim=0)
        f_h = torch.matmul(U_f, h_t)
        f_h = torch.sum(f_h, dim=1)
        f_c = torch.matmul(V_f, c_t)
        f_c = torch.sum(f_c, dim=1)
        f = self.sigmoid(f_x + f_h + f_c).squeeze(dim=2)
        # output gate
        o_x = torch.matmul(self.W_o, X)
        U_o = self.U_o[UV_index].unsqueeze(dim=0)
        V_o = self.V_o[UV_index].unsqueeze(dim=0)
        o_h = torch.matmul(U_o, h_t)
        o_h = torch.sum(o_h, dim=1)
        o_c = torch.matmul(V_o, c_t)
        o_c = torch.sum(o_c, dim=1)
        o = self.sigmoid(o_x + o_h + o_c).squeeze(dim=2)
        # next cell
        c_tilde_x = torch.matmul(self.W_c, X)
        U_c = self.U_c[UV_index].unsqueeze(dim=0)
        c_tilde_h = torch.matmul(U_c, h_t)
        c_tilde_h = torch.sum(c_tilde_h, dim=1)
        c_tilde = self.tanh(c_tilde_x + c_tilde_h).squeeze(dim=2)
        c_t = c_t.squeeze(dim=3).squeeze(dim=1)
        c_new = torch.mul(f, c_t) + torch.mul(i, c_tilde)
        # next hidden
        h_new = torch.mul(o, self.tanh(c_new))
        return (h_new, c_new)

    def init_hidden_state(self, batch_size):
        """ 初始化开始的隐状态为0
        """
        return (torch.zeros(size=[batch_size, self.hidden_size], dtype=torch.float32).to(self.device),
                torch.zeros(size=[batch_size, self.hidden_size], dtype=torch.float32).to(self.device))
    
    def get_UV(self, t):
        """ 根据t，返回这一步骤参与的group
        :t: int时间步
        """
        max_index = np.math.log2(t) + 1e-6
        indexs = torch.arange(0, min(self.g, max_index), 1, dtype=torch.long)
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