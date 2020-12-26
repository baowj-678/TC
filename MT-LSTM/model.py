""" MT-LSTM文本分类模型
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/31
"""
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from MT_LSTM import MT_LSTM
import torch


class Model(nn.Module):
    def __init__(self, vocab, hidden_size, device, ave_length=19, embedding_size=128, num_class=5, embedding=None, p=0.1, g=3):
        super(Model, self).__init__()
        self.device = device
        # Embedding
        if embedding is None:
            self.embedding = nn.Embedding(num_embeddings=len(vocab),
                                          embedding_dim=embedding_size,
                                          padding_idx=0).to(device)
        else:
            self.embedding = embedding
        # MT_LSTM
        self.mt_lstm = MT_LSTM(ave_length=ave_length,
                               input_size=embedding_size,
                               hidden_size=hidden_size,
                               batch_first=True,
                               device=device,
                               g=g)
        # MLP
        self.linear = nn.Linear(in_features=2*hidden_size,
                                out_features=num_class,
                                bias=True).to(device)
        # Dropout
        self.dropout = nn.Dropout(p=p)
        # Softmax
        self.softmax = nn.Softmax(dim=1)
        # Loss
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, X, label=None):
        """ 前向传播
        @param:
        :X: packed_sentence:
        :label: (torch)
        """
        X = X.to(self.device)
        input, length = pad_packed_sequence(X, batch_first=True)
        # Embedding
        input = self.embedding(input)
        # LSTM
        input = pack_padded_sequence(input, length, batch_first=True)
        output, hidden_state = self.mt_lstm(input)
        output, out_length = pad_packed_sequence(output)
        h, c = hidden_state # (batch_size, hidden_size)
        out = torch.cat((h, c), dim=1) # (batch_size, 2*hidden_size)
        # MLP
        predict = self.linear(out) # (batch_size, num_class)
        if label is None:
            return torch.argmax(predict, dim=1) # (batch_size)
        else:
            label = label.to(self.device).squeeze(dim=1)
            loss = self.loss(predict, label)
            return loss
