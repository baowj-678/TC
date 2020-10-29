from torch.nn.utils.rnn import *
import torch

a = torch.zeros(6, 7)
b = torch.tensor([7,6,4,3,2,1])
print(a)
a = pack_padded_sequence(a, b, batch_first=True)
print(pad_packed_sequence(a))
