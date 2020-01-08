import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch import nn

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MultiHeadAttention', 'ScaledDotProductAttention']


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.

        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )

class FeedForward(nn.Module):
    def __init__(self, hidden_dim, eps = 1.e-8):
        super(self.__class__, self).__init__()
        self.ln = nn.LayerNorm(hidden_dim, eps = eps)
        self.lr1 = nn.Linear(hidden_dim, 4*hidden_dim, bias =True)
        self.lr2 = nn.Linear(4*hidden_dim, hidden_dim, bias =True)
        
        
    def forward(self, x):
        out = F.relu(self.lr1(x))
        out = self.lr2(out) + x
        out = self.ln(out)
        
        return out
      

class conv1d(nn.Conv1d):
    def __init__(self,input_, output_channels, dilation=1 ,groups=1, kernel_size = 1,bias=False, causal = False):
        #print(dilation)
        if causal:
            padding=(kernel_size-1)*dilation
        else:
            padding=(kernel_size-1)*dilation//2
        super(conv1d,self).__init__(input_, output_channels, kernel_size,
                                           stride=1, padding=padding, dilation=dilation,
                                           groups=groups, bias=bias)
    def forward(self,x):
        output=super(conv1d,self).forward(x)
        #print(output.shape)
        return output[:,:,:x.size(2)]
  

class ResidualBlock(nn.Module):

    def __init__( self,input_,input_channel, interm_channels=20, output_channels=5, kernel_size=1, dilation=1, causal=True):
        super(ResidualBlock, self).__init__()
        output_channels = output_channels or input_
        interm_channels = interm_channels or input_ // 2
        self.layernorm1 = nn.LayerNorm(input_,eps=1.e-8)
        self.layernorm2 = nn.LayerNorm(input_,eps=1.e-8)
        self.layernorm3 = nn.LayerNorm(input_,eps=1.e-8)
        
        self.conv1 = nn.Conv1d(input_channel, interm_channels, kernel_size=1)
       # print('int',input_)
        self.conv2 = conv1d(
            interm_channels, interm_channels,dilation=dilation,kernel_size=kernel_size, causal=causal)
        self.conv3 = nn.Conv1d(interm_channels, output_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layernorm1(x)
        #print('ss',out.shape)
        out=self.relu(out)
        #print('kk',out.shape[1])
        out = self.conv1(out)
        #print('gg',out.size())
        out = self.layernorm2(out)
        #print('ll',out)
        out = self.conv2(self.relu(out))
        out = self.layernorm3(out)
        out = self.conv3(self.relu(out))
        out += x
        return out

    
def Sin_position_encoding(inputs, mask, repr_dim):
    T = inputs.size(1)
    pos = torch.arange(0, T, dtype = torch.float64).view(-1, 1)
    i = np.arange(0, repr_dim, 2, np.float64)
    denom = torch.tensor(np.reshape(np.power(10000.0, i / repr_dim), [1, -1]))
    enc = torch.cat([torch.sin(pos/denom), torch.cos(pos/denom)], dim = 1)
    return  (enc.transpose(0, 1).float() * mask.view(-1)).transpose(0, 1)



