import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch import nn
import modules
from tqdm import trange

class attention(nn.Module):
    def __init__(self, size_layer, embedded_size, size, output_size,
                 num_blocks = 2,
                 num_heads = 8,
                 min_freq = 50):
        super(self.__class__, self).__init__()
        self.size_layer = size_layer
        self.embedded_size = embedded_size
        self.size = size
        self.output_size = output_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads 
        self.min_freq = min_freq
        
        self.linear1 = nn.Linear(6, embedded_size)
        self.linear2 = nn.Linear(embedded_size, output_size)

        self.dropout = nn.Dropout(0.25)
        self.multihead_att1 = modules.MultiHeadAttention(embedded_size, self.num_heads)
        self.feedforward1 = modules.FeedForward(hidden_dim=embedded_size)
        
        self.multihead_att2 = modules.MultiHeadAttention(embedded_size, self.num_heads)
        self.feedforward2 = modules.FeedForward(hidden_dim=embedded_size)
        
    def forward(self, x):
        encoder_embedded = self.dropout(self.linear1(x))
        en_masks = torch.sign(torch.mean(x, dim = 2))
        encoder_embedded += modules.Sin_position_encoding(x, en_masks, self.embedded_size)
        encoder_embedded = self.multihead_att1(q = encoder_embedded,
                                          k = encoder_embedded,
                                          v = encoder_embedded,
                                          mask = en_masks)
        encoder_embedded = self.feedforward1(encoder_embedded)
      
        encoder_embedded = self.multihead_att2(q = encoder_embedded,
                                          k = encoder_embedded,
                                          v = encoder_embedded,
                                          mask = en_masks)
        encoder_embedded = self.feedforward2(encoder_embedded)
        logits = self.linear2(encoder_embedded[-1])
        return logits
      
    def save(self, path = 'attention_is_all_you_need.pth'):
        torch.save(self.state_dict(), path)
        
        
    def load(self, path = 'attention_is_all_you_need.pth'):
        self.load_state_dict(torch.load(path))
        self.eval()
        
        
        
        
class bidirectional_lstm(nn.Module):
    def __init__(self,
                 num_layers, 
                 size, 
                 size_layer, 
                 output_size, 
                 forget_bias = 0.):
        super(self.__class__, self).__init__()
        self.size_layer = size_layer
        self.num_layers = num_layers
        self.size = size
        self.output_size = output_size
        self.forget_bias = forget_bias
        
        self.lstm = nn.LSTM(size, size_layer, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(size_layer*2, output_size)
    def forward(self,x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.size_layer) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.size_layer)
#         print(x.shape,h0.shape,c0.shape)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
#         out= torch.cat((out),2)

    def save(self, path = 'bidirectional_lstm.pth'):
        torch.save(self.state_dict(), path)
        
    def load(self, path = 'bidirectional_lstm.pth'):
        self.load_state_dict(torch.load(path))
        self.eval()
    
    


class ByteNet(nn.Sequential):

    def __init__(self, num_channels=5, input_channel=5,num_sets=1, dilation_rates=[1,2,4,8,16,1,2,4,8,16], kernel_size=1, block=modules.ResidualBlock, causal=True):
        super(ByteNet, self).__init__()
        for s in range(num_sets):
            for r in dilation_rates:
                self.add_module('block%s_%s' % (s, r),
                                block(num_channels, input_channel,kernel_size=kernel_size, dilation=r, causal=causal))
                
    def save(self, path = 'byte_net.pth'): 
        torch.save(self.state_dict(), path) 

    def load(self, path = 'byte_net.pth'): 
        self.load_state_dict(torch.load(path)) 
        self.eval()