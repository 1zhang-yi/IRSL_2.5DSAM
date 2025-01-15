import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class CrossSliceAttention(nn.Module):
    def __init__(self, input_channels):
        super(CrossSliceAttention, self).__init__()
        self.linear_q_a = nn.Conv2d(in_channels=input_channels, out_channels=input_channels//2, kernel_size=(1, 1), bias=False)
        self.linear_q_b = nn.Conv2d(in_channels=input_channels//2, out_channels=input_channels, kernel_size=(1, 1),
                                    bias=False)
        self.linear_k_a = nn.Conv2d(in_channels=input_channels, out_channels=input_channels//2, kernel_size=(1, 1), bias=False)
        self.linear_k_b = nn.Conv2d(in_channels=input_channels//2, out_channels=input_channels, kernel_size=(1, 1),
                                  bias=False)
        self.linear_v_a = nn.Conv2d(in_channels=input_channels, out_channels=input_channels//2, kernel_size=(1, 1), bias=False)
        self.linear_v_b = nn.Conv2d(in_channels=input_channels//2, out_channels=input_channels, kernel_size=(1, 1),
                                  bias=False)

    def forward(self, features):
        B, _, H, W = features.shape
        q = self.linear_q_b(self.linear_q_a(features))
        q = q.view(q.size(0), -1)
        k = self.linear_k_b(self.linear_k_a(features))
        k = k.view(k.size(0), -1)
        v = self.linear_v_b(self.linear_v_a(features))
        v = v.view(k.size(0), -1)
        x = torch.matmul(q, k.permute(1, 0))/np.sqrt(q.size(1))
        x = torch.softmax(x, dim=1)
        out = (x @ v).view(B, H, W, -1).permute(0, 3, 1, 2)

        return out


class MultiHeadedCrossSliceAttentionModule(nn.Module):
    def __init__(self, input_channels, heads=3, input_size=(32, 32), batch_size=12):
        super(MultiHeadedCrossSliceAttentionModule, self).__init__()
        self.attentions = []
        self.linear1 = nn.Conv2d(in_channels=heads*input_channels, out_channels=input_channels, kernel_size=(1,1))
        self.norm1 = nn.LayerNorm([batch_size, input_channels, input_size[0], input_size[1]])
        self.linear2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=(1,1))
        self.norm2 = nn.LayerNorm([batch_size, input_channels, input_size[0], input_size[1]])

        for i in range(heads):
            self.attentions.append(CrossSliceAttention(input_channels))
        self.attentions = nn.Sequential(*self.attentions)

    def forward(self, features):
        features = features.permute(0, 3, 1, 2)
        for i in range(len(self.attentions)):
            x_ = self.attentions[i](features)
            if i == 0:
                x = x_
            else:
                x = torch.cat((x, x_), dim=1)
        out = self.linear1(x)
        x = F.gelu(out) + features
        out_ = self.norm1(x)
        out = self.linear2(out_)
        x = F.gelu(out) + out_
        out = self.norm2(x).permute(0, 2, 3, 1)
        return out

# net = MultiHeadedCrossSliceAttentionModule(input_channels=768, heads=3, input_size=(32, 32), batch_size=12)
# x = torch.rand((12, 32, 32, 768))
# y = net(x)
# print(y.shape)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, is_pe_learnable=True, max_len=20):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        pe = torch.zeros(max_len, d_model, 1, 1)
        pe[:, 0::2, 0, 0] = torch.sin(position*div_term)
        pe[:, 1::2, 0, 0] = torch.cos(position*div_term)
        self.pe = nn.Parameter(pe.clone(), is_pe_learnable)
        #self.register_buffer('pe',self.pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :, :, :]

    def get_pe(self):
        return self.pe[:, :, 0, 0]


