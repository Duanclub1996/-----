# language: Python
import os
import sys
import math
import tqdm
import time
import pathlib
import glob
import scipy
import torch
import logging
import torchvision 
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import scipy.io as scio
import matplotlib.pyplot as plt
import torch.nn.functional  as F
from einops import rearrange, repeat
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from einops.layers.torch import Rearrange, Reduce
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split



class MultiHeadAttention(nn.Module):
   def __init__(self, d_model, num_heads):
       super(MultiHeadAttention, self).__init__()
       self.d_model = d_model
       self.num_heads = num_heads
       self.head_dim = d_model // num_heads  # 32

       self.WQ = nn.Linear(d_model, d_model)
       self.WK = nn.Linear(d_model, d_model)
       self.WV = nn.Linear(d_model, d_model)

       self.WO = nn.Linear(d_model, d_model)

   def scaled_dot_product_attention(self, Q, K, V, mask=None):
       d_k = Q.size(-1)
       scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

       if mask is not None:
           scores = scores.masked_fill(mask == 0, float('-inf'))

       attention_weights = F.softmax(scores, dim=-1)
       output = torch.matmul(attention_weights, V)

       return output, attention_weights

   def split_heads(self, x):
    #    print(x.shape)
    #    print(self.head_dim)
    #    print(self.num_heads)
       batch_size, seq_length, _ = x.size()
       x = x.view(batch_size, seq_length, self.num_heads, self.head_dim)
       return x.transpose(1, 2)

   def combine_heads(self, x):
    #    print(x.shape)
       batch_size, _, seq_length, _ = x.size()
       x = x.transpose(1, 2)
       return x.contiguous().view(batch_size, seq_length, self.d_model)

   def forward(self, Q, K, V, mask=None):
    #    print(Q.shape)
    #    print(Q.shape)
       Q = self.split_heads(self.WQ(Q))
       K = self.split_heads(self.WK(K))
       V = self.split_heads(self.WV(V))

       output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
       output = self.combine_heads(output)
       output = self.WO(output)

       return output, attention_weights

class EncoderLayer(nn.Module):
   def __init__(self, d_model, num_heads, dff, dropout=0.1):
       super(EncoderLayer, self).__init__()
       self.mha = MultiHeadAttention(d_model, num_heads)
       self.ffn = nn.Sequential(
           nn.Linear(d_model, dff),
           nn.ReLU(),
           nn.Dropout(dropout),
           nn.Linear(dff, d_model),
           nn.Dropout(dropout)
       )

       self.layernorm1 = nn.LayerNorm(d_model)
       self.layernorm2 = nn.LayerNorm(d_model)

       self.dropout = nn.Dropout(dropout)

   def forward(self, x, mask=None):
    #    print(x.shape)
       attn_output, _ = self.mha(x, x, x, mask)
       out1 = self.layernorm1(x + self.dropout(attn_output))
       ffn_output = self.ffn(out1)
       out2 = self.layernorm2(out1 + self.dropout(ffn_output))

       return out2


class VIT(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, image_size, patch_size, num_classes, dropout=0.1):
      super(VIT, self).__init__()
      self.num_layers = num_layers
      self.d_model = d_model
      self.num_heads = num_heads
      self.dff = dff
      self.image_size = image_size
      self.patch_size = patch_size
      self.num_classes = num_classes
      self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),  
            nn.Linear(192, d_model)
        )
      self.positional_encoding = nn.Parameter(torch.randn(1, 1 + (image_size // patch_size) ** 2, d_model))
      self.dropout = nn.Dropout(dropout)
      self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)])
      self.layernorm = nn.LayerNorm(d_model)
      self.fc = nn.Linear(d_model, num_classes)
      self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
      
    def forward(self, x):
       x = self.patch_embedding(x)
       b, n, _ = x.shape  # shape (b, n, 768)
       cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
       x = torch.cat((cls_tokens, x), dim=1)
       x = x + self.positional_encoding
       x = self.dropout(x)
    #    print(x.shape)
       mask = None
       for layer in self.encoder_layers:
            x = layer(x, mask)
       x = self.layernorm(x)
       x = torch.mean(x, dim=1)
       x = self.fc(x)
       return F.softmax(x,dim=-1)

# vit = VIT(num_layers=12, d_model=768, num_heads=12, dff=3072, image_size=224, patch_size=16, num_classes=10)
