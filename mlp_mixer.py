# MLP-Mixer

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
from functools import partial
from einops import rearrange, repeat
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from einops.layers.torch import Rearrange, Reduce
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer( image_size, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        # 1. 将图片拆成多个patches
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        # 2. 用一个全连接网络对所有patch进行处理，提取出tokens
        nn.Linear((patch_size ** 2)*3 , dim), # 
        # 3. 经过N个Mixer层，混合提炼特征信息
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        # 4. 最后一个全连接层进行类别预测
        nn.Linear(dim, num_classes)
    )