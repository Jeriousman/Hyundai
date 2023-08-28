# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 20:44:51 2023

@author: hojun
"""

import torch
import torch.nn.functional as F
import torch.nn as nn


class PreMLP(nn.Module):
    def __init__(self, img_dim, seq_len, embed_dim):
        super(PreMLP, self).__init__()
        
        self.img_dim = img_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        # assert self.seq_len * self.embed_dim == 0, 'hidden dimension must be equal to sequence length multiplied by embedding dimension'
        
        
        self.linear1 = nn.Linear(self.img_dim, self.img_dim)
        # self.act1 = nn.LeakyReLU()
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(self.img_dim, self.img_dim)
        # self.act2 = nn.LeakyReLU()
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(self.img_dim, self.seq_len*self.embed_dim)
        
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        
        x = self.linear2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        x = self.linear3(x)
        x = x.view(-1, self.seq_len, self.embed_dim)
        # x = self.linear3(x)
        
        return x
    
# data.shape
# model = PreMLP(900, 4800, 300, 16).to(device)
# zz = model(data)
# zz.shape

class PostMLP(nn.Module):
    def __init__(self, seq_len, embed_dim, label_dim):
        super(PostMLP, self).__init__()
        
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.label_dim = label_dim
        
        self.linear1 = nn.Linear(self.embed_dim * self.seq_len, self.embed_dim * self.seq_len)
        # self.act1 = nn.LeakyReLU()
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        
        self.linear2 = nn.Linear(self.embed_dim * self.seq_len, self.embed_dim * self.seq_len)
        # self.act2 = nn.LeakyReLU()
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        
        self.linear3 = nn.Linear(self.embed_dim * self.seq_len, self.label_dim)
        
        
        
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.act1(x)
        
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.act2(x)
        
        x = self.linear3(x)
        
        return x
# model = PostMLP(900, 189).to(device)
# zz = model(data)
# zz.shape
    

# q= torch.randn(5, 4, 256).to('cuda')
# q.shape
# attn = nn.MultiheadAttention(64, 4, batch_first=True)
# a, b = attn(q,q,q)
# a.shape
# b.shape

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim = 32 , num_heads = 2, batch_first=True):
        super(SelfAttentionBlock, self).__init__()
        '''
        input: batch, seq, embed
        output: (batch, seq, embed), (batch, seq, seq)
        '''
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        assert self.embed_dim % self.num_heads == 0, 'embed_dim must be divisible by nun_heads '
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=self.batch_first)
        
    def forward(self, x):

        attn_output, attn_output_weights = self.multihead_attn(x, x, x)
        return attn_output, attn_output_weights

# zz.shape
# zzz = zz.view(-1, 64, 4)
# zzz.shape
# zz.shape
# model = SelfAttentionBlock(16, 2, True).to(device)
# xx, xxx = model(zz)
# xx.shape
# xxx.shape
