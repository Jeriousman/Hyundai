#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 16:37:39 2022

@author: hojun
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def activation_layer(activation_name="relu", emb_dim=None):
    """Construct activation layers

    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation

    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation


class MLPLayers(nn.Module):
    r"""MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """


    def __init__(
        self, layers, dropout=0.0, activation="relu", norm=False, init_method=None
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.norm = norm
        self.init_method = init_method
        

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:]) ##한칸씩밀면서 input output이 자동으로 설정된다   ##layers = ex) [40, 256, 256, 256, 1]  
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.norm == 'bn':
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            elif self.norm == 'in':
                mlp_modules.append(nn.InstanceNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            if self.init_method == "norm":
                module.weight.data.normal_(module.weight.data, 0, 0.01)
                # module.weight.data.normal_(0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):  ##결국에 마지막 output dim은 1이라 하나의 scalar값만나온다 
        return self.mlp_layers(input_feature)


class PreMLP(nn.Module):
    def __init__(self, input_dim, height, width, embed_dim=64):
        super(PreMLP, self).__init__()
        
        '''
        embed_dim: multihead로 나뉘기 전의 embedding dimension
        '''
        
        self.input_dim = input_dim
        self.seq_len = height * width
        self.embed_dim = embed_dim
        # assert self.seq_len * self.embed_dim == 0, 'hidden dimension must be equal to sequence length multiplied by embedding dimension'
        
        
        self.linear1 = nn.Linear(self.input_dim, self.input_dim)
        # self.act1 = nn.LeakyReLU()
        self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(self.input_dim, self.input_dim)
        # self.act2 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.linear3 = nn.Linear(self.input_dim, self.seq_len*self.embed_dim)
        
        
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
    


class PostMLP(nn.Module):
    def __init__(self, height, width, embed_dim, output_dim):
        super(PostMLP, self).__init__()
        
        self.seq_len = height * width
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        
        self.linear1 = nn.Linear(self.embed_dim * self.seq_len, self.embed_dim * self.seq_len)
        # self.act1 = nn.LeakyReLU()
        self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.linear2 = nn.Linear(self.embed_dim * self.seq_len, self.embed_dim * self.seq_len)
        # self.act2 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.linear3 = nn.Linear(self.embed_dim * self.seq_len, self.output_dim)  ##RGB 이기 때문에 3. gray scale이면 1
        
        
        
        
    def forward(self, x):
        x = x.view(-1, self.seq_len*self.embed_dim)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.act1(x)
        
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.act2(x)
        
        x = self.linear3(x)
        
        return x




class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim=64 , num_heads=4, batch_first=True):
        super(SelfAttentionBlock, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        assert self.embed_dim % self.num_heads == 0, 'embed_dim must be divisible by nun_heads '
        
        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, self.batch_first)
        
    def forward(self, x):

        attn_output, attn_output_weights = self.multihead_attn(x, x, x)
        
        
        return attn_output
        



"""
Architecture based on InfoGAN paper.
"""

class InfoGeneratorBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.tconv1 = nn.ConvTranspose2d(168, 448, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(448)

        self.tconv2 = nn.ConvTranspose2d(448, 256, 4, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.tconv3 = nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False)

        self.tconv4 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)

        self.tconv5 = nn.ConvTranspose2d(64, 3, 4, 2, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))

        img = torch.tanh(self.tconv5(x))

        return img

class InfoDiscriminatorBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)

        return x

class InfoDHeadBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(256, 1, 4)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))

        return output

class InfoQHeadBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(256, 128, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_disc = nn.Conv2d(128, 40, 1)
        self.conv_mu = nn.Conv2d(128, 4, 1)
        self.conv_var = nn.Conv2d(128, 4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        disc_logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var

