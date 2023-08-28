import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from typing import List
import torch
from torch import tensor
import pandas as pd
import os
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from utils import gradient_penalty, save_checkpoint, load_checkpoint
# from model import Discriminator, Generator, initialize_weights
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
# import cv2
from layers import activation_layer, MLPLayers, SelfAttentionBlock, PreMLP, PostMLP

##제원정보에 대한 업데이트 해야한다. 
class Discriminator(nn.Module):
    def __init__(self, mlp_dims, num_feat, activation, agg_mode, height_size, width_size, num_channel, dropout, norm): ##in_features = 784
        super().__init__()
        self.mlp_dims = mlp_dims
        self.num_feat = num_feat  ##durability and weight so 2.
        self.activation = activation
        self.agg_mode = agg_mode
        self.height_size = height_size
        self.width_size = width_size
        self.num_channel = num_channel
        self.dropout = dropout
        self.norm = norm
        
        self.linear_image = nn.Linear(self.height_size*self.width_size*self.num_channel, self.mlp_dims[0])
        self.linear_durability_weight = nn.Linear(self.num_feat, self.mlp_dims[0])
        self.linear_jewon = nn.Linear(self.height_size*self.width_size*self.num_channel, self.mlp_dims[0])

        self.mlp_image = MLPLayers(layers = self.mlp_dims, activation = self.activation, dropout=self.dropout, norm=self.norm)
        self.mlp_durability_weight = MLPLayers(layers = self.mlp_dims, activation = self.activation, dropout=self.dropout, norm=self.norm)
        self.mlp_jewon = MLPLayers(layers = self.mlp_dims, activation = self.activation, dropout=self.dropout, norm=self.norm)
        
        self.linear_concat = nn.Linear(self.mlp_dims[-1] * 3, self.mlp_dims[-1]) ## 3 bcuz noise, durability_weight and jewon three inputs
        self.linear_final = nn.Linear(self.mlp_dims[-1], 1)
        

        
        self.sigmoid = nn.Sigmoid()
       

    def forward(self, image, durability_weight, jewon): #durability, weight):
        if self.agg_mode == 'sum':
            image_hidden = self.linear_image(image) ## [B, z_dim] -> [B, mlp_dims[0]]

            image_output =  self.mlp_image(image_hidden)  #noise this is [B, mlp_dims[0]] -> [B, mlp_dims[-1]]. But it will be [B, z_dim, 1, 1] at training stage.
            
            durability_weight_hidden_states = self.linear_durability_weight(durability_weight) ##[B, num_feat] -> [B, mlp_dims[0]]  ##durability_weight has 2 as dimension size
            durability_weight_output = self.mlp_durability_weight(durability_weight_hidden_states)  ## [B, mlp_dims[0]] ->  [B, mlp_dims[-1]]
            
            jewon_hidden = jewon.view(-1, self.height_size*self.width_size*self.num_channel) ##flatten
            jewon_hidden_output = self.linear_jewon(jewon_hidden)
            jewon_hidden_output = self.mlp_jewon(jewon_hidden_output)
            
            summed_output = image_output + durability_weight_output + jewon_hidden_output 
            final_output = self.linear_final(summed_output) ##back to image size to produce images
            final_output = self.sigmoid(final_output) ##activate this if data is normalized to [0, 1]

            # final_output = self.sigmoid(summed_output) ##activate this if data is normalized to [-1, 1]
            
        if self.agg_mode == 'concat':
            
            image_hidden = self.linear_image(image)
            image_output =  self.mlp_image(image_hidden)

            durability_weight_hidden_states = self.linear_durability_weight(durability_weight) 
            durability_weight_output = self.mlp_durability_weight(durability_weight_hidden_states)

            jewon_hidden = jewon.view(-1, self.height_size*self.width_size*self.num_channel) ##flatten
            jewon_hidden = self.linear_jewon(jewon_hidden)
            jewon_hidden_output = self.mlp_jewon(jewon_hidden)

            concat_output = torch.concat([image_output, durability_weight_output, jewon_hidden_output], axis = -1)
            concat_output = self.linear_concat(concat_output)
            final_output = self.linear_final(concat_output) ##back to image size to produce images
            final_output = self.sigmoid(final_output) ##activate this if data is normalized to [0, 1]
        
        
        return final_output



class Critic(nn.Module):
    def __init__(self, mlp_dims, num_feat, activation, agg_mode, height_size, width_size, num_channel, dropout, norm): ##in_features = 784
        super().__init__()
        '''
        for WGAN. no sigmoid is used.
        '''
        
        self.mlp_dims = mlp_dims
        self.num_feat = num_feat  ##durability and weight so 2.
        self.activation = activation
        self.agg_mode = agg_mode
        self.height_size = height_size
        self.width_size = width_size
        self.num_channel = num_channel
        self.dropout = dropout ## 0.3
        self.norm = norm  ##False
        
        self.linear_image = nn.Linear(self.height_size*self.width_size*self.num_channel, self.mlp_dims[0])
        self.linear_durability_weight = nn.Linear(self.num_feat, self.mlp_dims[0])
        self.linear_jewon = nn.Linear(self.height_size*self.width_size*self.num_channel, self.mlp_dims[0])

        self.mlp_image = MLPLayers(layers = self.mlp_dims, activation = self.activation, dropout=self.dropout, norm=self.norm)
        self.mlp_durability_weight = MLPLayers(layers = self.mlp_dims, activation = self.activation, dropout=self.dropout, norm=self.norm)
        self.mlp_jewon = MLPLayers(layers = self.mlp_dims, activation = self.activation, dropout=self.dropout, norm=self.norm)
        
        self.linear_concat = nn.Linear(self.mlp_dims[-1] * 3, self.mlp_dims[-1]) ## 3 bcuz noise, durability_weight and jewon three inputs
        self.linear_final = nn.Linear(self.mlp_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()
       

    def forward(self, image, durability_weight, jewon): #durability, weight):
        if self.agg_mode == 'sum':
            image_hidden = self.linear_image(image) ## [B, z_dim] -> [B, mlp_dims[0]]

            image_output =  self.mlp_image(image_hidden)  #noise this is [B, mlp_dims[0]] -> [B, mlp_dims[-1]]. But it will be [B, z_dim, 1, 1] at training stage.
            
            durability_weight_hidden_states = self.linear_durability_weight(durability_weight) ##[B, num_feat] -> [B, mlp_dims[0]]  ##durability_weight has 2 as dimension size
            durability_weight_output = self.mlp_durability_weight(durability_weight_hidden_states)  ## [B, mlp_dims[0]] ->  [B, mlp_dims[-1]]
            
            jewon_hidden = jewon.view(-1, self.height_size*self.width_size*self.num_channel) ##flatten
            jewon_hidden_output = self.linear_jewon(jewon_hidden)
            jewon_hidden_output = self.mlp_jewon(jewon_hidden_output)
            
            summed_output = image_output + durability_weight_output + jewon_hidden_output 
            final_output = self.linear_final(summed_output) ##back to image size to produce images
            
            ##WGAN 에서는 sigmoid를 사용하지 않는다.
            # final_output = self.sigmoid(final_output) ##activate this if data is normalized to [0, 1]


            
        if self.agg_mode == 'concat':
            
            image_hidden = self.linear_image(image)
            image_output =  self.mlp_image(image_hidden)

            durability_weight_hidden_states = self.linear_durability_weight(durability_weight) 
            durability_weight_output = self.mlp_durability_weight(durability_weight_hidden_states)

            jewon_hidden = jewon.view(-1, self.height_size*self.width_size*self.num_channel) ##flatten
            jewon_hidden = self.linear_jewon(jewon_hidden)
            jewon_hidden_output = self.mlp_jewon(jewon_hidden)

            concat_output = torch.concat([image_output, durability_weight_output, jewon_hidden_output], axis = -1)
            concat_output = self.linear_concat(concat_output)
            final_output = self.linear_final(concat_output) ##back to image size to produce images
            final_output = self.sigmoid(final_output) ##activate this if data is normalized to [0, 1]
        
        
        return final_output



class Generator(nn.Module):
    def __init__(self, mlp_dims, z_dim, num_feat, activation, agg_mode, output_mode, height_size, width_size, num_channel, dropout, norm):
        super().__init__()
        self.mlp_dims = mlp_dims
        self.agg_mode = agg_mode
        self.output_mode = output_mode
        self.z_dim = z_dim
        self.num_feat = num_feat
        self.height_size = height_size
        self.width_size = width_size
        self.num_channel = num_channel
        self.activation = activation
        self.dropout = dropout ## 0.3
        self.norm = norm  ##False
        self.linear_noise = nn.Linear(self.z_dim, self.mlp_dims[0])
        self.linear_durability_weight = nn.Linear(self.num_feat, self.mlp_dims[0])
        self.linear_jewon = nn.Linear(self.height_size*self.width_size*self.num_channel, self.mlp_dims[0])

        self.mlp_noise = MLPLayers(layers = self.mlp_dims, activation = self.activation, dropout=self.dropout, norm=self.norm)
        self.mlp_durability_weight = MLPLayers(layers = self.mlp_dims, activation = self.activation, dropout=self.dropout, norm=self.norm)
        self.mlp_jewon = MLPLayers(layers = self.mlp_dims, activation = self.activation, dropout=self.dropout, norm=self.norm)
        
        self.linear_concat = nn.Linear(self.mlp_dims[-1] * 3, self.mlp_dims[-1]) ## 3 bcuz noise, durability_weight and jewon three inputs
        self.linear_final = nn.Linear(self.mlp_dims[-1], self.num_channel * self.height_size * self.width_size) ##back to image size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, noise, durability_weight, jewon):  ##image = [B, mlp_dims[0]]  [B, 256]
        if self.agg_mode == 'sum':
            noise_hidden = self.linear_noise(noise) ## [B, z_dim] -> [B, mlp_dims[0]]
            noise_output =  self.mlp_noise(noise_hidden)  #noise this is [B, mlp_dims[0]] -> [B, mlp_dims[-1]]. But it will be [B, z_dim, 1, 1] at training stage.     
            
            durability_weight_hidden_states = self.linear_durability_weight(durability_weight) ##[B, num_feat] -> [B, mlp_dims[0]]  ##durability_weight has 2 as dimension size
            durability_weight_output = self.mlp_durability_weight(durability_weight_hidden_states)  ## [B, mlp_dims[0]] ->  [B, mlp_dims[-1]]
            
            jewon_hidden = jewon.view(-1, self.height_size*self.width_size*self.num_channel) ##flatten
            jewon_hidden_output = self.linear_jewon(jewon_hidden)
            jewon_hidden_output = self.mlp_jewon(jewon_hidden_output)
            
            summed_output = noise_output + durability_weight_output + jewon_hidden_output
            
            final_output = self.linear_final(summed_output) ##back to image size to produce images
            
            if self.output_mode == 'none':
                final_output = final_output
            elif self.output_mode == 'sigmoid':
                final_output = self.sigmoid(final_output) ##activate this if data is normalized to [0, 1]
            elif self.output_mode == 'tanh':
                final_output = self.tanh(self.final_output) ##activate this if data is normalized to [-1, 1] 
            
        if self.agg_mode == 'concat':
            noise_hidden = self.linear_noise(noise)
            noise_output =  self.mlp_noise(noise_hidden)
            
            durability_weight_hidden_states = self.linear_durability_weight(durability_weight) 
            durability_weight_output = self.mlp_durability_weight(durability_weight_hidden_states)
            
            jewon_hidden = jewon.view(-1, self.height_size*self.width_size*self.num_channel) ##flatten
            jewon_hidden = self.linear_jewon(jewon_hidden)
            jewon_hidden_output = self.mlp_jewon(jewon_hidden)
            
            concat_output = torch.concat([noise_output, durability_weight_output, jewon_hidden_output], axis = -1)
            # print(concat_output.shape)
            concat_output = self.linear_concat(concat_output)
            
            final_output = self.linear_final(concat_output) ##back to image size to produce images
            
            if self.output_mode == 'none':
                final_output = final_output
            elif self.output_mode == 'sigmoid':
                final_output = self.sigmoid(final_output) ##activate this if data is normalized to [0, 1]
            elif self.output_mode == 'tanh':
                final_output = self.tanh(final_output) ##activate this if data is normalized to [-1, 1] 
            
            

        return final_output




"""
Discriminator and Generator implementation from DCGAN paper,
with removed Sigmoid() as output from Discriminator (and therefor
it should be called critic)
"""


class DCGAN_Discriminator(nn.Module):
    def __init__(self, channels_img, height_size, width_size, features_d):
        super(DCGAN_Discriminator, self).__init__()
        self.height_size = height_size
        self.width_size = width_size
        self.channels_img = channels_img
        # self.activation = activation
        # self.dropout = dropout
        
        self.disc = nn.Sequential(
            # input: N x 3 x 10 x 30
            nn.Conv2d(
                channels_img+4, features_d, kernel_size=3, stride=1, padding=1
            ),
            # nn.Conv2d(
            #     channels_img, features_d, kernel_size=3, stride=1, padding=1
            # ),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, (3,5), (2,3), 1),  ##(N, features_d, 5, 10)
            self._block(features_d * 2, features_d * 4, (3,5), (2,3), 1),  ##(N, features_d, 3, 3)
            self._block(features_d * 4, 1, 3, 1, 0),    ##(N, features_d, 1, 1)
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            # nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )
        

        self.mlp_dura_weight = MLPLayers(layers = [2, self.height_size*self.width_size], activation = 'relu', dropout=0.2, norm='bn') ## hidden size 64
        self.mlp_jewon = MLPLayers(layers = [self.height_size*self.width_size*self.channels_img, self.height_size*self.width_size*self.channels_img], activation = 'relu', dropout=0.2, norm='bn') ## hidden size 64
        
       

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.3),
        )

    def forward(self, image, durability_weight, jewon):
        # print('jewon: ', jewon.shape)
        jewon = jewon.view(-1, self.height_size*self.width_size*self.channels_img) ##flatten
        jewon_hidden = self.mlp_jewon(jewon).view(jewon.shape[0], self.channels_img, self.height_size, self.width_size)  ## N, 3, H, W
        # jewon_hidden = self.fixed_pixel(jewon)  ## N, 3, H, W
        durability_weight_hidden = self.mlp_dura_weight(durability_weight).view(durability_weight.shape[0], 1, self.height_size, self.width_size)
        # print('print: ', durability_weight_hidden.shape)
        # x = image+durability_weight_hidden+jewon_hidden
        x = torch.cat([image, durability_weight_hidden, jewon_hidden], dim =1)
        return self.disc(x)
    


class DCGAN_Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, height_size, width_size, features_g, embed_size):
        super(DCGAN_Generator, self).__init__()
        self.height_size = height_size
        self.width_size = width_size
        self.channels_img = channels_img
        
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise + embed_size*2, features_g * 16, kernel_size=5, stride=1, padding=1),  # img: 3x3
            # self._block(channels_noise, features_g * 16, kernel_size=5, stride=1, padding=1),  # img: 3x3
            self._block(features_g * 16, features_g * 8, (5,8), (1,2), 1),  # img: 5x10
            self._block(features_g * 8, features_g * 4, (6,12), (1,2), 0),  # img: 10x30
            self._block(features_g * 4, 3, 3, 1, 1),  # img: 10x30
            # nn.ConvTranspose2d(
            #     features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            # ),
            # Output: N x channels_img x 10 x 30
            nn.Tanh(),
        )
        self.mlp_dura_weight = MLPLayers(layers = [2, embed_size], activation = 'relu', dropout=0.2, norm='bn') ## hidden size 64
        self.mlp_jewon = MLPLayers(layers = [self.height_size*self.width_size*self.channels_img, embed_size], activation = 'relu', dropout=0.2, norm='bn') ## hidden size 64
        

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, noise, durability_weight, jewon):
        # latent vector z : N x noise_dim  x 1 x1 
        durability_weight_embedding = self.mlp_dura_weight(durability_weight).unsqueeze(2).unsqueeze(3)
        jewon_embedding = self.mlp_jewon(jewon.view(jewon.shape[0], -1)).unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, durability_weight_embedding, jewon_embedding], dim=1)

        return self.net(x)





def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"



def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.BatchNorm1d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            
       
     
class DCGAN_Discriminator_onlyimage(nn.Module):
    def __init__(self, channels_img, height_size, width_size, features_d):
        super(DCGAN_Discriminator_onlyimage, self).__init__()
        self.height_size = height_size
        self.width_size = width_size
        self.channels_img = channels_img
        # self.activation = activation
        # self.dropout = dropout
        
        self.disc = nn.Sequential(
            # input: N x 3 x 10 x 30
            nn.Conv2d(
                channels_img, features_d, kernel_size=3, stride=1, padding=1
            ),
            # nn.Conv2d(
            #     channels_img, features_d, kernel_size=3, stride=1, padding=1
            # ),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, (3,5), (2,3), 1),  ##(N, features_d, 5, 10)
            self._block(features_d * 2, features_d * 4, (3,5), (2,3), 1),  ##(N, features_d, 3, 3)
            self._block(features_d * 4, 1, 3, 1, 0),    ##(N, features_d, 1, 1)
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            # nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )
        


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            # nn.LayerNorm2d(out_channels),
            # nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)
    


class DCGAN_Generator_onlyimage(nn.Module):
    def __init__(self, channels_noise, channels_img, height_size, width_size, features_g):
        super(DCGAN_Generator_onlyimage, self).__init__()
        self.height_size = height_size
        self.width_size = width_size
        self.channels_img = channels_img
        
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, kernel_size=5, stride=1, padding=1),  # img: 3x3
            # self._block(channels_noise, features_g * 16, kernel_size=5, stride=1, padding=1),  # img: 3x3
            self._block(features_g * 16, features_g * 8, (5,8), (1,2), 1),  # img: 5x10
            self._block(features_g * 8, features_g * 4, (6,12), (1,2), 0),  # img: 10x30
            self._block(features_g * 4, 3, 3, 1, 1),  # img: 10x30
            # nn.ConvTranspose2d(
            #     features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            # ),
            # Output: N x channels_img x 10 x 30
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
            
            
         
            
# ##for WGAN. no sigmoid is used.
# class Critic_onlyimage(nn.Module):
#     def __init__(self, mlp_dims, activation, agg_mode, height_size, width_size, num_channel, dropout, norm): ##in_features = 784
#         super().__init__()
#         self.mlp_dims = mlp_dims
#         # self.num_feat = num_feat  ##durability and weight so 2.
#         self.activation = activation
#         self.agg_mode = agg_mode
#         self.height_size = height_size
#         self.width_size = width_size
#         self.num_channel = num_channel
#         self.dropout = dropout ## 0.3
#         self.norm = norm  ##False
        
#         self.linear_image = nn.Linear(self.height_size*self.width_size*self.num_channel, self.mlp_dims[0])
#         self.mlp_image = MLPLayers(layers = self.mlp_dims, activation = self.activation, dropout=self.dropout, norm=self.norm)
#         self.sigmoid = nn.Sigmoid()
       

#     def forward(self, image): #durability, weight):
#         if self.agg_mode == 'sum':
#             image_hidden = self.linear_image(image) ## [B, z_dim] -> [B, mlp_dims[0]]
#             image_output =  self.mlp_image(image_hidden)  #noise this is [B, mlp_dims[0]] -> [B, mlp_dims[-1]]. But it will be [B, z_dim, 1, 1] at training stage.
            


            
#         if self.agg_mode == 'concat':
            
#             image_hidden = self.linear_image(image)
#             image_output =  self.mlp_image(image_hidden)
#             final_output = self.sigmoid(image_output) ##activate this if data is normalized to [0, 1]
        
        
#         return final_output



# class Generator_onlyimage(nn.Module):
#     def __init__(self, mlp_dims, z_dim, activation, agg_mode, output_mode, height_size, width_size, num_channel, dropout, norm):
#         super().__init__()
#         self.mlp_dims = mlp_dims
#         self.agg_mode = agg_mode
#         self.output_mode = output_mode
#         self.z_dim = z_dim
#         # self.num_feat = num_feat
#         self.height_size = height_size
#         self.width_size = width_size
#         self.num_channel = num_channel
#         self.activation = activation
#         self.dropout = dropout ## 0.3
#         self.norm = norm  ##False
#         self.linear_noise = nn.Linear(self.z_dim, self.mlp_dims[0])
#         self.mlp_noise = MLPLayers(layers = self.mlp_dims, activation = self.activation, dropout=self.dropout, norm=self.norm)
        
#         self.linear_concat = nn.Linear(self.mlp_dims[-1] * 3, self.mlp_dims[-1]) ## 3 bcuz noise, durability_weight and jewon three inputs
#         self.linear_final = nn.Linear(self.mlp_dims[-1], self.num_channel * self.height_size * self.width_size) ##back to image size
#         self.tanh = nn.Tanh()
#         self.sigmoid = nn.Sigmoid()


#     def forward(self, noise):  ##image = [B, mlp_dims[0]]  [B, 256]
#         if self.agg_mode == 'sum':
#             noise_hidden = self.linear_noise(noise) ## [B, z_dim] -> [B, mlp_dims[0]]
#             noise_output =  self.mlp_noise(noise_hidden)  #noise this is [B, mlp_dims[0]] -> [B, mlp_dims[-1]]. But it will be [B, z_dim, 1, 1] at training stage.     
            
#             durability_weight_hidden_states = self.linear_durability_weight(durability_weight) ##[B, num_feat] -> [B, mlp_dims[0]]  ##durability_weight has 2 as dimension size
#             durability_weight_output = self.mlp_durability_weight(durability_weight_hidden_states)  ## [B, mlp_dims[0]] ->  [B, mlp_dims[-1]]
            
#             jewon_hidden = jewon.view(-1, self.height_size*self.width_size*self.num_channel) ##flatten
#             jewon_hidden_output = self.linear_jewon(jewon_hidden)
#             jewon_hidden_output = self.mlp_jewon(jewon_hidden_output)
            
#             summed_output = noise_output + durability_weight_output + jewon_hidden_output
            
#             final_output = self.linear_final(summed_output) ##back to image size to produce images
            
#             if self.output_mode == 'none':
#                 final_output = final_output
#             elif self.output_mode == 'sigmoid':
#                 final_output = self.sigmoid(final_output) ##activate this if data is normalized to [0, 1]
#             elif self.output_mode == 'tanh':
#                 final_output = self.tanh(self.final_output) ##activate this if data is normalized to [-1, 1] 
            
#         if self.agg_mode == 'concat':
#             noise_hidden = self.linear_noise(noise)
#             noise_output =  self.mlp_noise(noise_hidden)
            
#             if self.output_mode == 'none':
#                 final_output = final_output
#             elif self.output_mode == 'sigmoid':
#                 final_output = self.sigmoid(final_output) ##activate this if data is normalized to [0, 1]
#             elif self.output_mode == 'tanh':
#                 final_output = self.tanh(noise_output) ##activate this if data is normalized to [-1, 1] 
            
            

#         return final_output






class Discriminator_discrete(nn.Module):
    def __init__(self, num_durability_classes, num_weight_classes): ##in_features = 784
        super().__init__()


        self.num_durability_classes = num_durability_classes
        self.num_weight_classes = num_weight_classes
        
        
       
        self.durability_embed = nn.Sequential(
            # Z latent vector 100
            nn.Embedding(num_durability_classes, 32),
            nn.Linear(32, 64),
            # nn.ReLU(),
            )
        
        self.weight_embed = nn.Sequential(
            # Z latent vector 100
            nn.Embedding(num_weight_classes, 32),
            nn.Linear(32, 64),
            # nn.ReLU(),
            )
        
        self.fixed_pixels = nn.Sequential(
            # Z latent vector 100
            nn.Linear(900, 64),
            # nn.ReLU(),
            )
        
        self.image_linear = nn.Sequential(
            # Z latent vector 100
            nn.Linear(900, 832),
            # nn.ReLU(),
            )
        
        ##https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989 -> about batchnorm1d in mlp
        self.post_mlp = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(num_features=512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),

                nn.Linear(512, 256),
                nn.BatchNorm1d(num_features=256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(num_features=128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                nn.Linear(128, 1),
                # nn.BatchNorm1d(num_features=1),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
                )

        
        self.sigmoid = nn.Sigmoid()
       

    def forward(self, image, durability, weight, jewon): #durability, weight):
        
        
        # if self.agg_mode == 'sum':
        image_hidden = self.image_linear(image) ## [B, z_dim] -> [B, mlp_dims[0]]
        durability_hidden = self.durability_embed(durability)
        weight_hidden = self.weight_embed(weight)
        jewon_hidden = self.fixed_pixels(jewon)
        
        x = torch.cat([durability_hidden, weight_hidden, image_hidden, jewon_hidden], dim =1)
        x = self.post_mlp(x)
        x = self.sigmoid(x)
        
        
        return x



class Generator_discrete(nn.Module):
    def __init__(self, z_dim, num_durability_classes, num_weight_classes):
        super().__init__()
        

        self.durability_embed = nn.Sequential(

            nn.Embedding(num_durability_classes, 32),
            nn.Linear(32, 64),
            # nn.ReLU(),
            )
        
        self.weight_embed = nn.Sequential(

            nn.Embedding(num_weight_classes, 32),
            nn.Linear(32, 64),
            # nn.ReLU(),
            )
        
        self.fixed_pixels_linear = nn.Sequential(
            # Z latent vector 100
            nn.Linear(900, 64),
            # nn.ReLU(),
            )
        
        self.noise_linear = nn.Sequential(
            # Z latent vector 100
            nn.Linear(z_dim, 320),
            # nn.ReLU(),
            )

        self.post_mlp = nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(num_features=512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),

                nn.Linear(512, 784),
                nn.BatchNorm1d(num_features=784),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                nn.Linear(784, 900),
                nn.BatchNorm1d(num_features=900),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                # nn.Tanh()
                )
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, noise, durability, weight, jewon):  ##image = [B, mlp_dims[0]]  [B, 256]
        
        noise_hidden = self.noise_linear(noise)        
        durability_hidden = self.durability_embed(durability)
        weight_hidden = self.weight_embed(weight)
        jewon_hidden = self.fixed_pixels_linear(jewon)
        
        x = torch.cat([noise_hidden, durability_hidden, weight_hidden, jewon_hidden], dim=1)
        x = self.post_mlp(x)
        x = self.tanh(x)
        
        return x









class Generator_discrete_(nn.Module):
    def __init__(self, z_dim, num_durability_classes, num_weight_classes):
        super().__init__()
        

        self.durability_embed = nn.Sequential(

            nn.Embedding(num_durability_classes, 32),
            # nn.Linear(10, 10),
            # nn.ReLU(),
            )
        
        self.weight_embed = nn.Sequential(

            nn.Embedding(num_weight_classes, 32),
            # nn.Linear(10, 10),
            # nn.ReLU(),
            )
        
        self.fixed_pixels_linear = nn.Sequential(
            # Z latent vector 100
            nn.Linear(900, 900),
            # nn.ReLU(),
            )
        
        self.noise_linear = nn.Sequential(
            # Z latent vector 100
            nn.Linear(z_dim, 900),
            # nn.ReLU(),
            )

        self.post_mlp = nn.Sequential(
                nn.Linear(1864, 1024),
                nn.BatchNorm1d(num_features=1024),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),

                nn.Linear(1024, 1024),
                nn.BatchNorm1d(num_features=1024),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                nn.Linear(1024, 900),
                nn.BatchNorm1d(num_features=900),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                # nn.Tanh()
                )
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.final_linear = nn.Linear(900, 900)

    def forward(self, noise, durability, weight, jewon):  ##image = [B, mlp_dims[0]]  [B, 256]
        
        noise_hidden = self.noise_linear(noise)        
        durability_hidden = self.durability_embed(durability)
        weight_hidden = self.weight_embed(weight)
        jewon_hidden = self.fixed_pixels_linear(jewon)
        
        x = torch.cat([noise_hidden, durability_hidden, weight_hidden, jewon_hidden], dim=1)
        x = self.post_mlp(x)
        x = self.final_linear(x)
        x = self.tanh(x)
        
        return x


class Discriminator_discrete_wgan(nn.Module):
    def __init__(self, mode, input_dim, hidden_dim, output_dim=1): ##in_features = 784
        super(Discriminator_discrete_wgan, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mode = mode

        ##https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989 -> about batchnorm1d in mlp
        self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.LayerNorm(self.input_dim, elementwise_affine=True),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),

                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim, elementwise_affine=True),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim, elementwise_affine=True),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim, elementwise_affine=True),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                
                nn.Linear(self.hidden_dim, self.output_dim),
                # nn.BatchNorm1d(num_features=1),
                # nn.LeakyReLU(0.2),
                # nn.Dropout(0.3)
                )

        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
       

    def forward(self, image): #durability, weight):
        

        x = self.mlp(image)
        if self.mode == 'wgan':
            x = x
            return x
        else:
            x = self.sigmoid(x)
            return x




class Generator_discrete_wgan(nn.Module):
    def __init__(self, zdim, input_dim, output_dim, num_durability_classes, num_weight_classes):
        super(Generator_discrete_wgan, self).__init__()
        '''
        zdim: random noise dimension
        input_dim: input data dimension (900 or 420 in this case)
        num_durability_classes: number of durability (내구) data kinds.
        '''
        self.zdim = zdim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_durability_classes = num_durability_classes
        self.num_weight_classes = num_weight_classes
        

        self.durability_embed = nn.Sequential(

            nn.Embedding(self.num_durability_classes, 32),
            # nn.Linear(10, 10),
            # nn.ReLU(),
            )
        
        self.weight_embed = nn.Sequential(

            nn.Embedding(self.num_weight_classes, 32),
            # nn.Linear(10, 10),
            # nn.ReLU(),
            )
        
        self.fixed_pixels_linear = nn.Sequential(
            # Z latent vector 100
            nn.Linear(self.input_dim, self.input_dim),
            # nn.ReLU(),
            )
        
        self.noise_linear = nn.Sequential(
            # Z latent vector 100
            nn.Linear(zdim, self.input_dim),
            # nn.ReLU(),
            )

        self.post_mlp = nn.Sequential(
                nn.Linear(self.input_dim*2 + 32 + 32, 1024), ##1864
                nn.BatchNorm1d(num_features=1024),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),

                nn.Linear(1024, 1024),
                nn.BatchNorm1d(num_features=1024),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                nn.Linear(1024, self.output_dim),
                nn.BatchNorm1d(num_features=self.output_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                # nn.Tanh()
                )
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.final_linear = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, noise, durability, weight, jewon):  ##image = [B, mlp_dims[0]]  [B, 256]
        
        noise_hidden = self.noise_linear(noise)        
        durability_hidden = self.durability_embed(durability)
        weight_hidden = self.weight_embed(weight)
        jewon_hidden = self.fixed_pixels_linear(jewon)
        
        x = torch.cat([noise_hidden, durability_hidden, weight_hidden, jewon_hidden], dim=1)
        x = self.post_mlp(x)
        x = self.final_linear(x)
        x = self.tanh(x)
        
        return x





class Discriminator_discrete_sum(nn.Module):
    def __init__(self, input_dim, num_durability_classes, num_weight_classes, output_dim=1): ##in_features = 784
        super().__init__()

        self.input_dim = input_dim
        self.num_durability_classes = num_durability_classes
        self.num_weight_classes = num_weight_classes
        self.output_dim = output_dim
        
        
       
        self.durability_embed = nn.Sequential(
            # Z latent vector 100
            nn.Embedding(num_durability_classes, 64),
            nn.BatchNorm1d(num_features=64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 512),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, self.input_dim),
            # nn.ReLU(),
            )
        
        self.weight_embed = nn.Sequential(
            # Z latent vector 100
            nn.Embedding(num_weight_classes, 64),
            nn.Linear(64, 128),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 512),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, self.input_dim),
            # nn.ReLU(),
            )
        
        self.fixed_pixels = nn.Sequential(
            # Z latent vector 100
            nn.Linear(self.input_dim, self.input_dim),
            # nn.ReLU(),
            )
        
        self.image_linear = nn.Sequential(
            # Z latent vector 100
            nn.Linear(self.input_dim, self.input_dim),
            # nn.ReLU(),
            )
        
        ##https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989 -> about batchnorm1d in mlp
        self.post_mlp = nn.Sequential(
                nn.Linear(self.input_dim, 512),
                nn.BatchNorm1d(num_features=512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),

                nn.Linear(512, 256),
                nn.BatchNorm1d(num_features=256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(num_features=128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                nn.Linear(128, self.output_dim),
                # nn.BatchNorm1d(num_features=1),
                # nn.LeakyReLU(0.2),
                # nn.Dropout(0.3)
                )

        
        self.sigmoid = nn.Sigmoid()
       

    def forward(self, image, durability, weight, jewon): #durability, weight):
        
        
        # if self.agg_mode == 'sum':
        image_hidden = self.image_linear(image) ## [B, z_dim] -> [B, mlp_dims[0]]
        durability_hidden = self.durability_embed(durability)
        weight_hidden = self.weight_embed(weight)
        jewon_hidden = self.fixed_pixels(jewon)
        
        x = durability_hidden + weight_hidden + image_hidden + jewon_hidden
        x = self.post_mlp(x)
        x = self.sigmoid(x)
        
        
        return x



class Generator_discrete_sum(nn.Module):
    def __init__(self, zdim, input_dim, num_durability_classes, num_weight_classes):
        super().__init__()
        self.zdim = zdim
        self.input_dim = input_dim
        self.num_durability_classes = num_durability_classes
        self.num_weight_classes = num_weight_classes
        
        

        self.durability_embed = nn.Sequential(

            nn.Embedding(self.num_durability_classes, 64),
            nn.Linear(64, 64),
            # nn.ReLU(),
            )
        
        self.weight_embed = nn.Sequential(

            nn.Embedding(self.num_weight_classes, 64),
            nn.Linear(64, 64),
            # nn.ReLU(),
            )
        
        self.fixed_pixels_linear = nn.Sequential(
            # Z latent vector 100
            nn.Linear(self.input_dim, 64),
            # nn.ReLU(),
            )
        
        self.noise_linear = nn.Sequential(
            # Z latent vector 100
            nn.Linear(self.zdim, 64),
            # nn.ReLU(),
            )

        self.post_mlp = nn.Sequential(
                nn.Linear(64, 128),
                nn.BatchNorm1d(num_features=128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),

                nn.Linear(128, 256),
                nn.BatchNorm1d(num_features=256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                nn.Linear(256, 512),
                nn.BatchNorm1d(num_features=512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                nn.Linear(512, 784),
                nn.BatchNorm1d(num_features=784),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                nn.Linear(784, self.input_dim),
                nn.BatchNorm1d(num_features=self.input_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(self.input_dim, self.input_dim)
                
                # nn.Tanh()
                )
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, noise, durability, weight, jewon):  ##image = [B, mlp_dims[0]]  [B, 256]
        
        noise_hidden = self.noise_linear(noise)        
        durability_hidden = self.durability_embed(durability)
        weight_hidden = self.weight_embed(weight)
        jewon_hidden = self.fixed_pixels_linear(jewon)
        
        x = noise_hidden + durability_hidden + weight_hidden + jewon_hidden
        x = self.post_mlp(x)
        x = self.tanh(x)
        
        return x




class SelfAttentionMLPGenerator(nn.Module):
    def __init__(self, zc_dim, input_dim, height, width, output_dim, embed_dim = 64 , num_heads = 4):
        super().__init__() ##in_channel is the dimension of input altogether (z + c1 + c2 etc..)
        
        self.zc_dim = zc_dim
        self.input_dim = input_dim
        self.height = height
        self.width = width
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        
        # self.fixed_pixels_linear = nn.Sequential(
        #     # Z latent vector 100
        #     nn.Linear(out_channel, out_channel),  ##image_dim is same as out_channel anyways.
        #     # nn.ReLU(),
        #     )
        
        self.noise_codes_linear = nn.Linear(self.zc_dim, self.input_dim)
            # nn.Sequential(
            # # Z latent vector 100
            # nn.Linear(zc_dim, img_dim),
            # # nn.ReLU(),
            # )
        self.pre_mlp = PreMLP(input_dim=self.input_dim, height=self.height, width=self.width, embed_dim=self.embed_dim)
        self.selfattention = SelfAttentionBlock(embed_dim=self.embed_dim , num_heads=self.num_heads)
        self.post_mlp = PostMLP(height=self.height, width=self.width, embed_dim=self.embed_dim, output_dim=self.output_dim)
        # self.final_linear = nn.Linear(img_dim, img_dim)
        
        self.tanh = nn.Tanh()

    def forward(self, noise_codes, jewon):
        
        noise_codes_hidden = self.noise_codes_linear(noise_codes)        
        # jewon_hidden = self.fixed_pixels_linear(jewon)
        hidden = self.pre_mlp(noise_codes_hidden)     
        hidden = self.selfattention(hidden)
        hidden = self.post_mlp(hidden)
        img = self.tanh(hidden)
        
        return img


class SelfAttentionMLPDiscriminator(nn.Module):
    def __init__(self, input_dim, height, width, embed_dim, num_heads, mode, output_dim=1):
        super().__init__()
        
        self.mode = mode
        self.input_dim = input_dim
        self.height = height
        self.width = width
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        
        self.pre_mlp = PreMLP(self.input_dim, self.height, self.width, self.embed_dim)
        self.selfattention = SelfAttentionBlock(self.embed_dim, self.num_heads)
        self.post_mlp = PostMLP(self.height, self.width, self.embed_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, img):
        
        x = self.pre_mlp(img)
        x = self.selfattention(x)
        x = self.post_mlp(x)
        if self.mode == 'wgan':
            return x
        else:
            x = self.sigmoid(x)
            return x








class InfoGANMLPGenerator(nn.Module):
    def __init__(self, zc_dim, input_dim, hidden_dim, output_dim):
        super().__init__() ##in_channel is the dimension of input altogether (z + c1 + c2 etc..)
        self.zc_dim = zc_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # self.fixed_pixels_linear = nn.Sequential(
        #     # Z latent vector 100
        #     nn.Linear(out_channel, out_channel),  ##image_dim is same as out_channel anyways.
        #     # nn.ReLU(),
        #     )
        
        self.noise_codes_linear = nn.Sequential(
            # Z latent vector 100
            nn.Linear(self.zc_dim, self.input_dim),
            # nn.ReLU(),
            )

        self.post_mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.BatchNorm1d(num_features=self.hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),

                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(num_features=self.hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                nn.Linear(self.hidden_dim, self.output_dim),
                nn.BatchNorm1d(num_features=self.output_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                # nn.Tanh()
                )
        self.final_linear = nn.Linear(self.output_dim, self.output_dim)
        self.tanh = nn.Tanh()

    def forward(self, noise_codes, jewon):
        
        x = self.noise_codes_linear(noise_codes)        
        # jewon_hidden = self.fixed_pixels_linear(jewon)
        x = self.post_mlp(x)     
        x = self.final_linear(x)
        x = self.tanh(x)
        
        return x




class InfoGANMLPDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=256):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        ##https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989 -> about batchnorm1d in mlp
        self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.LayerNorm(self.input_dim, elementwise_affine=True),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),

                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim, elementwise_affine=True),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                nn.Linear(self.hidden_dim, self.output_dim),
                nn.LayerNorm(self.output_dim, elementwise_affine=True),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                nn.Linear(self.output_dim, self.output_dim),
                )
        self.sigmoid = nn.Sigmoid()

        
        # self.sigmoid = nn.Sigmoid()
        ##https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989 -> about batchnorm1d in mlp


    def forward(self, img):
        x = self.mlp(img)

        return x



class InfoGANSelfAttentionMLPGenerator(nn.Module):
    def __init__(self, zc_dim, input_dim, height, width, output_dim, embed_dim = 64, num_heads = 4):
        super().__init__() ##in_channel is the dimension of input altogether (z + c1 + c2 etc..)
        
        self.input_dim = input_dim
        self.height = height
        self.width = width
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.zc_dim = zc_dim
        
        # self.fixed_pixels_linear = nn.Sequential(
        #     # Z latent vector 100
        #     nn.Linear(out_channel, out_channel),  ##image_dim is same as out_channel anyways.
        #     # nn.ReLU(),
        #     )
        
        self.noise_codes_linear = nn.Linear(self.zc_dim, self.input_dim)
            # nn.Sequential(
            # # Z latent vector 100
            # nn.Linear(zc_dim, img_dim),
            # # nn.ReLU(),
            # )
        self.pre_mlp = PreMLP(self.input_dim, self.height, self.width, self.embed_dim)
        self.selfattention = SelfAttentionBlock(embed_dim=self.embed_dim, num_heads=self.num_heads)
        self.post_mlp = PostMLP(self.height, self.width, self.embed_dim, self.output_dim)
        # self.final_linear = nn.Linear(img_dim, img_dim)

        self.tanh = nn.Tanh()

    def forward(self, noise_codes, jewon):
        x = self.noise_codes_linear(noise_codes)        
        # jewon_hidden = self.fixed_pixels_linear(jewon)
        x = self.pre_mlp(x)     
        x = self.selfattention(x)
        # x = x.view(-1, self.height*self.width)
        x = self.post_mlp(x)
        x = self.tanh(x)
        
        return x








class InfoGANSelfAttentionMLDiscriminator(nn.Module):
    def __init__(self, input_dim, height, width, embed_dim, output_dim, num_heads):
        super(InfoGANSelfAttentionMLDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        self.height = height
        self.width = width
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        self.pre_mlp = PreMLP(self.input_dim, self.height, self.width, self.embed_dim)
        self.selfattention = SelfAttentionBlock(self.embed_dim, self.num_heads)
        self.post_mlp = PostMLP(self.height, self.width, self.embed_dim, self.output_dim)
        

    def forward(self, img):
        
        x = self.pre_mlp(img)
        x = self.selfattention(x)
        x = self.post_mlp(x)

        return x







class InfoGANMLPDHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mode=None):
        super().__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim, elementwise_affine=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.output_dim),
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mlp(x)
        if self.mode == 'wgan':
            return x
        else:
            x = self.sigmoid(x)
            return x
        


class InfoGANMLPQHead_discrete(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class_c1, num_class_c2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_class_c1 = num_class_c1
        self.num_class_c2 = num_class_c2
        
        
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim, elementwise_affine=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            )

        # self.disc = nn.Conv2d(128, 10, 1) ##discrete codes
        
        self.disc1 = nn.Linear(self.hidden_dim, self.num_class_c1) ##discrete codes
        self.disc2 = nn.Linear(self.hidden_dim, self.num_class_c2) ##discrete codes
        
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)
        # self.mu = nn.Linear(out_dim, num_continuous_c)
        # self.var = nn.Linear(out_dim, num_continuous_c)

    def forward(self, x):
        x = self.mlp(x)

        # disc_logits = self.conv_disc(x).squeeze()

        disc_logits1 = self.softmax1(self.disc1(x))
        disc_logits2 = self.softmax2(self.disc2(x))

        return  disc_logits1, disc_logits2 ##disc_logits,




class InfoGANMLPQHead_continuous(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_continuous_c):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_continuous_c = num_continuous_c
        
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim, elementwise_affine=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            )

        # self.disc = nn.Conv2d(128, 10, 1) ##discrete codes
        
        # self.disc = nn.Linear(out_dim, num_continuous_c) ##discrete codes
        # self.disc = nn.Linear(out_dim, num_continuous_c) ##discrete codes
        
        self.mu = nn.Linear(self.output_dim, self.num_continuous_c)
        self.var = nn.Linear(self.output_dim, self.num_continuous_c)

    def forward(self, x):
        x = self.mlp(x)

        # disc_logits = self.conv_disc(x).squeeze()

        mu = self.mu(x)
        var = torch.exp(self.var(x))

        return  mu, var ##disc_logits,







