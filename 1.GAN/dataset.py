#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 21:25:36 2022

@author: hojun
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler






class HyundaiDataset_discrete(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = os.listdir(root_dir)
        self.durability_list = []
        self.weight_list = []
        self.image_list = []
        self.fixed_pixels_list = []
        self.fixed_pixels_mask_list = []
        self.durability_emb_dict = {}
        self.weight_emb_dict = {}
        self.durability_embedding = []
        self.weight_embedding = []    
        self.imagename_list = []
                
        for datapoint in self.data:
            
            durability = float(datapoint.split('_')[0])  ##성능/중량  durability 내구성능
            weight = float(datapoint.split('_')[-2].strip('.bmp')) ##중량
            jewon = datapoint.split('_')[1:-1] ##제원 xyz before normalization
            
            img_name = datapoint  ##gangsung (strongness)
            self.imagename_list.append(img_name)
            
            img_path = os.path.join(self.root_dir, img_name)   
            image = io.imread(img_path)  ##PIL에서는 H,W,C 로 들어오지만 나중에 torch로 transform이되면 C,H,W가 된다

            mask = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8)  ## H, W, C
            mask[[7, 5], [11, 14], :] = 1 

            fixed_pixels = (image * mask) #+ gaussian_iid_noise
        

            self.image_list.append(image)
            self.durability_list.append(durability)
            self.weight_list.append(weight)
            self.fixed_pixels_list.append(fixed_pixels)
            self.fixed_pixels_mask_list.append(mask)  ##fixed_pixels [C, H, W]
    
        self.durability_emb_dict = {1.:0, 1.1:1, 1.2:2, 1.3:3, 1.4:4, 1.5:5, 1.6:6, 1.7:7, 1.8:8, 1.9:9}
        self.weight_emb_dict = {2.13:0, 2.14:1, 2.15:2, 2.16:3, 2.17:4, 2.18:5, 2.19:6, 2.2:7, 2.21:8, 2.22:9, 2.23:10, 2.24:11}
        
        self.durability_embedding = [self.durability_emb_dict[durability] for durability in self.durability_list]
        self.weight_embedding = [self.weight_emb_dict[weight] for weight in self.weight_list]   
        
        
        
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        imagename = self.imagename_list[index]
        image = self.image_list[index]  ## [H, W, C]
        fixed_pixels = self.fixed_pixels_list[index]  ## image * mask already done
        mask = self.fixed_pixels_mask_list[index]  ## only masking images
        mask = torch.tensor(mask, dtype=torch.float32).permute(2,0,1)
        durability = self.durability_embedding[index]
        weight= self.weight_embedding[index]
        
        
        
        if self.transform:
            transformed_image = self.transform(image)
            transformed_fixed_pixels = self.transform(fixed_pixels)
        
            
        
        return transformed_image, durability, weight, transformed_fixed_pixels, mask, imagename
    

 

class DropEmptyHyundaiDataset_discrete(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = os.listdir(root_dir)
        self.durability_list = []
        self.weight_list = []
        self.image_list = []
        self.fixed_pixels_list = []
        self.fixed_pixels_mask_list = []
        self.durability_emb_dict = {}
        self.weight_emb_dict = {}
        self.durability_embedding = []
        self.weight_embedding = []    
        self.imagename_list = []
                
        for datapoint in self.data:
            
            durability = float(datapoint.split('_')[0])  ##성능/중량  durability 내구성능
            weight = float(datapoint.split('_')[-2].strip('.bmp')) ##중량
            jewon = datapoint.split('_')[1:-1] ##제원 xyz before normalization
            
            img_name = datapoint  ##gangsung (strongness)
            self.imagename_list.append(img_name)
            
            img_path = os.path.join(self.root_dir, img_name)   
            image = io.imread(img_path)  ##PIL에서는 H,W,C 로 들어오지만 나중에 torch로 transform이되면 C,H,W가 된다
            
            mask = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8)  ## H, W, C
            mask[[7, 5], [11, 14], :] = 1 
            
            image = image[:, np.r_[0:6, 10:16, 20:22], :]
            mask = mask[:, np.r_[0:6, 10:16, 20:22], :]
            
            fixed_pixels = (image * mask) #+ gaussian_iid_noise
        

            self.image_list.append(image)
            self.durability_list.append(durability)
            self.weight_list.append(weight)
            self.fixed_pixels_list.append(fixed_pixels)
            self.fixed_pixels_mask_list.append(mask)  ##fixed_pixels [C, H, W]
    
        self.durability_emb_dict = {1.:0, 1.1:1, 1.2:2, 1.3:3, 1.4:4, 1.5:5, 1.6:6, 1.7:7, 1.8:8, 1.9:9}
        self.weight_emb_dict = {2.13:0, 2.14:1, 2.15:2, 2.16:3, 2.17:4, 2.18:5, 2.19:6, 2.2:7, 2.21:8, 2.22:9, 2.23:10, 2.24:11}
        
        self.durability_embedding = [self.durability_emb_dict[durability] for durability in self.durability_list]
        self.weight_embedding = [self.weight_emb_dict[weight] for weight in self.weight_list]   
        

    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        imagename = self.imagename_list[index]
        image = self.image_list[index]  ## [H, W, C]
        fixed_pixels = self.fixed_pixels_list[index]  ## image * mask already done
        mask = self.fixed_pixels_mask_list[index]  ## only masking images
        mask = torch.tensor(mask, dtype=torch.float32).permute(2,0,1)
        durability = self.durability_embedding[index]
        weight= self.weight_embedding[index]
        
  
        if self.transform:
            transformed_image = self.transform(image)
            transformed_fixed_pixels = self.transform(fixed_pixels)
        
            
        
        return transformed_image, durability, weight, transformed_fixed_pixels, mask, imagename





class HyundaiDataset_InfoGan(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = os.listdir(root_dir)
        self.durability_weight_list = []
        self.image_list = []
        self.fixed_pixels_list = []
        self.fixed_pixels_mask_list = []
        self.imagename_list = []
        self.durability_list = []
        self.weight_list = []                
                
        for datapoint in self.data:
            durability = float(datapoint.split('_')[0])  ##성능/중량  durability 내구성능  ##Durability has 10 categories
            weight = float(datapoint.split('_')[-2].strip('.bmp')) ##중량  ##Weight has 12 categories

            self.durability_list.append(durability)
            self.weight_list.append(weight)

            jewon = datapoint.split('_')[1:-1] ##제원 xyz before normalization
            
            img_name = datapoint  ##gangsung (strongness)
            self.imagename_list.append(img_name)
            
            img_path = os.path.join(root_dir, img_name)   
            image = io.imread(img_path)  ##PIL에서는 H,W,C 로 들어오지만 나중에 torch로 transform이되면 C,H,W가 된다

            mask = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8)  ## H, W, C
            mask[[7, 5], [11, 14], :] = 1 

            fixed_pixels = (image * mask) #+ gaussian_iid_noise
        
            
            self.image_list.append(image)
            self.fixed_pixels_list.append(fixed_pixels)
            self.fixed_pixels_mask_list.append(mask)  ##fixed_pixels [C, H, W]
    

        
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        imagename = self.imagename_list[index]
        image = self.image_list[index]  ## [H, W, C]
        fixed_pixels = self.fixed_pixels_list[index]  ## image * mask already done
        mask = self.fixed_pixels_mask_list[index]  ## only masking images
        mask = torch.tensor(mask, dtype=torch.float32).permute(2,0,1)
        # durability_weight = self.durability_weight_list[index] ##this is for the latent code C in InfoGan
        durability = self.durability_list[index] ##this is for the latent code C in InfoGan
        weight = self.weight_list[index] ##this is for the latent code C in InfoGan


        if self.transform:
            transformed_image = self.transform(image)
            transformed_fixed_pixels = self.transform(fixed_pixels)
        
            
        return transformed_image, durability, weight, transformed_fixed_pixels, mask, imagename
    



