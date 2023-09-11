# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:17:33 2023

@author: jake_
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
# import cv2
# import albumentations as A

import numpy as np


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler , MinMaxScaler
import matplotlib.pyplot as plt
from IPython import display
import os







class TorchHyundaiData(torch.utils.data.Dataset):
    def __init__(self, x_rootdir, y_rootdir, transform=None):

        self.x_rootdir = x_rootdir
        self.y_rootdir = y_rootdir
        self.transform = transform
        self.data = os.listdir(self.x_rootdir)
        self.target = os.listdir(self.y_rootdir)


        self.x_data_list = []    
        self.y_data_list = []
                
        for target_dataname in self.target:
            y = pd.read_csv(os.path.join(self.y_rootdir, target_dataname))    
            
            
            
            
            self.y_data_list.append(y.to_numpy())
            

            x_image_name = target_dataname[7:-4]  + '.bmp'
            img_path = os.path.join(self.x_rootdir, x_image_name)   
            x_data = io.imread(img_path) 
            self.x_data_list.append(x_data)

        self.x = np.stack(self.x_data_list)

            
        self.y = np.stack(self.y_data_list)
        self.y = np.squeeze(self.y, axis=2)

            
        
        
        
    
    def __len__(self):
        return len(self.x_data_list)
    
    def __getitem__(self, index):
        image = self.x[index]
        target = self.y[index]  ## [H, W, C]

        
        if self.transform:
            transformed_image = self.transform(image)

        
            
        
        return transformed_image, target
    
    



class TorchDropEmptyHyundaiData(torch.utils.data.Dataset):
    def __init__(self, x_rootdir, y_rootdir, transform=None):
        # self.data = pd.read_csv(csv_file)
        self.x_rootdir = x_rootdir
        self.y_rootdir = y_rootdir
        self.transform = transform
        self.data = os.listdir(self.x_rootdir)
        self.target = os.listdir(self.y_rootdir)


        self.x_data_list = []    
        self.y_data_list = []
                
        for target_dataname in self.target:
            y = pd.read_csv(os.path.join(self.y_rootdir, target_dataname))    
            
            
            
            
            self.y_data_list.append(y.to_numpy())
            
            
            x_image_name = target_dataname[7:-4]  + '.bmp'
            img_path = os.path.join(self.x_rootdir, x_image_name)   
            x_data = io.imread(img_path) 
            self.x_data_list.append(x_data[:, np.r_[0:6, 10:16, 20:22], :]) ## 공백열을 제거하기 위한 처리과정

        self.x = np.stack(self.x_data_list)
            
        self.y = np.stack(self.y_data_list)
        self.y = np.squeeze(self.y, axis=2)

            
 
    
    def __len__(self):
        return len(self.x_data_list)
    
    def __getitem__(self, index):
        image = self.x[index]
        target = self.y[index]  ## [H, W, C]

        
        if self.transform:
            transformed_image = self.transform(image)

        
            
        
        return transformed_image, target




class FormTorchData(torch.utils.data.Dataset):
  def __init__(self, x, y):
    super(FormTorchData, self).__init__()
    # store the raw tensors
    self._x = x
    self._y = y

  def __len__(self):
    # a DataSet must know it size
    return self._x.shape[0]

  def __getitem__(self, index):
    x = self._x[index, :]
    y = self._y[index, :]
    return x, y




class FormTorchData_inference(torch.utils.data.Dataset):
  def __init__(self, x):
    super(FormTorchData, self).__init__()
    # store the raw tensors
    self._x = x


  def __len__(self):
    # a DataSet must know it size
    return self._x.shape[0]

  def __getitem__(self, index):
    x = self._x[index, :]
    return x







class TorchHyundaiData_inference(torch.utils.data.Dataset):
    def __init__(self, x_rootdir, transform=None):
        self.x_rootdir = x_rootdir
        self.transform = transform
        self.data = os.listdir(self.x_rootdir)
        self.x_data_list = []    

                
        for dataname in self.data:
            img_path = os.path.join(self.x_rootdir, dataname)   
            x_data = io.imread(img_path) 
            self.x_data_list.append(x_data)

        self.x = np.stack(self.x_data_list)
           
                  
    def __len__(self):
        return len(self.x_data_list)
    
    def __getitem__(self, index):
        image = self.x[index]


        
        if self.transform:
            transformed_image = self.transform(image)
        
        return transformed_image



class TorchDropEmptyHyundaiData_inference(torch.utils.data.Dataset):
    def __init__(self, x_rootdir, transform=None):
        # self.data = pd.read_csv(csv_file)
        self.x_rootdir = x_rootdir
        self.transform = transform
        self.data = os.listdir(self.x_rootdir)
        self.x_data_list = []    
                
        for dataname in self.data:
            img_path = os.path.join(self.x_rootdir, dataname) 
            x_data = io.imread(img_path) 
            self.x_data_list.append(x_data[:, np.r_[0:6, 10:16, 20:22]]) ## 공백열을 제거하기 위한 처리과정
        
        self.x = np.stack(self.x_data_list)

    def __len__(self):
        return len(self.x_data_list)
    
    def __getitem__(self, index):
        image = self.x[index]


        
        if self.transform:
            transformed_image = self.transform(image)

        
        return transformed_image

