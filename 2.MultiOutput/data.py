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
# y_rootdir = r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\Second_part\data\result_new"
# x_rootdir = r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\data"
# def HyundayMultioutputData(x_rootdir, y_root_dir):
#     datanames = os.listdir(y_rootdir)

#     y_data = []
#     x_data = []
#     for dataname in datanames:
#         y = pd.read_csv(os.path.join(y_rootdir, dataname))    
#         y_data.append(y.to_numpy())


#         x_image_name = dataname[:-11]  + '.bmp'
#         img_path = os.path.join(x_rootdir, x_image_name)   
#         image = io.imread(img_path) 
#         x_data.append(image)
        

#     x_data = np.stack(x_data)
#     x_data = x_data.reshape(len(x_data), -1)
        
#     y_data = np.stack(y_data)
#     y_data = np.squeeze(y_data, axis=2)
    
#     return x_data, y_data



# dir(train_dataset)
# train_dataset._x[0]
# from matplotlib import pyplot as plt
# from matplotlib import image as mpimg


# plt.imshow(train_dataset._x[1000].reshape(10, 30, 3))
# plt.show()



                
# img_grid_fake = torchvision.utils.make_grid(fake[20:30])#, normalize=True)
# data = os.listdir(r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\data")

# x_data = io.imread(os.path.join(r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\data", data[4646])) 
# plt.imshow(x_data)
# plt.show()
# # zzzz = x_data[:, np.r_[0:6, 10:16, 20:22]]

# x_data.shape

# zzzz = x_data[:, np.r_[0:6, 10:16, 20:22]]
# zzzz.shape
# plt.imshow(zzzz)
# plt.show()






class TorchHyundaiData(torch.utils.data.Dataset):
    def __init__(self, x_rootdir, y_rootdir, transform=None):
        # self.data = pd.read_csv(csv_file)
        self.x_rootdir = x_rootdir
        self.y_rootdir = y_rootdir
        self.transform = transform
        self.data = os.listdir(self.x_rootdir)
        self.target = os.listdir(self.y_rootdir)
        # self.minmax = MinMaxScaler()
        # self.standard = StandardScaler()
        # self.norm_mode = norm_mode

        self.x_data_list = []    
        self.y_data_list = []
                
        for target_dataname in self.target:
            y = pd.read_csv(os.path.join(self.y_rootdir, target_dataname))    
            
            
            
            
            self.y_data_list.append(y.to_numpy())
            
            
            
            # x_image_name = target_dataname[:-11]  + '.bmp'
            x_image_name = target_dataname[7:-4]  + '.bmp'
            img_path = os.path.join(self.x_rootdir, x_image_name)   
            x_data = io.imread(img_path) 
            self.x_data_list.append(x_data)

        self.x = np.stack(self.x_data_list)
        # x = x.reshape(len(x), -1)
            
        self.y = np.stack(self.y_data_list)
        self.y = np.squeeze(self.y, axis=2)
        
        # self.y = self.minmax.fit_transform(self.y)
            
        
        
        
    
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
        # self.minmax = MinMaxScaler()
        # self.standard = StandardScaler()
        # self.norm_mode = norm_mode

        self.x_data_list = []    
        self.y_data_list = []
                
        for target_dataname in self.target:
            y = pd.read_csv(os.path.join(self.y_rootdir, target_dataname))    
            
            
            
            
            self.y_data_list.append(y.to_numpy())
            
            
            
            # x_image_name = target_dataname[:-11]  + '.bmp'
            x_image_name = target_dataname[7:-4]  + '.bmp'
            img_path = os.path.join(self.x_rootdir, x_image_name)   
            x_data = io.imread(img_path) 
            self.x_data_list.append(x_data[:, np.r_[0:6, 10:16, 20:22], :]) ## 공백열을 제거하기 위한 처리과정

        self.x = np.stack(self.x_data_list)
        # x = x.reshape(len(x), -1)
            
        self.y = np.stack(self.y_data_list)
        self.y = np.squeeze(self.y, axis=2)
        
        # self.y = self.minmax.fit_transform(self.y)
            
 
    
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
        # self.data = pd.read_csv(csv_file)
        self.x_rootdir = x_rootdir
        self.transform = transform
        self.data = os.listdir(self.x_rootdir)
        self.x_data_list = []    

                
        for dataname in self.data:
            img_path = os.path.join(self.x_rootdir, dataname)   
            x_data = io.imread(img_path) 
            self.x_data_list.append(x_data)

        self.x = np.stack(self.x_data_list)
        # x = x.reshape(len(x), -1)
            
                  
    def __len__(self):
        return len(self.x_data_list)
    
    def __getitem__(self, index):
        image = self.x[index]


        
        if self.transform:
            transformed_image = self.transform(image)
        
        return transformed_image


# zz = os.listdir(args.img_data_path)
# for dataname in zz:
#     img_path = os.path.join(args.img_data_path, dataname)    
#     x_data = io.imread(img_path) 
#     break


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
        
        
        # self.y = self.minmax.fit_transform(self.y)
            
    def __len__(self):
        return len(self.x_data_list)
    
    def __getitem__(self, index):
        image = self.x[index]


        
        if self.transform:
            transformed_image = self.transform(image)

        
        return transformed_image



# class TorchHyundaiData(Dataset) :
#     def __init__(self, x, y, transform=None) :
#         self.x = np.float32(x)
#         self.y = y
#         self.transform = transform

#     def __len__(self,) :
#         return len(self.y)

#     def __getitem__(self, idx) :
        
#         image = self.x[idx]   
#         target = self.y[idx]
        
#         if self.transform:
#             transformed_image = self.transform(image)
#             transformed_image = transformed_image.reshape(1, -1)

        
#         return transformed_image, target
