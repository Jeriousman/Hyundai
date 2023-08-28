# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:47:05 2023

@author: jake_
"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from IPython import display
from tqdm import  tqdm

def update(input , target , model, criterion , optimizer, max_norm=5, device='cpu') :
    optimizer.zero_grad()
    output = model(input).to(device)
    # loss = torch.sqrt(criterion(output , target.float())) ##torch.sqrt makes MSE to RMSE ##https://discuss.pytorch.org/t/rmse-loss-function/16540
    loss = criterion(output , target.float()) ##MSE
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    return loss 



def one_epoch(dataloader , model, criterion , optimizer, device='cpu' ) :
    result = torch.FloatTensor([0])
    for idx , (input , target) in enumerate(dataloader) :
        loss = update(input.to(device) , target , model.to(device), criterion.to(device) , optimizer)
        result = torch.add(result , loss)
    else :
        result /= idx+1
        return result.detach().cpu().numpy()

def visualize(result) :
    display.clear_output(wait=True)
    plt.plot(result)
    plt.show()

def train(n_epochs , train_dataloader , valid_dataloader, model, criterion , optimizer , log_interval=10, device='cpu') :
    train_loss = 0.0
    last_loss = 0.0
    train_rmse = 0.0
    valid_rmse = 0.0
    for epoch in tqdm(range(n_epochs)) :
        train_loss = one_epoch(train_dataloader , model.to(device), criterion.to(device) , optimizer)
        if epoch > 0 :
            
            # train_loss += loss.item()
            # running_loss += loss.item()
            # train_loss = train_loss / len(train_dataloader)
            
            # train_loss.append(loss)
            # train_rmse.append(np.sqrt(loss))
            if epoch % log_interval == 0 :
                print(f'Train loss at epoch {epoch} : {train_loss}')
                print(f'Train RMSE at epoch {epoch} : {np.sqrt(train_loss)}')
                visualize(train_loss)

    
    
    with torch.no_grad():
        model.eval()
        val_loss = one_epoch(valid_dataloader , model.to(device), criterion.to(device) , optimizer)
        # valid_rmse.append(np.sqrt(val_loss))
            
        if epoch % log_interval == 0 :
            # visualize(train_loss)
            print(f'Valid loss at epoch {epoch} : {val_loss}')
            
    train_loss = 0.0
    last_loss = 0.0
    train_rmse = 0.0
    valid_rmse = 0.0
        
    
    
    return 


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.BatchNorm1d, nn.InstanceNorm2d, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

            