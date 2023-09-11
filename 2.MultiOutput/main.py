# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:28:10 2023

@author: hojun
"""

import sklearn
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.datasets import load_digits
from sklearn.multioutput import RegressorChain, MultiOutputRegressor
from sklearn.linear_model import RidgeCV, Ridge, LassoLars, SGDRegressor, MultiTaskLasso, LinearRegression, ElasticNet, Lasso, RANSACRegressor, TheilSenRegressor, HuberRegressor, PassiveAggressiveRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR   
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage import io

import xgboost as xgb
import lightgbm as lgb

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms



import os
import pandas as pd
from tqdm import tqdm
import json
import argparse
from loggers import log
import time 
import statistics as stat
import joblib 
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
# import cv2
# import albumentations as A
from utils import initialize_weights
from torch_multiout_models import MultOutRegressor, MultOutChainedRegressor, predict, MultOutRegressorSelfAttentionMLP, MultOutChainedSelfAttentionRegressor
from data import TorchHyundaiData, FormTorchData, TorchDropEmptyHyundaiData





parser = argparse.ArgumentParser(description='Hyundai Multi Output Regression Task')
parser.add_argument("--img_data_path", '-idp', default=r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\data", type=str,  help = 'path of directory that contains image data')          # extra value default="decimal"
parser.add_argument("--label_data_path", '-ldp', default=r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\Second_part\results_send", type=str,  help = 'path of directory that contains label data')          # extra value default="decimal"
parser.add_argument("--mode", '-m', default='ml', type=str, help='weather to use classical machine learning (ml) or Deep Learning MLP (dl)')
parser.add_argument("--datashape", '-ds', default='dropempty', type=str, help='weather to use original data or data dropping useless black columns. Choose between original and dropempty.')
parser.add_argument("--train_size", '-ts', default=0.8, type=float, help='Proportion of train data. 0.8 -> 80%')
parser.add_argument("--validation_size", '-vs', default=0.1, type=float, help='Proportion of train data. 0.1 -> 10%')
parser.add_argument("--log_path", '-lp', default=r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\Second_part\logs\lgb_tutorial", type=str, help='Serving as logging file path and model saving path')
parser.add_argument("--x_scale", '-xs', default='one', type=str, help='normalizing input data. Choose between one, minmax, and standard')
parser.add_argument("--y_scale", '-ys', default='minmax', type=str, help='normalizing label data. Choose between minmax, and standard')
parser.add_argument("--model", '-dm', default='lgb', type=str, help='which model to use for the task')

##DL part
parser.add_argument("--batch_size", '-b', default=512, type=int, help='Mini Batch Size')
parser.add_argument("--device", '-d', default='cuda', type=str, help='Which device to use. "cpu" or "cuda"')
parser.add_argument("--dl_lr", '-dlr', default=0.001, type=float, help='Deep learning model optimizer learning rate')
parser.add_argument("--epochs", '-e', default=100, type=int,  help='Number of epochs to iterate over')
parser.add_argument("--image_dim", '-im', default=420, type=int, help='flatten image size. For the specific case, 10 * 30 * 3.')
parser.add_argument("--label_dim", '-ld', default=189, type=int, help='Label vector dimension')
parser.add_argument("--seq_len", '-sql', default=140, type=int, help='sequence length which is height * width here.')
parser.add_argument("--embed_dim", '-ed', default=64, type=int, help='embedding dimension for self-attention layer')
parser.add_argument("--optimizer", '-opt', default='adam', type=str, help='which optimizer to use')
args = parser.parse_args()


def main(args):
    n_cpu = os.cpu_count()
    # set a logger file
    isExist = os.path.exists(f'{args.log_path}')
    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(f'{args.log_path}')
        print("The new directory is created!")
        
    
    logger = log(path=f'{args.log_path}', file='log_summary') ##logging path 설정하여 나중에 파일 저장.
    starttime=time.time()
    
    
    logger.info('Start working on the script...!')
    logger.info('='*64)
    
    logger.info(f'The training information & parameters: {args}') ##logging all argparse information
    logger.info('='*64)
    
    config = {'img_data_path': args.img_data_path, 'label_data_path': args.label_data_path, 'mode' : args.mode, 'datashape' : args.datashape, 'train_size':args.train_size, 
              'validation_size':args.validation_size, 'log_path':args.log_path,'x_scale':args.x_scale, 'y_scale':args.y_scale, 'model':args.model, 'batch_size':args.batch_size,
              'device':args.device, 'dl_lr':args.dl_lr, 'epochs':args.epochs, 'image_dim':args.image_dim, 'label_dim':args.label_dim, 'seq_len':args.seq_len, 'embed_dim':args.embed_dim,
              'optimizer':args.optimizer}
    
    with open(f'{args.log_path}/config.json', 'w') as fp:
        json.dump(config, fp)
        
    logger.info('config.json saved') 

    if args.mode == 'dl':
        
        transformation = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  ##incase I wanna scale image between [0,1] transforms.ToTensor() is suffice. if -1, 1, normalize.
        )
        
        if args.datashape == 'original':
            TorchHyundaiDataset = TorchHyundaiData(args.img_data_path, args.label_data_path, transform=transformation) # 간단 예시
        elif args.datashape == 'dropempty':
            TorchHyundaiDataset = TorchDropEmptyHyundaiData(args.img_data_path, args.label_data_path, transform=transformation) # 간단 예시            
        else:
            raise ValueError('choose datashape between original and dropempty')
    

    
        dataset_size = len(TorchHyundaiDataset)
        train_size = int(dataset_size * args.train_size)
        validation_size = int(dataset_size * args.validation_size)
        test_size = dataset_size - train_size - validation_size
        train_dataset, validation_dataset, test_dataset = random_split(TorchHyundaiDataset, [train_size, validation_size, test_size])


        logger.info(f'Train size: {train_size}')
        logger.info(f'Validation size: {validation_size}')
        logger.info(f'Test size: {test_size}')



        
        if args.y_scale == 'minmax':
            
            y_scaler = MinMaxScaler()
            
            x_train = train_dataset.dataset.x[train_dataset.indices]
            x_train = np.stack([transformation(x).reshape(args.image_dim) for x in x_train])
            y_train = train_dataset.dataset.y[train_dataset.indices]
            y_train = y_scaler.fit_transform(y_train)
            y_train = torch.from_numpy(y_train)
            
            x_valid = validation_dataset.dataset.x[validation_dataset.indices]
            x_valid = np.stack([transformation(x).reshape(args.image_dim) for x in x_valid])
            y_valid = validation_dataset.dataset.y[validation_dataset.indices]
            y_valid = y_scaler.transform(y_valid)
            y_valid = torch.from_numpy(y_valid)
            
            x_test = test_dataset.dataset.x[test_dataset.indices]
            x_test = np.stack([transformation(x).reshape(args.image_dim) for x in x_test])
            y_test = test_dataset.dataset.y[test_dataset.indices]
            y_test = y_scaler.transform(y_test)
            y_test = torch.from_numpy(y_test)
            
            

            pickle.dump(y_scaler, open(f'{args.log_path}/y_scaler.pkl', 'wb'))
        
        elif args.y_scale == 'standard':
            y_scaler = StandardScaler()
            
            
            x_train = train_dataset.dataset.x[train_dataset.indices]
            x_train = np.stack([transformation(x).reshape(args.image_dim) for x in x_train])
            x_train = torch.from_numpy(x_train)
            y_train = train_dataset.dataset.y[train_dataset.indices]
            y_train = y_scaler.fit_transform(y_train)
            y_train = torch.from_numpy(y_train)
            
            x_valid = validation_dataset.dataset.x[validation_dataset.indices]
            x_valid = np.stack([transformation(x).reshape(args.image_dim) for x in x_valid])
            x_valid = torch.from_numpy(x_valid)
            y_valid = validation_dataset.dataset.y[validation_dataset.indices]
            y_valid = y_scaler.transform(y_valid)
            y_valid = torch.from_numpy(y_valid)
            
            x_test = test_dataset.dataset.x[test_dataset.indices]
            x_test = np.stack([transformation(x).reshape(args.image_dim) for x in x_test])
            x_test = torch.from_numpy(x_test)
            y_test = test_dataset.dataset.y[test_dataset.indices]
            y_test = y_scaler.transform(y_test)
            y_test = torch.from_numpy(y_test)
            
            pickle.dump(y_scaler, open(f'{args.log_path}/y_scaler.pkl', 'wb'))

        else:
            x_train = train_dataset.dataset.x[train_dataset.indices]
            x_train = np.stack([transformation(x).reshape(args.image_dim) for x in x_train])
            x_train = torch.from_numpy(x_train)
            y_train = train_dataset.dataset.y[train_dataset.indices]
            # y_train_scaled = scaler.fit_transform(y_train)
            y_train = torch.from_numpy(y_train)
            
            x_valid = validation_dataset.dataset.x[validation_dataset.indices]
            x_valid = np.stack([transformation(x).reshape(args.image_dim) for x in x_valid])
            x_valid = torch.from_numpy(x_valid)
            y_valid = validation_dataset.dataset.y[validation_dataset.indices]
            # y_valid_scaled = scaler.transform(y_valid)
            y_valid = torch.from_numpy(y_valid)
            
            x_test = test_dataset.dataset.x[test_dataset.indices]
            x_test = np.stack([transformation(x).reshape(args.image_dim) for x in x_test])
            x_test = torch.from_numpy(x_test)
            y_test = test_dataset.dataset.y[test_dataset.indices]
            # y_test_scaled = scaler.transform(y_test)
            y_test = torch.from_numpy(y_test)
        
        
        train_dataset = FormTorchData(x_train, y_train)
        validation_dataset = FormTorchData(x_valid, y_valid)
        test_dataset = FormTorchData(x_test, y_test)
        


                
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        
        device = args.device
        
        if args.model == 'mor':
            model = MultOutRegressor(args.image_dim , args.label_dim).to(device)

        
        elif args.model == 'morsa':
            model = MultOutRegressorSelfAttentionMLP(img_dim=args.image_dim, seq_len=args.seq_len, embed_dim=args.embed_dim).to(device)
        
        elif args.model == 'mocr':
            model = MultOutChainedRegressor(args.image_dim , args.label_dim, order=sorted(set(range(0, args.label_dim)))).to(device)

        elif args.model == 'mocrsa':
            model = MultOutChainedSelfAttentionRegressor(args.image_dim, args.label_dim, order=sorted(set(range(0, args.label_dim))), hidden_dim=args.embed_dim, seed = 2023).to(device)
        
        else:
            raise Exception('you must choose model argument between mor and mocr if using deep learning')
        
        logger.info(f'model specification: {model}')
        
        if args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.dl_lr)
        
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.dl_lr)
            
        elif args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.dl_lr)

        criterion = nn.MSELoss()
        
        initialize_weights(model)
        
        logger.info('DNN training start!')     
        for e in tqdm(range(args.epochs)):
            total_train_loss = 0.0
            total_val_loss= 0.0
            total_train_rmse=0.0
            total_val_rmse=0.0
            model.train()

            for data, labels in train_dataloader:
                batch_size = data.shape[0]
                data, labels = data.float().reshape(batch_size, -1).to(device), labels.float().to(device)
                
    
                # Forward Pass
                preds = model(data)
                

                
                # Find the Loss
                loss = criterion(preds, labels)
                train_rmse =  torch.sqrt(loss)
                # Calculate gradients 
                
                loss.backward()
                # Update Weights
                optimizer.step()
                # Clear the gradients
                optimizer.zero_grad()
                # Calculate Loss
                total_train_loss += loss.item() ##  축적해서 loss를 더해 준 후 마지막에 평균을 내어서 epoch 평균 loss를 출력
                total_train_rmse += train_rmse.item()
            
            avg_train_loss = total_train_loss/len(train_dataloader)
            avg_train_rmse = total_train_rmse/len(train_dataloader)
            logger.info(f'\n Epoch {e+1} \t\t Average train loss: {avg_train_loss}')
            logger.info(f'\n Epoch {e+1} \t\t Average train RMSE: {avg_train_rmse}')
            logger.info('='*64)
            
            model.eval()
            with torch.no_grad():
              for val_data, val_labels in validation_dataloader:
                  
                batch_size = val_data.shape[0]
                  
                val_data, val_labels = val_data.float().reshape(batch_size, -1).to(device), val_labels.float().to(device)
                
                val_preds = model(val_data)

                                
                
                val_loss = criterion(val_preds, val_labels)
                val_rmse =  torch.sqrt(val_loss)
                
                total_val_loss += val_loss.item()
                total_val_rmse += val_rmse.item()
            
            avg_valid_loss = total_val_loss/len(validation_dataloader)
            avg_valid_rmse = total_val_rmse/len(validation_dataloader)
            logger.info(f'\n Epoch {e+1} \t\t Average validation loss: {avg_valid_loss}')
            logger.info(f'\n Epoch {e+1} \t\t Average validation RMSE: {avg_valid_rmse}')

            logger.info('='*64)

        
        
        
        model.eval()
        ##making result and prediction by test data
        test_predictions = []
        test_data_list = []
        test_label_list = []
        
        test_predictions_original = []
        test_label_list_original = []
        
        for i, (test_data, test_label) in enumerate(test_dataloader):
            batch_size = test_data.shape[0]
            test_preds = model(test_data.reshape(batch_size, -1).to(device)).to(device)#.to('cpu'))
            
            if args.y_scale == 'standard' or args.y_scale == 'minmax':
                test_preds = y_scaler.inverse_transform(test_preds.detach().cpu())
                test_label = y_scaler.inverse_transform(test_label.detach().cpu())
            else:
                pass
    
            test_predictions.append(torch.tensor(test_preds).detach().cpu())
            test_label_list.append(torch.tensor(test_label).detach().cpu())

            

            
        test_predictions = (torch.cat([tensor for tensor in test_predictions], dim=0)).numpy() ## putting all together into a tensor
        test_label_list = (torch.cat([tensor for tensor in test_label_list], dim=0)).numpy() ## putting all together into a tensor
        

        logger.info('prediction has been made with test dataset')
        logger.info('saving predictions as numpy array...')
        with open(f'{args.log_path}/predictions.npy', 'wb') as f:
            np.save(f, test_predictions)
            
        with open(f"{args.log_path}/predictions.pickle", 'wb') as f:
            pickle.dump(test_predictions, f, protocol=pickle.HIGHEST_PROTOCOL)            
        

        test_predictions_df = pd.DataFrame(test_predictions)
        test_label_list_df = pd.DataFrame(test_label_list)
        
        test_predictions_df.to_csv(f"{args.log_path}/predictions.csv", index=False)
        test_label_list_df.to_csv(f"{args.log_path}/groundtruth.csv", index=False)
    


        model_saving_full_path = f"{args.log_path}/model.pt"


        torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'train_size': args.train_size,
                    'validation_size': args.validation_size,
                    'lr': args.dl_lr,
                    'datashape': args.datashape,
                    'image_dim': args.image_dim,
                    'seq_len': args.seq_len,
                    'embed_dim': args.embed_dim,
                    'label_dim': args.label_dim,
                    }, model_saving_full_path)
        logger.info(f'Trained torch model is saved at {model_saving_full_path}')
    
        logger.info(f'Training took {time.time()-starttime} seconds')
        logger.info('The task is done!') 
    
    

    
        # https://discuss.pytorch.org/t/efficient-method-to-gather-all-predictions/8008
        ##RMSE calculated for each column of label data
        rmse_by_cols = {}
        for i, column in enumerate(range(test_label_list.shape[1])):
            rmse = mean_squared_error(test_label_list[:,column], test_predictions[:,column], squared = False)
            rmse_by_cols[i] = rmse
                     
        mean_by_cols = {column : stat.mean(abs(test_label_list[:,column] - test_predictions[:,column])) for column in range(test_label_list.shape[1])}
        stdev_by_cols = {column : stat.stdev(abs(test_label_list[:,column] - test_predictions[:,column])) for column in range(test_label_list.shape[1])}
       
        ##실제값과 예측값이 제일 작은 것을 컬럼마다
        min_by_cols = {column : np.min(abs(test_label_list[:,column] - test_predictions[:,column])) for column in range(test_label_list.shape[1])}
        
        ##실제값과 예측값이 제일 큰 것을 컬럼마다
        max_by_cols = {column : np.max(abs(test_label_list[:,column] - test_predictions[:,column])) for column in range(test_label_list.shape[1])}
        
        error_percentage_by_cols = {column :  np.sqrt(np.mean(np.square(( abs((test_label_list[:,column]  - test_predictions[:,column] + 1e-15)  / (test_label_list[:,column] + 1e-15)))), axis=0)) for column in range(test_label_list.shape[1])}

        with open(f'{args.log_path}/rmse_by_cols.json', 'w') as fp:
            json.dump(rmse_by_cols, fp)
        logger.info('rmse_by_cols.json file is saved') 

        with open(f'{args.log_path}/mean_by_cols.json', 'w') as fp:
            json.dump(mean_by_cols, fp)
        logger.info('mean_by_cols.json file is saved') 
            
        with open(f'{args.log_path}/stdev_by_cols.json', 'w') as fp:
            json.dump(stdev_by_cols, fp)
        logger.info('stdev_by_cols.json file is saved') 
            
        with open(f'{args.log_path}/error_percentage_by_cols.json', 'w') as fp:
            json.dump(error_percentage_by_cols, fp) 
        logger.info('error_percentage_by_cols.json file is saved') 
        
        with open(f'{args.log_path}/min_by_cols.json', 'w') as fp:
            json.dump(stdev_by_cols, fp)
        logger.info('min_by_cols.json file is saved') 
        
        with open(f'{args.log_path}/max_by_cols.json', 'w') as fp:
            json.dump(stdev_by_cols, fp)
        logger.info('max_by_cols.json file is saved') 

    
        logger.info(f'The whole process took {time.time()-starttime} seconds!') 
    
    elif args.mode == 'ml':
    
        
        y_rootdir = args.label_data_path
        x_rootdir = args.img_data_path 
        datanames = os.listdir(y_rootdir)
        
        
        logger.info('Data is being processed....')
        
        
        if args.x_scale == 'one':
            
            transformation = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  ##incase I wanna scale image between [0,1] transforms.ToTensor() is suffice. if -1, 1, normalize.
            )
            
            
            if args.datashape == 'original':
                TorchHyundaiDataset = TorchHyundaiData(args.img_data_path, args.label_data_path, transform=transformation) # 간단 예시
            elif args.datashape == 'dropempty':
                TorchHyundaiDataset = TorchDropEmptyHyundaiData(args.img_data_path, args.label_data_path, transform=transformation) # 간단 예시            
            else:
                raise ValueError('choose datashape between original and dropempty')
                
            dataset_size = len(TorchHyundaiDataset)
            train_size = int(dataset_size * args.train_size)
            validation_size = int(dataset_size * args.validation_size)
            test_size = dataset_size - train_size - validation_size
            train_dataset, validation_dataset, test_dataset = random_split(TorchHyundaiDataset, [train_size, validation_size, test_size])
    
            # https://stackoverflow.com/questions/75948550/how-to-convert-a-pytorch-dataset-object-to-a-pandas-dataframe
            x_train = train_dataset.dataset.x[train_dataset.indices]
            x_train = np.stack([transformation(x).reshape(args.image_dim) for x in x_train])
            y_train = train_dataset.dataset.y[train_dataset.indices]
            
            x_valid = validation_dataset.dataset.x[validation_dataset.indices]
            x_valid = np.stack([transformation(x).reshape(args.image_dim) for x in x_valid])
            y_valid = validation_dataset.dataset.y[validation_dataset.indices]
            
            x_test = test_dataset.dataset.x[test_dataset.indices]
            x_test = np.stack([transformation(x).reshape(args.image_dim) for x in x_test])
            y_test = test_dataset.dataset.y[test_dataset.indices]
       
        elif args.x_scale == 'minmax' or  args.x_scale == 'standard':
            y_data = []
            x_data = []
            for dataname in datanames:
                data = pd.read_csv(f'{y_rootdir}/{dataname}')    
                y_data.append(data.to_numpy())
    
                x_image_name = dataname[7:-4]  + '.bmp'
                img_path = os.path.join(x_rootdir, x_image_name)   
                image = io.imread(img_path) 
                if args.datashape == 'original':
                    x_data.append(image)
                elif args.datashape == 'dropempty':
                    x_data.append(image[:, np.r_[0:6, 10:16, 20:22]])
                else:
                    raise ValueError('error!')
            
            
            x_data = np.stack(x_data)
            x_data = x_data.reshape(len(x_data), -1)

            y_data = np.stack(y_data)
            y_data = np.squeeze(y_data, axis=2)
            
            x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=args.validation_size, shuffle=True, random_state=2023)
            
            if args.x_scale == 'standard':
                x_scaler = StandardScaler()
            elif args.x_scale == 'minmax':
                x_scaler = MinMaxScaler()
                
            x_train = x_scaler.fit_transform(x_train)
            x_valid = x_scaler.transform(x_valid)
            pickle.dump(x_scaler, open(f'{args.log_path}/x_scaler.pkl', 'wb'))
                    
        else: 
            raise ValueError('You must choose among one, standard, and minmax')
            # pass ##이 경우 전혀 스케일링이나 정규화되지 않은 데이터가 들어가게 된다.
    
        if args.y_scale == 'standard' or args.y_scale == 'minmax':
            if args.y_scale == 'minmax':
                y_scaler = MinMaxScaler()
            elif args.y_scale == 'standard':    
                y_scaler = StandardScaler()
            
            y_train = y_scaler.fit_transform(y_train)
            y_valid = y_scaler.transform(y_valid)
            pickle.dump(y_scaler, open(f'{args.log_path}/y_scaler.pkl', 'wb'))
            
        else: 
            pass 
        



        logger.info('Data is split to train, valid, and test data')
        logger.info(f'Train size: {len(x_train)}')
        logger.info(f'Validation size: {len(x_valid)}')
        logger.info('Data preparation is done!')
        ##Model selection
        
        ##multi-output-able models by nature
        if args.model == 'linear':
            model = LinearRegression()
        elif args.model == 'KNeiReg':
            model = KNeighborsRegressor(random_state=2023)
        elif args.model == 'RCV':
            model = RidgeCV()    
        elif args.model == 'Ridge':
            model = Ridge(random_state=2023)        
        elif args.model == 'lgb':
            model = MultiOutputRegressor(lgb.LGBMRegressor(random_state=2023, n_jobs=n_cpu-2, importance_type='split', learning_rate=0.09145898164917113,
                                         max_depth=6, n_estimators=368,  num_leaves=33, reg_lambda=0.3824778839846842) )  ## , device = 'gpu'
        elif args.model == 'xgb':
            model = MultiOutputRegressor(xgb.XGBRegressor(random_state=2023, tree_method='gpu_hist', gpu_id=0, n_jobs=n_cpu-2)) ##
        elif args.model == 'Lasso': ##Careful. Slow!  ## You might want to increase the number of iterations
            model = Lasso(random_state=2023)
        elif args.model == 'ElasticNet':  ##Careful. Slow!
            model = ElasticNet(random_state=2023)
        elif args.model == 'RanForestReg':   ## very slow 
            model = RandomForestRegressor(random_state=2023)
        elif args.model == 'DecisionTree':
            model = DecisionTreeRegressor(random_state=2023)
        elif args.model == 'RANSACReg':  ##ValueError: RANSAC could not find a valid consensus set. All `max_trials` iterations were skipped because each randomly chosen sub-sample failed the passing criteria. See estimator attributes for diagnostics (n_skips*).
            model = RANSACRegressor(random_state=2023)
            
        ##Multioutput support for single output models    
        elif args.model == 'HuberReg':  ## 2000 seconds. not too bad permormance 
            model = MultiOutputRegressor(HuberRegressor(random_state=2023))
        elif args.model == 'GBReg':  ## 1800 seconds not good performance 
            model = MultiOutputRegressor(GradientBoostingRegressor(random_state=2023)) ##very slow!!
        elif args.model == 'ABReg': ## 1778 seconds not good 
            model = MultiOutputRegressor(AdaBoostRegressor(random_state=2023))
        elif args.model == 'SVR':  ## it takes too much time. ##increase the number of iterations.  Liblinear failed to converge, increase the number of iterations.
            model = MultiOutputRegressor(LinearSVR(random_state=2023))            
        elif args.model == 'SGDReg':
            model = MultiOutputRegressor(SGDRegressor(random_state=2023))          
        elif args.model == 'PAReg': ##fast 369 seconds and good performance 
            model = MultiOutputRegressor(PassiveAggressiveRegressor(max_iter=100, random_state=2023, tol=1e-3))      
        elif args.model == 'TSReg': ##takes so long time 
            model = MultiOutputRegressor(TheilSenRegressor(random_state=2023))
        
        ##Chained regressor variants
        elif args.model == 'HuberRegChained':
            model = RegressorChain(HuberRegressor(random_state=2023))
        elif args.model == 'GBRegChained':
            model = RegressorChain(GradientBoostingRegressor(random_state=2023))
        elif args.model == 'ABRegChained':
            model = RegressorChain(AdaBoostRegressor(random_state=2023))
        elif args.model == 'SVRChained':
            model = RegressorChain(LinearSVR(random_state=2023))            
        elif args.model == 'SGDRegChained':
            model = RegressorChain(SGDRegressor(random_state=2023))          
        elif args.model == 'PARegChained':
            model = RegressorChain(PassiveAggressiveRegressor(random_state=2023))      
        elif args.model == 'TSRegChained':
            model = RegressorChain(TheilSenRegressor(random_state=2023))  
        else:
            raise ValueError('correct model name must be inserted')
        
        logger.info(f'{model} is chosen as the model')    
        
        
        logger.info(f'Fitting data to the model...')   
        ##train model
        model.fit(x_train, y_train)
        
        
        logger.info(f'Predicting values with validation dataset...')
        ##predict outputs
        y_pred = model.predict(x_valid)
        
                
        
        logger.info(f'Saving the predictions...')

                    
    
        mse = mean_squared_error(y_valid, y_pred)
        logger.info(f'MSE value: {mse}')
        rmse = mean_squared_error(y_valid, y_pred, squared = False)
        logger.info(f'RMSE value: {rmse}')
        ##save the fitted model  ##https://scikit-learn.org/stable/model_persistence.html
        joblib.dump(model, f"{args.log_path}/model.joblib") 
        logger.info(f'The model is saved at {args.log_path}/model.joblib')


        if args.y_scale == 'minmax' or args.y_scale == 'standard':
            y_pred = y_scaler.inverse_transform(y_pred)
            y_valid = y_scaler.inverse_transform(y_valid)
        
        else:
            pass


        rmse_by_cols = {}
        for i, column in enumerate(range(y_valid.shape[1])):
            rmse_col = mean_squared_error(y_valid[:,column], y_pred[:,column], squared = False)
            rmse_by_cols[i] = rmse_col
                     
        mean_by_cols = {column : stat.mean(abs(y_valid[:,column] - y_pred[:,column])) for column in range(y_valid.shape[1])}
        stdev_by_cols = {column : stat.stdev(abs(y_valid[:,column] - y_pred[:,column])) for column in range(y_valid.shape[1])}
        
        ##실제값과 예측값이 제일 작은 것을 컬럼마다
        min_by_cols = {column : np.min(abs(y_valid[:,column] - y_pred[:,column])) for column in range(y_valid.shape[1])}
        
        ##실제값과 예측값이 제일 큰 것을 컬럼마다
        max_by_cols = {column : np.max(abs(y_valid[:,column] - y_pred[:,column])) for column in range(y_valid.shape[1])}
        
        
        
        error_percentage_by_cols = {column :  np.sqrt(np.mean(np.square(( abs((y_valid[:,column] - y_pred[:,column] + + 1e-15) / (y_valid[:,column] + + 1e-15)))), axis=0)) for column in range(y_valid.shape[1])}

        
        with open(f'{args.log_path}/rmse_by_cols.json', 'w') as fp:
            json.dump(rmse_by_cols, fp)
        logger.info('rmse_by_cols.json file is saved') 
        
        with open(f'{args.log_path}/mean_by_cols.json', 'w') as fp:
            json.dump(mean_by_cols, fp)
        logger.info('mean_by_cols.json file is saved') 
            
        with open(f'{args.log_path}/stdev_by_cols.json', 'w') as fp:
            json.dump(stdev_by_cols, fp)
        logger.info('stdev_by_cols.json file is saved') 
            
        with open(f'{args.log_path}/error_percentage_by_cols.json', 'w') as fp:
            json.dump(error_percentage_by_cols, fp)
            
        logger.info('error_percentage_by_cols.json file is saved') 

        with open(f'{args.log_path}/min_by_cols.json', 'w') as fp:
            json.dump(stdev_by_cols, fp)
        logger.info('min_by_cols.json file is saved') 
        
        with open(f'{args.log_path}/max_by_cols.json', 'w') as fp:
            json.dump(stdev_by_cols, fp)
        logger.info('max_by_cols.json file is saved') 

        
        with open(f"{args.log_path}/predictions.pickle", 'wb') as f:
            pickle.dump(y_pred, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        ##예측 결과에 대응하는 정답 결과물 저장
        with open(f"{args.log_path}/groundtruth.pickle", 'wb') as f:
            pickle.dump(y_valid, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        y_pred_df = pd.DataFrame(y_pred)
            
        y_valid_df = pd.DataFrame(y_valid)
        y_pred_df.to_csv(f"{args.log_path}/predictions.csv", index=False)
        y_valid_df.to_csv(f"{args.log_path}/groundtruth.csv", index=False)
            
            
            

        logger.info(f'The whole process took {time.time()-starttime} seconds!') 
        
    else:
        raise Exception('you must choose mode argument between dl and ml')
        
        
if __name__ == "__main__":
    main(args)
