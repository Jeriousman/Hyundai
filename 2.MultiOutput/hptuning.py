# -*- coding: utf-8 -*-
"""
Created on Sun May 14 10:31:07 2023

@author: hojun
"""


import os
import json
import math
import numpy as np
import argparse
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from tqdm import tqdm
import time 

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
import hyperopt
from hyperopt import tpe
from hyperopt import hp, fmin, tpe,Trials,STATUS_OK
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from ray import air

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from data import TorchHyundaiData, FormTorchData, TorchDropEmptyHyundaiData
from torch_multiout_models import MultOutRegressor, MultOutChainedRegressor, predict, MultOutRegressorSelfAttentionMLP
from loggers import log






parser = argparse.ArgumentParser(description='Hyundai Multi Output Regression Task')
parser.add_argument("--img_data_path", '-idp', default=r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\data", type=str,  help = 'path of directory that contains image data')       
parser.add_argument("--label_data_path", '-ldp', default=r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\Second_part\results_send", type=str,  help = 'path of directory that contains label data')         
parser.add_argument("--mode", '-m', default='ml', type=str, help='weather to use classical machine learning (ml) or Deep Learning MLP (dl)')
parser.add_argument("--datashape", '-ds', default='dropempty', type=str, help='weather to use original data or data dropping useless black columns. Choose between original and dropempty.')
parser.add_argument("--train_size", '-ts', default=0.8, type=float, help='Proportion of train data. 0.8 -> 80%')
parser.add_argument("--validation_size", '-vs', default=0.1, type=float, help='Proportion of train data. 0.1 -> 10%')
parser.add_argument("--log_path", '-op', default=r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\Second_part\hptuning\output", type=str, help='output of hyper-parameter tuning will be saved in the path')
parser.add_argument("--filename", '-fn', default="PAReg_test", type=str, help='model name that will be saved inside log_path') 
parser.add_argument("--model", '-md', default='lgb', type=str, help='model name of model that will be used for hyper-parameter tuning')
parser.add_argument("--image_dim", '-im', default=420, type=int, help='flatten image size. For the specific case, 10 * 30 * 3.')
parser.add_argument("--label_dim", '-ld', default=189, type=int, help='Label vector dimension')
parser.add_argument("--x_scale", '-xs', default='one', type=str, help='which scaler to be used')
parser.add_argument("--y_scale", '-ys', default='minmax', type=str, help='which normalizing to be used')
parser.add_argument("--max_evals", '-me', default=2, type=int, help='maximum number of hyper-parameter evaluation trials')

##DL part
parser.add_argument("--num_samples", '-ns', default=1, type=int,  help='how many tirlas will be sampled')
parser.add_argument("--gpus_per_trial", '-gpt', default=0, type=int,  help='how many gpu to use per trial. it can be 0.25, 1, anything you want. if 0.25, you only allocate 1/4 of gpu per trial but you can do 4 trials if u have 1 gpu')
parser.add_argument("--cpus_per_trial", '-cpt', default=16, type=int,  help='how many cpus to use per trial. You have to check how many cpus you got. This is only for pytorch deeplearning model HG with Ray Tune')
args = parser.parse_args()



n_cpu = os.cpu_count()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj) 



# set a logger file
isExist = os.path.exists(f'{args.log_path}/{args.filename}')
if not isExist:

    # Create a new directory because it does not exist
    os.makedirs(f'{args.log_path}/{args.filename}')
    print("The new directory is created!")
    
if args.mode == 'dl':
    
    
    def load_data(args):
        transformation = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  ##incase I wanna scale image between [0,1] transforms.ToTensor() is suffice. if -1, 1, normalize.
        )
    
        if args.datashape == 'original':
            TorchHyundaiDataset = TorchHyundaiData(args.img_data_path, args.label_data_path, transform=transformation) # 간단 예시
        elif args.datashape == 'dropempty':
            TorchHyundaiDataset = TorchDropEmptyHyundaiData(args.img_data_path, args.label_data_path, transform=transformation) # 간단 예시            
        else:
            raise ValueError('choose datashape between original and dropempty')
        return TorchHyundaiDataset
    
        
        
    def train_multioutput(config):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.model == 'mor':
            model = MultOutRegressor(args.image_dim , args.label_dim, p=config['p']).to(device)
        elif args.model == 'mosa':
            model = MultOutRegressorSelfAttentionMLP(img_dim=420, seq_len=140, embed_dim=64).to(device)
        elif args.model == 'mocr':
        
            model = MultOutChainedRegressor(args.image_dim , args.label_dim, order=sorted(set(range(0, args.label_dim)))).to(device)
        else:
            raise Exception('you must choose model argument between mor and mocr')
            
            
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            print('GPU activated')
            if torch.cuda.device_count() > 1:
                print('DataParallel activated')
                model = nn.DataParallel(model)
        model.to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
        criterion = nn.MSELoss().to(device)
        
        loaded_checkpoint = session.get_checkpoint()
        if loaded_checkpoint:
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
               model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
        
        TorchHyundaiDataset = load_data(args)
        
        dataset_size = len(TorchHyundaiDataset)
        train_size = int(dataset_size * 0.8)
        validation_size = int(dataset_size * 0.1)
        test_size = dataset_size - train_size - validation_size
        generator_train = torch.Generator().manual_seed(2023)
        train_dataset, validation_dataset, test_dataset = random_split(TorchHyundaiDataset, [train_size, validation_size, test_size],
                                                                       generator=generator_train)
        

        transformation = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  ##incase I wanna scale image between [0,1] transforms.ToTensor() is suffice. if -1, 1, normalize.
        )

        
        if args.y_scale == 'minmax' or args.y_scale == 'standard':
            if args.y_scale == 'minmax':
                y_scaler = MinMaxScaler()
            elif args.y_scale == 'standard':
                y_scaler = StandardScaler()
            
            
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
            

        else:
            x_train = train_dataset.dataset.x[train_dataset.indices]
            x_train = np.stack([transformation(x).reshape(args.image_dim) for x in x_train])
            x_train = torch.from_numpy(x_train)
            y_train = train_dataset.dataset.y[train_dataset.indices]
            y_train = torch.from_numpy(y_train)
            
            x_valid = validation_dataset.dataset.x[validation_dataset.indices]
            x_valid = np.stack([transformation(x).reshape(args.image_dim) for x in x_valid])
            x_valid = torch.from_numpy(x_valid)
            y_valid = validation_dataset.dataset.y[validation_dataset.indices]
            y_valid = torch.from_numpy(y_valid)
            
            x_test = test_dataset.dataset.x[test_dataset.indices]
            x_test = np.stack([transformation(x).reshape(args.image_dim) for x in x_test])
            x_test = torch.from_numpy(x_test)
            y_test = test_dataset.dataset.y[test_dataset.indices]
            y_test = torch.from_numpy(y_test)
                        
        
        
        train_dataset = FormTorchData(x_train, y_train)
        validation_dataset = FormTorchData(x_valid, y_valid)
        test_dataset = FormTorchData(x_test, y_test)
        
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=False)
        validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=False)    
    
    
    
    
        for epoch in range(0, 100):  # loop over the dataset multiple times
            total_train_loss = 0.0
            total_val_loss= 0.0
            total_train_rmse=0.0
            total_val_rmse=0.0
            model.train()
            
            for data, labels in train_dataloader:
                batch_size = data.shape[0]
                data, labels = data.reshape(batch_size, -1).to(device), labels.float().to(device)
                
                

                # Forward Pass
                targets = model(data)
                    
                # Find the Loss
                ##https://discuss.pytorch.org/t/regression-with-multiple-outputs/175428/4
                loss = criterion(targets, labels)
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
            print(f'\n Epoch {epoch+1} \t\t Averge Training Loss: {total_train_loss / len(train_dataloader)}')
            print(f'\n Epoch {epoch+1} \t\t Averge Training RMSE: {total_train_rmse / len(train_dataloader)}')            
    
    
            model.eval()
            with torch.no_grad():
              for val_data, val_labels in validation_dataloader:
                  
                batch_size = val_data.shape[0]
                  
                val_data, val_labels = val_data.float().reshape(batch_size, -1).to(device), val_labels.float().to(device)
                
                outputs = model(val_data)
                
                
                val_loss = criterion(outputs, val_labels)
                val_rmse =  torch.sqrt(val_loss)
                
                total_val_loss += val_loss.item()
                total_val_rmse += val_rmse.item()
            
            avg_valid_loss = total_val_loss/len(validation_dataloader)
            avg_valid_rmse = total_val_rmse/len(validation_dataloader)
            print(f'\n Epoch {epoch+1} \t\t Averge Validation Loss: {total_val_loss / len(validation_dataloader)}')
            print(f'\n Epoch {epoch+1} \t\t Averge Validation RMSE: {total_val_rmse / len(validation_dataloader)}')
    
    
            # Here we save a checkpoint. It is automatically registered with
            # Ray Tune and can be accessed through `session.get_checkpoint()`
            # API in future iterations.
            os.makedirs(args.log_path, exist_ok=True)
            torch.save(
                (model.state_dict(), optimizer.state_dict()), f"{args.log_path}\{args.filename}.pt")
            checkpoint = Checkpoint.from_directory(f"{args.log_path}")
            session.report({"loss": (total_val_rmse / len(validation_dataloader))}, checkpoint=checkpoint)
        print("Finished Training")
        




        
    def scaler_getter(args):

        
        TorchHyundaiDataset = load_data(args)
        
        dataset_size = len(TorchHyundaiDataset)
        train_size = int(dataset_size * 0.8)
        validation_size = int(dataset_size * 0.1)
        test_size = dataset_size - train_size - validation_size
        generator_train = torch.Generator().manual_seed(2023)
        train_dataset, validation_dataset, test_dataset = random_split(TorchHyundaiDataset, [train_size, validation_size, test_size],
                                                                      generator=generator_train)
        
        transformation = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  ##incase I wanna scale image between [0,1] transforms.ToTensor() is suffice. if -1, 1, normalize.
        )
        
        if args.y_scale == 'minmax' or args.y_scale == 'standard':
            if args.y_scale == 'minmax':
                y_scaler = MinMaxScaler()
            elif args.y_scale == 'standard':
                y_scaler = StandardScaler()
            
            
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
            
            
        else:
            x_train = train_dataset.dataset.x[train_dataset.indices]
            x_train = np.stack([transformation(x).reshape(args.image_dim) for x in x_train])
            x_train = torch.from_numpy(x_train)
            y_train = train_dataset.dataset.y[train_dataset.indices]
            y_train = torch.from_numpy(y_train)
            
            x_valid = validation_dataset.dataset.x[validation_dataset.indices]
            x_valid = np.stack([transformation(x).reshape(args.image_dim) for x in x_valid])
            x_valid = torch.from_numpy(x_valid)
            y_valid = validation_dataset.dataset.y[validation_dataset.indices]
            y_valid = torch.from_numpy(y_valid)
            
            x_test = test_dataset.dataset.x[test_dataset.indices]
            x_test = np.stack([transformation(x).reshape(args.image_dim) for x in x_test])
            x_test = torch.from_numpy(x_test)
            y_test = test_dataset.dataset.y[test_dataset.indices]
            y_test = torch.from_numpy(y_test)
            
                        
        
        
        return y_scaler
    
        
        
        
    def test_best_model(best_result, y_scaler):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.model == 'mor':
            best_trained_model = MultOutRegressor(args.image_dim , args.label_dim, p=best_result.config['p']).to(device)
        elif args.model == 'mosa':
            # model = MultOutRegressorSelfAttentionMLP(img_dim=args.image_dim, hidden_dim=4480, seq_len=140, embed_dim=32).to(device)
            model = MultOutRegressorSelfAttentionMLP(img_dim=420, seq_len=140, embed_dim=64).to(device)
        elif args.model == 'mocr':
        
            best_trained_model = MultOutChainedRegressor(args.image_dim , args.label_dim, order=sorted(set(range(0, args.label_dim)))).to(device)
            
        
        best_trained_model.to(device)
    
        checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), f"{args.filename}.pt")
    
        transformation = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  ##incase I wanna scale image between [0,1] transforms.ToTensor() is suffice. if -1, 1, normalize.
        )    
    
    
        model_state, optimizer_state = torch.load(checkpoint_path)
        best_trained_model.load_state_dict(model_state)
        
        TorchHyundaiDataset = load_data(args)
        dataset_size = len(TorchHyundaiDataset)
        train_size = int(dataset_size * 0.8)
        validation_size = int(dataset_size * 0.1)
        test_size = dataset_size - train_size - validation_size
        generator_test = torch.Generator().manual_seed(2023)
        train_dataset, validation_dataset, test_dataset = random_split(TorchHyundaiDataset, [train_size, validation_size, test_size],
                                                                       generator=generator_test)
        
        if args.y_scale == 'minmax' or args.y_scale == 'standard':

            x_test = test_dataset.dataset.x[test_dataset.indices]
            x_test = np.stack([transformation(x).reshape(args.image_dim) for x in x_test])
            y_test = test_dataset.dataset.y[test_dataset.indices]
            y_test_scaled = y_scaler.transform(y_test)
            y_test_scaled = torch.from_numpy(y_test_scaled)
            test_dataset = FormTorchData(x_test, y_test_scaled)
        
        else:
            x_test = test_dataset.dataset.x[test_dataset.indices]
            x_test = np.stack([transformation(x).reshape(args.image_dim) for x in x_test])
            y_test = test_dataset.dataset.y[test_dataset.indices]
            y_test = torch.from_numpy(y_test)
            test_dataset = FormTorchData(x_test, y_test)
            

        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        

        criterion = nn.MSELoss().to(device)
        
        total_test_loss = 0.0
        total_test_rmse = 0.0
        predictions = []
        for i, (data, label) in enumerate(test_dataloader):
            batch_size = data.shape[0]
            preds = best_trained_model(data.reshape(batch_size, -1).to(device))
            label = label.float().to(device)
        
            
            test_loss = criterion(preds, label)
            test_rmse =  torch.sqrt(test_loss)
            
            total_test_loss += test_loss.item()
            total_test_rmse += test_rmse.item()
            
            predictions.append(preds)
            
        avg_test_loss = total_test_loss/len(test_dataloader)
        avg_test_rmse = total_test_rmse/len(test_dataloader)
        print(f'Best trial test set \n \t\t Averge test Loss: {avg_test_loss}')
        print(f'Best trial test set \n \t\t Averge test RMSE: {avg_test_rmse}')        
        predictions = torch.cat([tensor for tensor in predictions], dim=0) ## putting all together into a tensor
        return predictions
    
    
    
    def main(args):
        
        config = {

            "lr": tune.grid_search([0.001, 0.05, 0.01]),
            "p": tune.grid_search([0.2, 0.3, 0.4, 0.5]),
            "batch_size": tune.grid_search([64, 128]),
            "epochs": tune.choice([1, 2, 3]),
            }

        
        scaler = scaler_getter(args)
        
        
        scheduler = ASHAScheduler(
            max_t=args.max_evals,
            grace_period=1,
            reduction_factor=2)
        
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(train_multioutput),
                resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial}
            ),
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                scheduler=scheduler,
                num_samples=args.num_samples,
            ),
            param_space=config,
            run_config=air.RunConfig(storage_path=args.log_path, name=args.filename)
        )
        results = tuner.fit()
        
        
        best_result = results.get_best_result("loss", "min")
        dir(best_result)
        
        
        
        print("Best trial config: {}".format(best_result.config))
        best_result_info = best_result.metrics
        best_result_metrics = best_result.metrics_dataframe
        best_result_metrics.to_csv(f"{args.log_path}/{args.filename}_best_result_metrics.csv", index=False)
        best_result_info = pd.DataFrame.from_dict(best_result_info)

        
        
        print("Best trial final validation loss: {}".format(
            best_result.metrics["loss"]))
    
    
        predictions = test_best_model(best_result, scaler)
        return predictions
    
    main(args)
    
    

elif args.mode == 'ml':
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
        train_size = int(dataset_size * 0.8)
        validation_size = int(dataset_size * 0.1)
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



    elif args.x_scale == 'standard':

        x_rootdir = args.img_data_path
        y_rootdir = args.label_data_path
        
        datanames = os.listdir(y_rootdir)
    
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
        
        x_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x_train)
        x_valid = x_scaler.transform(x_valid)
        
        
    elif args.x_scale == 'minmax':

        x_rootdir = args.img_data_path
        y_rootdir = args.label_data_path
        
        datanames = os.listdir(y_rootdir)
    
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

        x_scaler = MinMaxScaler()
        x_train = x_scaler.fit_transform(x_train)
        x_valid = x_scaler.transform(x_valid)


    else: 
        x_rootdir = args.img_data_path
        y_rootdir = args.label_data_path
        
        datanames = os.listdir(y_rootdir)
    
        y_data = []
        x_data = []
        for dataname in datanames:
            data = pd.read_csv(f'{y_rootdir}/{dataname}')    
            y_data.append(data.to_numpy())
        
            x_image_name = dataname[7:-4]  + '.bmp'
            img_path = os.path.join(x_rootdir, x_image_name)   
            image = io.imread(img_path) 
            x_data.append(image)
        
        x_data = np.stack(x_data)
        x_data = x_data.reshape(len(x_data), -1)
            
        y_data = np.stack(y_data)
        y_data = np.squeeze(y_data, axis=2)
        
        x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=args.validation_size, shuffle=True, random_state=2023)



    
    if args.y_scale == 'standard':
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train)
        y_valid = y_scaler.transform(y_valid)
        
        
    elif args.y_scale == 'minmax':
        y_scaler = MinMaxScaler()
        y_train = y_scaler.fit_transform(y_train)
        y_valid = y_scaler.transform(y_valid)
    else: 
        pass ##이 경우 전혀 스케일링이나 정규화되지 않은 데이터가 들어가게 된다.
    
    
    if args.model == 'SGDReg':
        
    
   
        
        sgd_space = {
                'alpha': hp.uniform("alpha", 0.00001, 0.01),
                'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet', None]),
                'l1_ratio': hp.uniform('l1_ratio', 0, 1),
                'learning_rate': hp.choice('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']),
                'early_stopping' : hp.choice('early_stopping', [True, False])
                # 'max_iter' : hp.randint('max_iter', 10000),
            }
        
        
        
        
        
        def objective(args):
            model = MultiOutputRegressor(SGDRegressor(**args, random_state= 2023))
        
            model.fit(x_train, y_train)
        
            y_pred = model.predict(x_valid)
            
        
            print("Hyperparameters : {}".format(args)) ## This can be commented if not needed.
            print("RMSE mean: {}\n".format(mean_squared_error(y_valid, y_pred, squared = False)))
            print("RMSE sum: {}\n".format(mean_squared_error(y_valid, y_pred, squared = False)))
            
            return {'loss': mean_squared_error(y_valid, y_pred, squared = False), 'status': STATUS_OK}
        
        trials = Trials()
        
        start = time.time()
        best_results = fmin(objective,
                                     space=sgd_space,
                                     algo=tpe.suggest,
                                     trials=trials,
                                     max_evals=args.max_evals)

        end = time.time()
        print(f'The hyper-parameter tuning took {end - start} seconds')
        
        hp_results = {}
        hp_results['x_scale'] = args.x_scale
        hp_results['y_scale'] = args.y_scale
        hp_results['model_name'] = args.model
        hp_results['best_hp'] = trials.best_trial['misc']['vals']
        hp_results['best_loss'] = trials.best_trial['result']['loss']
        hp_results = json.dumps(hp_results, cls=NpEncoder)
        with open(f'{args.log_path}/{args.filename}/hp_results.json', "w") as file:
            json.dump(hp_results, file)  
    
    
    
    elif args.model == 'Ridge':
    
        
        ridge_space = {
                'alpha': hp.uniform("alpha", 0, 5),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'solver': hp.choice('solver', ['svd', 'lsqr', 'sparse_cg', 'sag', ]), ##'saga', 'lbfgs'
                # 'max_iter' : hp.randint('max_iter', 10000),
                }
        
        
        
        def objective(args):
            model = Ridge(**args, random_state= 2023) 
        
            model.fit(x_train, y_train)
        
            y_pred = model.predict(x_valid)
            
                
            print("Hyperparameters : {}".format(args)) ## This can be commented if not needed.
            print("RMSE: {}\n".format(mean_squared_error(y_valid, y_pred, squared = False)))

            
            
            return {'loss': mean_squared_error(y_valid, y_pred, squared = False), 'status': STATUS_OK}#, 'params': args}
        
        trials = Trials()
        

        
        start = time.time()
        best_results = fmin(objective,
                                     space=ridge_space,
                                     algo=tpe.suggest,
                                     trials=trials,
                                     max_evals=args.max_evals,
                                     )

        end = time.time()
        print(f'The hyper-parameter tuning took {end - start} seconds')
        
        
        hp_results = {}
        hp_results['x_scale'] = args.x_scale
        hp_results['y_scale'] = args.y_scale
        hp_results['model_name'] = args.model
        hp_results['best_hp'] = trials.best_trial['misc']['vals']
        hp_results['best_loss'] = trials.best_trial['result']['loss']
        hp_results = json.dumps(hp_results, cls=NpEncoder)
        with open(f'{args.log_path}/{args.filename}/hp_results.json', "w") as file:
            json.dump(hp_results, file)   

    
    elif args.model == 'RCV':
        
        rcv_space = {
         
                'alphas':
                    [
                        hp.uniform('alpha1', 0.01, 1.0),
                        hp.uniform('alpha2', 0.7, 3.0),
                       hp.uniform('alpha3', 5, 13)
                    ],    
                
                # 'scoring': hp.choice('scoring', [None, 'auto']),
                'alpha_per_target': hp.choice('alpha_per_target', [True, False]),
                # 'max_iter' : hp.randint('max_iter', 10000),
                }
        
        
        
        def objective(args):
            model = RidgeCV(**args, random_state= 2023)
        
            model.fit(x_train, y_train)
        
            y_pred = model.predict(x_valid)


            print("Hyperparameters : {}".format(args)) ## This can be commented if not needed.
            print("RMSE: {}\n".format(mean_squared_error(y_valid, y_pred, squared = False)))
            
            
            return {'loss': mean_squared_error(y_valid, y_pred, squared = False), 'status': STATUS_OK}
        
        trials = Trials()

        
        start = time.time()
        best_results = fmin(objective,
                                     space=rcv_space,
                                     algo=tpe.suggest,
                                     trials=trials,
                                     max_evals=args.max_evals)

        end = time.time()
        print(f'The hyper-parameter tuning took {end - start} seconds')
        
        
        hp_results = {}
        hp_results['x_scale'] = args.x_scale
        hp_results['y_scale'] = args.y_scale
        hp_results['model_name'] = args.model
        hp_results['best_hp'] = trials.best_trial['misc']['vals']
        hp_results['best_loss'] = trials.best_trial['result']['loss']
        hp_results = json.dumps(hp_results, cls=NpEncoder)
        with open(f'{args.log_path}/{args.filename}/hp_results.json', "w") as file:
            json.dump(hp_results, file) 
    
    
    
    
    elif args.model == 'KNeiReg':
        
        
        kneireg_space = {
         
                'weights': hp.choice('weights', ['uniform', 'distance']),
                'n_neighbors' : hp.randint('n_neighbors', 2, 20),
                'leaf_size' : hp.randint('leaf_size', 5, 50),
                'p' : hp.choice('p', [1, 2]),
                'algorithm' : hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                }
        
        
        
        def objective(args):
            model = KNeighborsRegressor(**args)
        
            model.fit(x_train, y_train)
        
            y_pred = model.predict(x_valid)

        
            print("Hyperparameters : {}".format(args)) ## This can be commented if not needed.
            print("RMSE: {}\n".format(mean_squared_error(y_valid, y_pred, squared = False)))

            
            return {'loss': mean_squared_error(y_valid, y_pred, squared = False), 'status': STATUS_OK}
        
        trials = Trials()

        
        start = time.time()
        best_results = fmin(objective,
                                     space=kneireg_space,
                                     algo=tpe.suggest,
                                     trials=trials,
                                     max_evals=args.max_evals)

        end = time.time()
        print(f'The hyper-parameter tuning took {end - start} seconds')
        
        
        hp_results = {}
        hp_results['x_scale'] = args.x_scale
        hp_results['y_scale'] = args.y_scale
        hp_results['model_name'] = args.model
        hp_results['best_hp'] = trials.best_trial['misc']['vals']
        hp_results['best_loss'] = trials.best_trial['result']['loss']
        hp_results = json.dumps(hp_results, cls=NpEncoder)
        with open(f'{args.log_path}/{args.filename}/hp_results.json', "w") as file:
            json.dump(hp_results, file)      
    
    
    
    elif args.model == 'RanForestReg':
    
        
        randomforest_space = {

                'criterion': hp.choice('criterion', ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']),
                'n_estimators' : hp.randint('n_estimators', 10, 1000),
                'min_samples_split' : hp.randint('min_samples_split', 2, 10),
                'min_samples_leaf' : hp.randint('min_samples_leaf', 1, 5),
                }
        
        
        
        def objective(args):
            model = RandomForestRegressor(**args, random_state= 2023)
        
            model.fit(x_train, y_train)
        
            y_pred = model.predict(x_valid)

        
            print("Hyperparameters : {}".format(args)) ## This can be commented if not needed.
            print("RMSE: {}\n".format(mean_squared_error(y_valid, y_pred, squared = False)))
            
            
            return {'loss': mean_squared_error(y_valid, y_pred, squared = False), 'status': STATUS_OK}
        
        trials = Trials()
        

        
        
        start = time.time()
        best_results = fmin(objective,
                                     space=randomforest_space,
                                     algo=tpe.suggest,
                                     trials=trials,
                                     max_evals=args.max_evals)
                                     
        end = time.time()
        print(f'The hyper-parameter tuning took {end - start} seconds')
        
        
        hp_results = {}
        hp_results['x_scale'] = args.x_scale
        hp_results['y_scale'] = args.y_scale
        hp_results['model_name'] = args.model
        hp_results['best_hp'] = trials.best_trial['misc']['vals']
        hp_results['best_loss'] = trials.best_trial['result']['loss']
        hp_results = json.dumps(hp_results, cls=NpEncoder)
        with open(f'{args.log_path}/{args.filename}/hp_results.json', "w") as file:
            json.dump(hp_results, file)    
    
    
    elif args.model == 'GBReg': ##The hyper-parameter tuning took 46411.84270000458 seconds. why soooo long?
        
        gradientboost_space = {

                'loss': hp.choice('loss', ['squared_error', 'absolute_error', 'huber', 'quantile']),
                'learning_rate' : hp.uniform('learning_rate', 0.001, 0.3),
                'n_estimators' : hp.randint('n_estimators', 25, 400),
                'subsample' : hp.uniform('subsample', 0.1, 1.0),
                'min_samples_leaf' : hp.randint('min_samples_leaf', 1, 2),
                }
        
        
        
        def objective(args):
            model = MultiOutputRegressor(GradientBoostingRegressor(**args, random_state= 2023))
        
            model.fit(x_train, y_train)
        
            y_pred = model.predict(x_valid)

        
            print("Hyperparameters : {}".format(args)) ## This can be commented if not needed.
            print("RMSE: {}\n".format(mean_squared_error(y_valid, y_pred, squared = False)))
            
            
            return {'loss': mean_squared_error(y_valid, y_pred, squared = False), 'status': STATUS_OK}
        
        trials = Trials()
        
        start = time.time()
        best_results = fmin(objective,
                                     space=gradientboost_space,
                                     algo=tpe.suggest,
                                     trials=trials,
                                     max_evals=args.max_evals)

        end = time.time()
        print(f'The hyper-parameter tuning took {end - start} seconds')
        
        
        hp_results = {}
        hp_results['x_scale'] = args.x_scale
        hp_results['y_scale'] = args.y_scale
        hp_results['model_name'] = args.model
        hp_results['best_hp'] = trials.best_trial['misc']['vals']
        hp_results['best_loss'] = trials.best_trial['result']['loss']
        hp_results = json.dumps(hp_results, cls=NpEncoder)
        with open(f'{args.log_path}/{args.filename}/hp_results.json', "w") as file:
            json.dump(hp_results, file) 
    
    
    
    
    elif args.model == 'SVR': ##it takes too much time  ##FutureWarning: The default value of `dual` will change from `True` to `'auto'` in 1.5. Set the value of `dual` explicitly to suppress the warning.  warnings.warn(
        
        linsvr_space = {
         
                'C' : hp.uniform('C', 0.3, 2.0),
                'max_iter' : hp.randint('max_iter', 500, 2500),
                }
        
     
        
        def objective(args):
            model = MultiOutputRegressor(LinearSVR(**args, random_state= 2023))
        
            model.fit(x_train, y_train)
        
            y_pred = model.predict(x_valid)

        
            print("Hyperparameters : {}".format(args)) ## This can be commented if not needed.
            print("RMSE: {}\n".format(mean_squared_error(y_valid, y_pred, squared = False)))
            
            
            return {'loss': mean_squared_error(y_valid, y_pred, squared = False), 'status': STATUS_OK}
        
        trials = Trials()
        
        start = time.time()
        best_results = fmin(objective,
                                     space=linsvr_space,
                                     algo=tpe.suggest,
                                     trials=trials,
                                     max_evals=args.max_evals)

        end = time.time()
        print(f'The hyper-parameter tuning took {end - start} seconds')
        
        
        hp_results = {}
        hp_results['x_scale'] = args.x_scale
        hp_results['y_scale'] = args.y_scale
        hp_results['model_name'] = args.model
        hp_results['best_hp'] = trials.best_trial['misc']['vals']
        hp_results['best_loss'] = trials.best_trial['result']['loss']
        hp_results = json.dumps(hp_results, cls=NpEncoder)
        with open(f'{args.log_path}/{args.filename}/hp_results.json', "w") as file:
            json.dump(hp_results, file) 
    
    
    elif args.model == 'PAReg': ##110 seconds for 1 trial done
    
        
        PAReg_space = {
                'C' :  hp.quniform('C', 0.01, 1.0, 0.1),
                'early_stopping': hp.choice('early_stopping', [True, False]),
                'shuffle': hp.choice('shuffle', [True, False]),
                'loss': hp.choice('loss', ['epsilon_insensitive', 'squared_epsilon_insensitive']),
        
                }
        
        
        
        def objective(args):  
            model = MultiOutputRegressor(PassiveAggressiveRegressor(**args, random_state= 2023))
        
            model.fit(x_train, y_train)
        
            y_pred = model.predict(x_valid)

        
            print("Hyperparameters : {}".format(args)) ## This can be commented if not needed.
            print("RMSE: {}\n".format(mean_squared_error(y_valid, y_pred, squared = False)))
            
            
            return {'loss': mean_squared_error(y_valid, y_pred, squared = False), 'status': STATUS_OK}
        
        trials = Trials()
        
        start = time.time()
        best_results = fmin(objective,
                                     space=PAReg_space,
                                     algo=tpe.suggest,
                                     trials=trials,
                                     max_evals=args.max_evals)

        end = time.time()
        print(f'The hyper-parameter tuning took {end - start} seconds')
        
        
        hp_results = {}
        hp_results['x_scale'] = args.x_scale
        hp_results['y_scale'] = args.y_scale
        hp_results['model_name'] = args.model
        hp_results['best_hp'] = trials.best_trial['misc']['vals']
        hp_results['best_loss'] = trials.best_trial['result']['loss']
        hp_results = json.dumps(hp_results, cls=NpEncoder)
        with open(f'{args.log_path}/{args.filename}/hp_results.json', "w") as file:
            json.dump(hp_results, file)  





    elif args.model == 'lgb': ##110 seconds for 1 trial done

        
        lgb_space = {

                'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart', 'rf']),
                'n_estimators' :  hp.randint('n_estimators', 50, 500),
                'num_leaves' :  hp.randint('num_leaves', 20, 100),
                'max_depth ': hp.randint('max_depth ', 5, 50),
                'importance_type': hp.choice('importance_type', ['split', 'gain']),
                'reg_alpha': hp.uniform("reg_alpha", 0.1, 0.5),
                'reg_lambda': hp.uniform("reg_lambda", 0.1, 0.5),
                'learning_rate': hp.uniform("learning_rate", 0.05, 0.5),
                }
        
        
        
        def objective(args):  
            model = MultiOutputRegressor(lgb.LGBMRegressor(**args, n_jobs=n_cpu-2, random_state= 2023))
        
            model.fit(x_train, y_train)
        
            y_pred = model.predict(x_valid)

        
            print("Hyperparameters : {}".format(args)) ## This can be commented if not needed.
            print("RMSE: {}\n".format(mean_squared_error(y_valid, y_pred, squared = False)))
            
            
            return {'loss': mean_squared_error(y_valid, y_pred, squared = False), 'status': STATUS_OK}
        
        trials = Trials()
        
        start = time.time()
        best_results = fmin(objective,
                                     space=lgb_space,
                                     algo=tpe.suggest,
                                     trials=trials,
                                     max_evals=args.max_evals)

        end = time.time()
        print(f'The hyper-parameter tuning took {end - start} seconds')
        
        
        hp_results = {}
        hp_results['x_scale'] = args.x_scale
        hp_results['y_scale'] = args.y_scale
        hp_results['model_name'] = args.model
        hp_results['best_hp'] = trials.best_trial['misc']['vals']
        hp_results['best_loss'] = trials.best_trial['result']['loss']
        hp_results = json.dumps(hp_results, cls=NpEncoder)
        with open(f'{args.log_path}/{args.filename}/hp_results.json', "w") as file:
            json.dump(hp_results, file)  
            
            



    elif args.model == 'xgb':

        
        xgb_space = {

                'n_estimators' :  hp.randint('n_estimators', 10, 1200),
                'max_depth ': hp.randint('max_depth ', 5, 50),
                'grow_policy': hp.choice('grow_policy', [0, 1]),
                'learning_rate': hp.uniform("learning_rate", 0.0001, 0.1),
                'tree_method': hp.choice('tree_method', ['gpu_hist']),
                'booster': hp.choice('booster', ['gbtree', 'gblinear', 'dart']),
        
                }
        
        
        
        def objective(args):  
            model = MultiOutputRegressor(xgb.XGBRegressor(**args, n_jobs=n_cpu-2, random_state= 2023))
        
            model.fit(x_train, y_train)
        
            y_pred = model.predict(x_valid)

        
            print("Hyperparameters : {}".format(args)) ## This can be commented if not needed.
            print("RMSE: {}\n".format(mean_squared_error(y_valid, y_pred, squared = False)))

            
            return {'loss': mean_squared_error(y_valid, y_pred, squared = False), 'status': STATUS_OK}
        
        trials = Trials()
        
        start = time.time()
        best_results = fmin(objective,
                                     space=xgb_space,
                                     algo=tpe.suggest,
                                     trials=trials,
                                     max_evals=args.max_evals)

        end = time.time()
        print(f'The hyper-parameter tuning took {end - start} seconds')
        
        
        hp_results = {}
        hp_results['x_scale'] = args.x_scale
        hp_results['y_scale'] = args.y_scale
        hp_results['model_name'] = args.model
        hp_results['best_hp'] = trials.best_trial['misc']['vals']
        hp_results['best_loss'] = trials.best_trial['result']['loss']
        hp_results = json.dumps(hp_results, cls=NpEncoder)
        with open(f'{args.log_path}/{args.filename}/hp_results.json', "w") as file:
            json.dump(hp_results, file)  



            
    else:
        raise Exception("Invalid model name is used in {args.model}. Try again")
        
else:
    raise Exception('you must choose mode argument between dl and ml')
    
    
   
 

