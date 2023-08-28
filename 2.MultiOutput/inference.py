# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 19:30:20 2023

@author: hojun
"""
import os
from skimage import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib 
from torch_multiout_models import MultOutRegressor, MultOutChainedRegressor, predict, MultOutRegressorSelfAttentionMLP, MultOutChainedSelfAttentionRegressor
from data import TorchHyundaiData, FormTorchData, TorchDropEmptyHyundaiData, TorchHyundaiData_inference, TorchDropEmptyHyundaiData_inference, FormTorchData_inference
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
import argparse
import torch
from tqdm import tqdm
import numpy as np 
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split



parser = argparse.ArgumentParser(description='Hyundai Multi Output Regression Inferencing')
parser.add_argument("--model_dir_path", '-idp', type=str,  default=r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\Second_part\logs\lgb_gpu_xscale_new_test", help = 'path of directory that contains image data')          # extra value default="decimal"
parser.add_argument("--img_data_path", '-dm', default=r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\data", type=str, help='Model name saved the trained models as')
parser.add_argument("--output_path", '-lp', default=r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\Second_part\logs\lgb_gpu_xscale_new_test\outputs", type=str, help='Serving as logging file path and model saving path')
##DL part
parser.add_argument("--device", '-d', default='cpu', type=str, help='Which device to use. "cpu" or "cuda"')
args = parser.parse_args()



def main(args):
    
    
    
    isExist = os.path.exists(f'{args.output_path}')
    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(f'{args.output_path}')
        print("The new output directory is created!")
        

    
    model_dir_path = args.model_dir_path
    

    # Opening JSON file
    with open(rf'{args.model_dir_path}/config.json') as json_file:
        config = json.load(json_file)
    
    
    if os.path.exists(os.path.join(model_dir_path, 'model.pt')):
        model_path = os.path.join(model_dir_path, 'model.pt')    
        
    elif os.path.exists(os.path.join(model_dir_path, 'model.joblib')):
        model_path = os.path.join(model_dir_path, 'model.joblib')    
        
    model_name = model_path.split("\\")[-1]
        
    if model_name == 'model.pt':
        
        checkpoint = torch.load(f'{model_path}')

        datashape = checkpoint['datashape']

        transformation = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  ##incase I wanna scale image between [0,1] transforms.ToTensor() is suffice. if -1, 1, normalize.
        )


        if not os.path.exists(f'{args.model_dir_path}/x_scaler.pkl'):
        
            if datashape == 'original':
                TorchHyundaiDataset = TorchHyundaiData_inference(args.img_data_path, transform=transformation) # 간단 예시
            elif datashape == 'dropempty':
                TorchHyundaiDataset = TorchDropEmptyHyundaiData_inference(args.img_data_path, transform=transformation) # 간단 예시            
            else:
                raise ValueError('choose datashape between original and dropempty')

        dataloader = DataLoader(TorchHyundaiDataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
                
        device = args.device
        

        
        image_dim = checkpoint['image_dim']
        label_dim = checkpoint['label_dim']
        seq_len = checkpoint['seq_len']
        embed_dim = checkpoint['embed_dim']
        
        if config['model'] == 'mor':
            model = MultOutRegressor(image_dim , label_dim).to(device)
            # model = MultOutRegressor(420 , args.label_dim).to(device)
        
        elif config['model'] == 'morsa':
            model = MultOutRegressorSelfAttentionMLP(img_dim=image_dim, seq_len=seq_len, embed_dim=embed_dim).to(device)
        
        elif config['model'] == 'mocr':
            model = MultOutChainedRegressor(image_dim , label_dim, order=sorted(set(range(0, label_dim)))).to(device)

        elif config['model'] == 'mocrsa':
            model = MultOutChainedSelfAttentionRegressor(image_dim, label_dim, order=sorted(set(range(0, label_dim))), hidden_dim=embed_dim, seed = 2023).to(device)
        
        else:
            raise Exception("You must choose a model between mor and mocr if you want to inference from pytorch model")
            

        model.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer']) 
        model.eval()

        if os.path.exists(f'{args.model_dir_path}/y_scaler.pkl'):
            
            y_scaler = pickle.load(open(f'{args.model_dir_path}/y_scaler.pkl', 'rb'))
            print('loading y_scaler...')
        else:
            pass
            
        # model.eval()
        model.to(device)
        ##making result and prediction by test data
        test_predictions = []

        predictions = []
        for i, data in tqdm(enumerate(dataloader)):
            batch_size = data.shape[0]
            preds = model(data.reshape(batch_size, -1).to(device))#.to('cpu'))
            if os.path.exists(f'{args.model_dir_path}/y_scaler.pkl'):
                preds = torch.tensor(y_scaler.inverse_transform(preds.detach().cpu()))
            else:    
                pass
                
                
            predictions.append(preds)

        test_predictions = (torch.cat([tensor for tensor in predictions], dim=0)).numpy() ## putting all together into a tensor
    
        with open(f'{args.output_path}/inference.npy', 'wb') as f:
            np.save(f, test_predictions)
            
        with open(f"{args.output_path}/inference.pickle", 'wb') as f:
            pickle.dump(test_predictions, f, protocol=pickle.HIGHEST_PROTOCOL)     


        test_predictions_df = pd.DataFrame(test_predictions)
        test_predictions_df.to_csv(f"{args.output_path}/inference.csv", index=False)




    elif model_name == 'model.joblib':  
        if os.path.exists(f'{args.model_dir_path}/x_scaler.pkl'):
            
            datanames = os.listdir(config['img_data_path'])
            x_data = []
            for dataname in tqdm(datanames): 
                img_path = os.path.join(config['img_data_path'], dataname)   
                image = io.imread(img_path)   ##dimension: 10 x 30 x 3
                if config['datashape'] == 'original':
                    x_data.append(image)
                elif config['datashape'] == 'dropempty':
                    x_data.append(image[:, np.r_[0:6, 10:16, 20:22]])
                else:
                    raise ValueError('error!')
                
        
            x_data = np.stack(x_data)
            x_data = x_data.reshape(len(x_data), -1)
            x_data = x_data.astype(np.float32)
            
            x_scaler = pickle.load(open(f'{args.model_dir_path}/x_scaler.pkl', 'rb'))
            x_data = x_scaler.fit_transform(x_data)

  
        elif not os.path.exists(f'{args.model_dir_path}/x_scaler.pkl'):
            
        
            transformation = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  ##incase I wanna scale image between [0,1] transforms.ToTensor() is suffice. if -1, 1, normalize.
            )
            
            datashape = config['datashape']
            
            
            if datashape == 'original':
                TorchHyundaiDataset = TorchHyundaiData_inference(args.img_data_path, transform=transformation) # 간단 예시
            elif datashape == 'dropempty':
                TorchHyundaiDataset = TorchDropEmptyHyundaiData_inference(args.img_data_path, transform=transformation) # 간단 예시            
            else:
                raise ValueError('choose datashape between original and dropempty')

            x_data = TorchHyundaiDataset.x
            x_data = np.stack([transformation(x).reshape(config['image_dim']) for x in x_data])

        else:  
            raise ValueError('data must be normalized by scalers or one')

        

        model = joblib.load(f'{args.model_dir_path}/model.joblib')
        y_pred = model.predict(x_data)
        # y_pred[:10]

        if os.path.exists(f'{args.model_dir_path}/y_scaler.pkl'):
            y_scaler = pickle.load(open(f'{args.model_dir_path}/y_scaler.pkl', 'rb'))
            y_pred = y_scaler.inverse_transform(y_pred)
            print('inference inverse transformed')
        else:
            pass
            
        with open(f"{args.output_path}/inference.pickle", 'wb') as f:
            pickle.dump(y_pred, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        y_pred_df = pd.DataFrame(y_pred)
        y_pred_df.to_csv(f"{args.output_path}/inference.csv", index=False)
        
        
if __name__ == "__main__":
    main(args)        
    