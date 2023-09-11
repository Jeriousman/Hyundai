#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 19:23:12 2023

@author: hannah
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from typing import List
import torch
import torch.nn as nn
from torch import tensor
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from utils import gradient_penalty, save_checkpoint, load_checkpoint, gradient_penalty_onlyimage1d, torch_initialize_weights, NormalNLLLoss, UnNormalize, UnToTensor
from models import  Generator_discrete_wgan, Discriminator_discrete_wgan, initialize_weights, InfoGANMLPDiscriminator, InfoGANSelfAttentionMLDiscriminator, InfoGANSelfAttentionMLPGenerator, InfoGANMLPGenerator, InfoGANMLPQHead_discrete, InfoGANMLPDHead, SelfAttentionMLPDiscriminator, SelfAttentionMLPGenerator
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from layers import activation_layer, MLPLayers
from dataset import HyundaiDataset_discrete
import numpy as np
from torchvision.utils import save_image
import random
import argparse
import torch.nn.functional as F




parser = argparse.ArgumentParser(description='Hyundai Image Generation Inference')
parser.add_argument("--model_path", '-mp', default=r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\CGAN\logs\vanilagan_wgangploss_vanila_dropempty_400epochs_test\vanila_model.pt", type=str, help='Path of saved model that will be used for inference ')
parser.add_argument("--gen_img_path", '-gip', default=r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\CGAN\logs\vanilagan_wgangploss_vanila_dropempty_400epochs_test\inference_output", type=str, help='path of saving newly generated images as final output')
parser.add_argument("--num_gen_image", '-ngi', default=10, type=int, help='MOR = Multi Output Regressor, MOCR = Multi Output Chained Regressor')
parser.add_argument("--durability", '-d', default=1.6, type=float, help='Durability to choose. It should be among 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9')
parser.add_argument("--mass", '-m', default=2.21, type=float, help='Mass to choose. It should be among 2.13, 2.14, 2.15, 2.16, 2.17, 2.18, 2.19, 2.2, 2.21, 2.22, 2.23, 2.24')
parser.add_argument("--strut1", '-s1', nargs='+',  type=int, default=[255, 255, 255], help='Values of fixed pixel 1') ##required=True,
parser.add_argument("--strut2", '-s2', nargs='+',  type=int, default=[255, 255, 255], help='Values of fixed pixel 2') ##required=True,
parser.add_argument("--device", '-dv', default='cuda', type=str, help='Which device to use between cpu and cuda')
args = parser.parse_args()



def main(args):
    
    model_path = args.model_path.split('\\')
    model_name = model_path[-1]
    
    if model_name == 'vanila_model.pt':
            
            
        checkpoint = torch.load(f'{args.model_path}')

        
            
        disc_state_dict =  checkpoint['disc_state_dict']
        gen_state_dict =  checkpoint['gen_state_dict']
        opt_disc_state_dict =  checkpoint['opt_disc']
        opt_gen_state_dict =  checkpoint['opt_gen']
        learning_rate_gen = checkpoint['learning_rate_gen']
        learning_rate_disc =  checkpoint['learning_rate_disc']
        batch_size =  checkpoint['batch_size']
        epochs =  checkpoint['epochs']
        constraint_lambda =  checkpoint['constraint_lambda']
        lambda_gp =  checkpoint['lambda_gp']
        critic_iter = checkpoint['critic_iter']
        opt_beta1 = checkpoint['opt_beta1']
        opt_beta2 = checkpoint['opt_beta2']
        critic_iter = checkpoint['critic_iter']
        gan_model = checkpoint['gan_model']
        disc_mode = checkpoint['disc_mode']
        gen_mode = checkpoint['gen_mode']
        num_durability_classes = checkpoint['num_durability_classes']
        num_weight_classes = checkpoint['num_weight_classes']
        zc_dim = checkpoint['zc_dim']
        input_dim = checkpoint['input_dim']
        hidden_dim = checkpoint['hidden_dim']
        embed_dim = checkpoint['embed_dim']
        output_dim = checkpoint['output_dim']
        num_heads = checkpoint['num_heads']
        height = checkpoint['height']
        width = checkpoint['width']
        loss_mode = checkpoint['loss_mode']
        datashape = checkpoint['datashape']
        optimizer = checkpoint['optimizer']
        log_path = checkpoint['log_path']
        
        
        device = args.device
        

            
        if disc_mode == 'vanila':
            disc = Discriminator_discrete_wgan(mode=loss_mode, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device) ##args
        elif disc_mode == 'selfattention':
            disc = SelfAttentionMLPDiscriminator(input_dim=input_dim, height=height, width=width, embed_dim=embed_dim, num_heads=num_heads, output_dim=output_dim, mode=loss_mode).to(device)  ##need to update

        if gen_mode == 'vanila':
            gen = Generator_discrete_wgan(zdim=zc_dim, input_dim=input_dim, output_dim=input_dim, num_durability_classes=num_durability_classes, num_weight_classes=num_weight_classes).to(device)
        elif gen_mode == 'selfattention':
            # zc_dim = args.zdim + num_durability_classes + num_weight_classes
            zc_dim = zc_dim + 1 + 1
            gen = SelfAttentionMLPGenerator(zc_dim=zc_dim, input_dim=input_dim, height=height, width=width, output_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads).to(device) ##need to update

        gen.apply(initialize_weights)
        disc.apply(initialize_weights)
        
        

        if optimizer == 'adam':    
            opt_gen = optim.Adam(gen.parameters(), lr=learning_rate_gen, betas=(opt_beta1, opt_beta2))
            opt_disc = optim.Adam(disc.parameters(), lr=learning_rate_disc, betas=(opt_beta1, opt_beta2))
        elif optimizer == 'adamw':    
            opt_gen = optim.AdamW(gen.parameters(), lr=learning_rate_gen, betas=(opt_beta1, opt_beta2))
            opt_disc = optim.AdamW(disc.parameters(), lr=learning_rate_disc, betas=(opt_beta1, opt_beta2))
        elif optimizer == 'rmsprop':    
            opt_gen = optim.RMSprop(gen.parameters(), lr=learning_rate_gen)
            opt_disc = optim.RMSprop(disc.parameters(), lr=learning_rate_disc)

        
        disc.load_state_dict(disc_state_dict)
        gen.load_state_dict(gen_state_dict)
        opt_disc.load_state_dict(opt_disc_state_dict)
        opt_gen.load_state_dict(opt_gen_state_dict)

        
        durability_emb_dict = {1.:0, 1.1:1, 1.2:2, 1.3:3, 1.4:4, 1.5:5, 1.6:6, 1.7:7, 1.8:8, 1.9:9}
        weight_emb_dict = {2.13:0, 2.14:1, 2.15:2, 2.16:3, 2.17:4, 2.18:5, 2.19:6, 2.2:7, 2.21:8, 2.22:9, 2.23:10, 2.24:11}
        
    
        
        ## 하나의 인퍼런스를 진행하기 위해 차원을 맞추었음
        
        durability = torch.tensor(durability_emb_dict[args.durability]).long().to(device)
        durability = durability.unsqueeze(0)
        
        weight = torch.tensor(weight_emb_dict[args.mass]).long().to(device)
        weight = weight.unsqueeze(0)

        disc.eval()
        gen.eval()
        
        if datashape == 'original':
            
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            # mask = mask.astype('uint8')
            strut1 = np.array(args.strut1)
            print(strut1)
            strut2 = np.array(args.strut2)
            print(strut2)
            strut1 = strut1[np.newaxis, np.newaxis, :]
            strut2 = strut2[np.newaxis, np.newaxis, :]
            mask[7,11,:] = strut1
            mask[5,14,:] = strut2
            jewon = mask
 
        
            transformation = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  ##incase I wanna scale image between [0,1] transforms.ToTensor() is suffice. if -1, 1, normalize.
            )
            jewon = transformation(jewon)
            

            jewon = jewon.view(-1, input_dim).to(device)
            
            ##making mask. when there are values, make them 1
            mask = torch.tensor(mask, dtype=torch.float32)
            mask = mask > 0  ##if larger than 0, it will turn True
            mask = mask.long()  ## True becomes 1, False becomes 0
            mask = mask.view(-1, input_dim).to(device)
        

            
            isExist = os.path.exists(args.gen_img_path)
            if not isExist:
               # Create a new directory because it does not exist
               os.makedirs(args.gen_img_path)
               print("The new directory is created!")
        
            with torch.no_grad():
                for i in range(args.num_gen_image):
                    

                    noise = torch.randn(1, zc_dim).to(device)
                    fake = gen(noise, durability, weight, jewon)#.reshape(-1, 3, 10, 30)
                    fake = fake.reshape(3, height, width)

                    unnormalize_gen_img = UnNormalize((0.5,), (0.5,))(fake)
                    inversed_gen_img = UnToTensor(unnormalize_gen_img)
                    
                    reshaped_jewon=jewon.reshape(3, height, width)
                    
                    unnormalize_jewon = UnNormalize((0.5,), (0.5,))(reshaped_jewon)
                    untotensored_jewon = UnToTensor(unnormalize_jewon)
                    final_original_jewon = untotensored_jewon
                    
                    inversed_gen_img[:, [7, 5], [11, 14]]  = final_original_jewon[:, [7, 5], [11, 14]] 
                    
                    
                    inversed_gen_img = inversed_gen_img.transpose((1, 2, 0))
                    inversed_gen_img = Image.fromarray(inversed_gen_img.astype(np.uint8))
                    # inversed_gen_img.save(f'{args.gen_img_path}/{i}.png')  
                    inversed_gen_img.save(f'{args.gen_img_path}/{i}.bmp')  


            
        elif datashape == 'dropempty':
            
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            
            strut1 = np.array(args.strut1)
            print(strut1)
            strut2 = np.array(args.strut2)
            print(strut2)
            strut1 = strut1[np.newaxis, np.newaxis, :]
            strut2 = strut2[np.newaxis, np.newaxis, :]
            
            mask[7,7,:] = strut1
            mask[5,10,:] = strut2
            jewon = mask

        
            transformation = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  ##incase I wanna scale image between [0,1] transforms.ToTensor() is suffice. if -1, 1, normalize.
            )
            jewon = transformation(jewon)
            
            jewon = jewon.view(-1, input_dim).to(device)
            
            
            ##making mask. when there are values, make them 1
            mask = torch.tensor(mask, dtype=torch.float32)
            mask = mask > 0  ##if larger than 0, it will turn True
            mask = mask.long()  ## True becomes 1, False becomes 0
            mask = mask.view(-1, input_dim).to(device)
        

            
            isExist = os.path.exists(args.gen_img_path)
            if not isExist:
               # Create a new directory because it does not exist
               os.makedirs(args.gen_img_path)
               print("The new directory is created!")
               
            with torch.no_grad():
                for i in range(args.num_gen_image):
                    

                    noise = torch.randn(1, zc_dim).to(device)
                    
   
                    fake = gen(noise, durability, weight, jewon)#.reshape(-1, 3, 10, 30)
                    fake = fake.reshape(3, height, width)
                    unnormalize_gen_img = UnNormalize((0.5,), (0.5,))(fake)
                    inversed_gen_img = UnToTensor(unnormalize_gen_img)
                    
                    
                    
                    ##2
                    mask = np.zeros((3, 10, 30), dtype=np.uint8)
                    
                    
                    final_original_fake = mask
                    
                    final_original_fake[:, :, 0:6] = inversed_gen_img[:, :, 0:6]
                    final_original_fake[:, :, 10:16] = inversed_gen_img[:, :, 6:12]
                    final_original_fake[:, :, 20:22] = inversed_gen_img[:, :, 12:14]
                    

                    reshaped_jewon=jewon.reshape(3, height, width)
                    
            
                    unnormalize_jewon = UnNormalize((0.5,), (0.5,))(reshaped_jewon)
                    untotensored_jewon = UnToTensor(unnormalize_jewon)

                    final_original_jewon = np.zeros((3, 10, 30), dtype=np.uint8)
                    final_original_jewon[:, :, 0:6] = untotensored_jewon[:, :, 0:6]
                    final_original_jewon[:, :, 10:16] = untotensored_jewon[:, :, 6:12]
                    final_original_jewon[:, :, 20:22] = untotensored_jewon[:, :, 12:14]
                    
                    final_original_fake[:, [7, 5], [11, 14]]  = final_original_jewon[:, [7, 5], [11, 14]] 
                    
                    final_original_fake = final_original_fake.transpose((1, 2, 0))
                    final_original_fake = Image.fromarray(final_original_fake.astype(np.uint8))
                    final_original_fake.save(f'{args.gen_img_path}/{i}.bmp')

    

    
    elif model_name == 'infogan_model.pt':
    
        checkpoint = torch.load(f'{args.model_path}')
        
        
        disc_state_dict =  checkpoint['disc_state_dict']
        gen_state_dict =  checkpoint['gen_state_dict']
        dhead_state_dict = checkpoint['dhead_state_dict']
        qhead_state_dict = checkpoint['qhead_state_dict']
        opt_disc =  checkpoint['opt_disc']
        opt_gen =  checkpoint['opt_gen']
        learning_rate_gen = checkpoint['learning_rate_gen']
        learning_rate_disc =  checkpoint['learning_rate_disc']
        batch_size =  checkpoint['batch_size']
        epochs =  checkpoint['epochs']
        constraint_lambda =  checkpoint['constraint_lambda']
        lambda_gp =  checkpoint['lambda_gp']
        critic_iter = checkpoint['critic_iter']
        opt_beta1 = checkpoint['opt_beta1']
        opt_beta2 = checkpoint['opt_beta2']
        gan_model = checkpoint['gan_model']
        disc_mode = checkpoint['disc_mode']
        gen_mode = checkpoint['gen_mode']
        num_durability_classes = checkpoint['num_durability_classes']
        num_weight_classes = checkpoint['num_weight_classes']
        zc_dim = checkpoint['zc_dim']
        input_dim = checkpoint['input_dim']
        hidden_dim = checkpoint['hidden_dim']
        embed_dim = checkpoint['embed_dim']
        output_dim = checkpoint['output_dim']
        num_heads = checkpoint['num_heads']
        height = checkpoint['height']
        width = checkpoint['width']
        loss_mode = checkpoint['loss_mode']
        datashape = checkpoint['datashape']
        optimizer = checkpoint['optimizer']
        log_path = checkpoint['log_path']
        
        
        device = args.device
        
        

    
        if args.gan_model == 'infogan':
            
            if args.disc_mode == 'vanila':
                disc = InfoGANMLPDiscriminator(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim).to(device)
            elif args.disc_mode == 'selfattention':
                disc = InfoGANSelfAttentionMLDiscriminator(input_dim=input_dim, height=height, width=width, embed_dim=embed_dim, output_dim=hidden_dim, num_heads=num_heads).to(device)

            if args.gen_mode == 'infogan':
                # zc_dim = args.zdim + num_durability_classes + num_weight_classes
                gen = InfoGANMLPGenerator(zc_dim=zc_dim, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim).to(device)
            elif args.gen_mode == 'selfattention':
                # zc_dim = args.zdim + num_durability_classes + num_weight_classes
                gen = InfoGANSelfAttentionMLPGenerator(zc_dim=zc_dim, input_dim=input_dim, height=height, width=width, output_dim=input_dim, num_heads=num_heads).to(device)
            
            dhead = InfoGANMLPDHead(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, mode=loss_mode).to(device) ##input_dim은 InfoGANSelfAttentionMLDiscriminator의 아웃풋이기 때문에 hidden_dim이 된다
            qhead = InfoGANMLPQHead_discrete(input_dim=hidden_dim, hidden_dim=hidden_dim, num_class_c1=num_durability_classes, num_class_c2=num_weight_classes).to(device)

            gen.apply(initialize_weights)
            disc.apply(initialize_weights)
            dhead.apply(initialize_weights)
            qhead.apply(initialize_weights)##똑같다    



        if optimizer == 'adam':    
            opt_gen = optim.Adam([{'params': gen.parameters()}, {'params': qhead.parameters()}], lr=args.learning_rate_gen, betas=(args.opt_beta1, args.opt_beta2))
            opt_disc = optim.Adam([{'params': disc.parameters()}, {'params': dhead.parameters()}], lr=args.learning_rate_disc, betas=(args.opt_beta1, args.opt_beta2))    
        elif optimizer == 'adamw':    
            opt_gen = optim.AdamW([{'params': gen.parameters()}, {'params': qhead.parameters()}], lr=args.learning_rate_gen, betas=(args.opt_beta1, args.opt_beta2))
            opt_disc = optim.AdamW([{'params': disc.parameters()}, {'params': dhead.parameters()}], lr=args.learning_rate_disc, betas=(args.opt_beta1, args.opt_beta2))    
        elif optimizer == 'rmsprop':    
            opt_gen = optim.RMSprop([{'params': gen.parameters()}, {'params': qhead.parameters()}], lr=args.learning_rate_gen)
            opt_disc = optim.RMSprop([{'params': disc.parameters()}, {'params': dhead.parameters()}], lr=args.learning_rate_disc)  


        disc.load_state_dict(disc_state_dict)
        gen.load_state_dict(gen_state_dict)
        dhead.load_state_dict(dhead_state_dict)
        qhead.load_state_dict(qhead_state_dict)
        
        
        opt_disc.load_state_dict(opt_disc)
        opt_gen.load_state_dict(opt_gen)     
        
        
        
        gen.eval()
        disc.eval()       
        dhead.eval()
        qhead.eval()

        
        
        if datashape == 'original':
            

            noise_durability = np.random.randint(0, num_durability_classes, 1)
            durability_labels = torch.tensor(noise_durability, requires_grad=False).to(device)
            durability_labels = F.one_hot(durability_labels.to(torch.int64), num_classes= num_durability_classes)*(1-0.1) + 0.1/num_durability_classes # label smoothning
            
            noise_weight = np.random.randint(0, num_weight_classes, 1)
            weight_labels = torch.tensor(noise_weight, requires_grad=False).to(device)
            weight_labels = F.one_hot(weight_labels.to(torch.int64), num_classes= num_weight_classes)*(1-0.1) + 0.1/num_weight_classes # label smoothning 
            
            final_noise = torch.cat((noise, durability_labels, weight_labels), dim =1)
            
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            strut1 = np.array(args.strut1)
            print(strut1)
            strut2 = np.array(args.strut2)
            print(strut2)
            strut1 = strut1[np.newaxis, np.newaxis, :]
            strut2 = strut2[np.newaxis, np.newaxis, :]
            mask[7,11,:] = strut1
            mask[5,14,:] = strut2
            jewon = mask
  
        
            transformation = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  ##incase I wanna scale image between [0,1] transforms.ToTensor() is suffice. if -1, 1, normalize.
            )
            jewon = transformation(jewon)
            

            jewon = jewon.view(-1, input_dim).to(device)
            
            ##making mask. when there are values, make them 1
            mask = torch.tensor(mask, dtype=torch.float32)
            mask = mask > 0  ##if larger than 0, it will turn True
            mask = mask.long()  ## True becomes 1, False becomes 0
            mask = mask.view(-1, input_dim).to(device)
        

        
            
            
            isExist = os.path.exists(args.gen_img_path)
            if not isExist:
               # Create a new directory because it does not exist
               os.makedirs(args.gen_img_path)
               print("The new directory is created!")
        
            with torch.no_grad():
                for i in range(args.num_gen_image):
                    

                    fake = gen(final_noise, jewon)
                    fake = fake.reshape(3, height, width)
                    unnormalize_gen_img = UnNormalize((0.5,), (0.5,))(fake)
                    inversed_gen_img = UnToTensor(unnormalize_gen_img)
                    
                    reshaped_jewon=jewon.reshape(3, height, width)
                    
            
                    unnormalize_jewon = UnNormalize((0.5,), (0.5,))(reshaped_jewon)
                    untotensored_jewon = UnToTensor(unnormalize_jewon)
                    final_original_jewon = untotensored_jewon
                    
                    
                    inversed_gen_img[:, [7, 5], [11, 14]]  = final_original_jewon[:, [7, 5], [11, 14]] 
                    
                    inversed_gen_img = inversed_gen_img.transpose((1, 2, 0))
                    inversed_gen_img = Image.fromarray(inversed_gen_img.astype(np.uint8))
                    # inversed_gen_img.save(f'{args.gen_img_path}/{i}.png') 
                    inversed_gen_img.save(f'{args.gen_img_path}/{i}.bmp') 

                    
        elif datashape == 'dropempty':
            
   
            # Ground truth labels for Q model
            noise_durability = np.random.randint(0, num_durability_classes, 1)
            durability_labels = torch.tensor(noise_durability, requires_grad=False).to(device)
            durability_labels = F.one_hot(durability_labels.to(torch.int64), num_classes= num_durability_classes)*(1-0.1) + 0.1/num_durability_classes # label smoothning
            
            #for Q
            noise_weight = np.random.randint(0, num_weight_classes, 1)
            weight_labels = torch.tensor(noise_weight, requires_grad=False).to(device)
            weight_labels = F.one_hot(weight_labels.to(torch.int64), num_classes= num_weight_classes)*(1-0.1) + 0.1/num_weight_classes # label smoothning 
            
            final_noise = torch.cat((noise, durability_labels, weight_labels), dim =1)
            
            
        
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            strut1 = np.array(args.strut1)
            print(strut1)
            strut2 = np.array(args.strut2)
            print(strut2)
            strut1 = strut1[np.newaxis, np.newaxis, :]
            strut2 = strut2[np.newaxis, np.newaxis, :]
            mask[7,11,:] = strut1
            mask[5,14,:] = strut2
            jewon = mask
   
        
            transformation = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  ##incase I wanna scale image between [0,1] transforms.ToTensor() is suffice. if -1, 1, normalize.
            )
            jewon = transformation(jewon)
            
            jewon = jewon.view(-1, input_dim).to(device)
            
            ##making mask. when there are values, make them 1
            mask = torch.tensor(mask, dtype=torch.float32)
            mask = mask > 0  ##if larger than 0, it will turn True
            mask = mask.long()  ## True becomes 1, False becomes 0
            mask = mask.view(-1, input_dim).to(device)
        
            

            
            isExist = os.path.exists(args.gen_img_path)
            if not isExist:
               # Create a new directory because it does not exist
               os.makedirs(args.gen_img_path)
               print("The new directory is created!")
        
            with torch.no_grad():
                for i in range(args.num_gen_image):

                    fake = gen(final_noise, jewon)
                    fake = fake.reshape(3, height, width)
                    unnormalize_gen_img = UnNormalize((0.5,), (0.5,))(fake)
                    inversed_gen_img = UnToTensor(unnormalize_gen_img)
                    
                    mask = np.zeros((3, 10, 30), dtype=np.uint8)
                    

                    final_original_fake = mask
                    
                    
                    final_original_fake[:, :, 0:6] = inversed_gen_img[:, :, 0:6]
                    final_original_fake[:, :, 10:16] = inversed_gen_img[:, :, 6:12]
                    final_original_fake[:, :, 20:22] = inversed_gen_img[:, :, 12:14]      
                    
                    reshaped_jewon=jewon.reshape(3, height, width)
                    
            
                    unnormalize_jewon = UnNormalize((0.5,), (0.5,))(reshaped_jewon)
                    untotensored_jewon = UnToTensor(unnormalize_jewon)

                    final_original_jewon = np.zeros((3, 10, 30), dtype=np.uint8)
                    final_original_jewon[:, :, 0:6] = untotensored_jewon[:, :, 0:6]
                    final_original_jewon[:, :, 10:16] = untotensored_jewon[:, :, 6:12]
                    final_original_jewon[:, :, 20:22] = untotensored_jewon[:, :, 12:14]
                    
                    final_original_fake[:, [7, 5], [11, 14]]  = final_original_jewon[:, [7, 5], [11, 14]] 
                    
                    
                    final_original_fake = final_original_fake.transpose((1, 2, 0))
                    final_original_fake = Image.fromarray(final_original_fake.astype(np.uint8))
                    final_original_fake.save(f'{args.gen_img_path}/{i}.bmp')  

            
            
               
                
if __name__ == "__main__":
    main(args)        
                

    