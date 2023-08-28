# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 18:40:39 2023
References: 
https://github.com/Natsu6767/InfoGAN-PyTorch
https://medium.com/mlearning-ai/infogan-learning-to-generate-controllable-images-from-scratch-pytorch-31a49ffc7b98  
@author: hojun
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
import torch
import torch.nn as nn
from torch import tensor
import pandas as pd
import os
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty, save_checkpoint, load_checkpoint, gradient_penalty_onlyimage1d, torch_initialize_weights, NormalNLLLoss, UnNormalize, UnToTensor
from models import  Generator_discrete_wgan, Discriminator_discrete_wgan, initialize_weights, InfoGANMLPDiscriminator, InfoGANSelfAttentionMLDiscriminator, InfoGANSelfAttentionMLPGenerator, InfoGANMLPGenerator, InfoGANMLPQHead_discrete, InfoGANMLPDHead, SelfAttentionMLPDiscriminator, SelfAttentionMLPGenerator
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
# import cv2
import random
from layers import activation_layer, MLPLayers
from dataset import HyundaiDataset_discrete, DropEmptyHyundaiDataset_discrete
import numpy as np
import argparse
from tqdm import tqdm
from torchvision.utils import save_image
import time 
from loggers import log





parser = argparse.ArgumentParser(description='Hyundai image generation task')
parser.add_argument("--data_path", '-dp', default=r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\data", type=str,  help = 'path of directory that contains image data')          # extra value default="decimal"
parser.add_argument("--datashape", '-ds', default='dropempty', type=str, help='weather to use original data or data dropping useless black pixel columns. Choose between original and dropempty.')
parser.add_argument("--gan_model", '-gm', default='vanila', type=str, help='kinds of GAN model to use. Choose between infogan and vanila. Vanila means the simple plain GAN')
parser.add_argument("--loss_mode", '-lm', default='wgan', type=str, help='what kind of loss function to use. Choose among vanila, wgan and wgangp.')
parser.add_argument("--gen_mode", '-ge', default='vanila', type=str, help='what kind of generator to use. Choose between vanila and selfattention')
parser.add_argument("--disc_mode", '-dm', default='vanila', type=str, help='what kind of discriminator to use. Choose between vanila and selfattention')
parser.add_argument("--log_path", '-tld', default=r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\CGAN\logs\vanilagan_wganloss_vanila_dropempty_150", type=str, help='Path of directory to save all the logs')
parser.add_argument("--output_path", '-op', default=r"C:\Users\hojun\OneDrive\Desktop\Hojun\Hyundai\CGAN\logs\vanilagan_wganloss_vanila_dropempty_150\output", type=str, help='Path of directory to save some generated images after training')
parser.add_argument("--device", '-d', default='cpu', type=str, help='which device to use. cpu or cuda')
parser.add_argument("--learning_rate_gen", '-lrg', default=0.0002, type=float, help='learning rate of generator')
parser.add_argument("--learning_rate_disc", '-lrd', default=0.0002, type=float, help='learning rate of discriminator')
parser.add_argument("--zdim", '-z', default=100, type=int, help='noise vector dimension')
parser.add_argument("--height", '-he', default=10, type=int, help='size of y-axis (height). It is 10 in this task')
parser.add_argument("--width", '-wi', default=14, type=int, help='size of x-axis (width). It must be between 14 and 30 according to the datashape chosen')
parser.add_argument("--input_dim", '-id', default=420, type=int, help='flatten image size. In original image case, 900 = 10 * 30 * 3.')
parser.add_argument("--hidden_dim", '-hd', default=256, type=int, help='Hidden vector dimension for layers')
parser.add_argument("--embed_dim", '-ed', default=32, type=int, help='embedding dimension for self-attention layer.')
parser.add_argument("--output_dim", '-od', default=1, type=int, help='final output dimension for model to output. If discriminator, output dimension would be 1. If generator, output dimension would be same size as input image size')
parser.add_argument("--num_heads", '-nh', default=2, type=int, help='number of multi-head in multi-head self attention layer. If embed_dim is 32, num_heads must be a valid denominator such as 2 and 4. ')

parser.add_argument("--optimizer", '-opt', default='adam', type=str, help='which optimizer to use between adam, adamw, and rmsprop')
parser.add_argument("--weight_clip", '-wc', default=0.01, type=float, help='weight clipping technique to prevent gradient explosion in case of WGAN in loss_mode')
parser.add_argument("--batch_size", '-b', default=512, type=int, help='Mini batch size for dataloader')
parser.add_argument("--epochs", '-e', default=150, type=int, help='Number of epoch in trainig loop')
parser.add_argument("--constraint_lambda", '-cl', default=100, type=int,  help='Weight of fixed pixel constraint in the loss function. Can be float and int both')
parser.add_argument("--lambda_gp", '-lg', default=10, type=int, help='Weight of WGAN-GP in loss term')
parser.add_argument("--critic_iter", '-ci', default=5, type=int, help='number of iteration ratio for critic (discriminator) compared to generator. If 5, discriminator will be updated 5 times when generator is updated 1 time.')
parser.add_argument("--opt_beta1", '-ob1', default=0.0, type=float, help='tuple of 2 floats for betas of generator optimizer if optimizer is either adam or adamw. Example: (0.0, 0.9)')
parser.add_argument("--opt_beta2", '-ob2', default=0.9, type=float,  help='tuple of 2 floats for betas of discriminator optimizer if optimizer is either adam or adamw. Example: (0.0, 0.9)')

args = parser.parse_args()


# Set random seed for reproducibility.



def main(args):
    

    
    # set a logger file
    isExist = os.path.exists(f'{args.log_path}')
    if not isExist:
        # os.umask(0)
        # os.makedirs(f'{args.tensorboard_log_dir}/{args.model_name}/', mode=0o777)
        # Create a new directory because it does not exist
        os.makedirs(f'{args.log_path}')
        print("The new directory is created!")
        
    
    logger = log(path=f'{args.log_path}', file='log_summary') ##logging path 설정하여 나중에 파일 저장.
    
    print('Start working on the script...!')
    logger.info('Start working on the script...!')
    logger.info('='*64)
    
    logger.info(f'The training information & parameters: {args}') ##logging all argparse information
    logger.info('='*64)
    
    seed = 2023
    random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Random Seed: , {seed}")
    
    # python train_fcgan_PCGANISCloss_WGAN-GP.py --data_path '/home/hannah/Hyundai/data' --norm_mean 0.5 --norm_stdv 0.5 --device 'cuda' --learning_rate_gen 0.0005 --learning_rate_disc 0.0005 --zdim 100 --image_dim 900 --batch_size 512 --epochs 100 --constraint_lambda 10 --lambda_gp 10 --critic_iter 5 --opt_beta1 0.0 --opt_beta2 0.9 --disc_mode 'wgan' --tensorboard_log_dir '/home/hannah/Hyundai/CGAN/logs' --model_dir '/home/hannah/Hyundai/CGAN/models' --model_name 'wgan_test'
    
    
    transformation = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  ##incase I wanna scale image between [0,1] transforms.ToTensor() is suffice. if -1, 1, normalize.
    )
    
    
   
        
    if args.datashape == 'original':
        dataset = HyundaiDataset_discrete(args.data_path, transform=transformation) # 간단 예시
    elif args.datashape == 'dropempty':
        dataset = DropEmptyHyundaiDataset_discrete(args.data_path, transform=transformation) # 간단 예시            
    else:
        raise ValueError('choose datashape between original and dropempty')    


    
    print('dataset loading done')
    logger.info('Dataset loading done')
    logger.info('='*64)
    '''
    WGAN
    1. RMSPROP
    2. WEIGHT CLIP
    3. No sigmoid in discriminator
    4. No batchnorm but layernorm in discriminator
    '''
    
    # Hyperparameters etc.
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # lr = 5e-5
    # z_dim = 100
    # image_dim = 10 * 30 * 3  # 784
    # batch_size = 512
    # num_epochs = 100
    num_durability_classes = len(dataset.durability_emb_dict)
    num_weight_classes = len(dataset.weight_emb_dict)
    
    step = 0
    # modelname = f'fcgan_constraint_lambda-{args.constraint_lambda}_lambda_gp_-{args.lambda_gp}_critic_iter-{args.critic_iter}_norm_mean-{args.norm_mean}_norm_stdv-{args.norm_stdv}_learning_rate_gen-{args.learning_rate_gen}_learning_rate_disc-{args.learning_rate_disc}_zdim-{args.zdim}_image_dim-{args.image_dim}_batch_size-{args.batch_size}_epochs-{args.epochs}_critic_iter-{args.critic_iter}_opt_gen_betas-{args.opt_gen_betas}_opt_disc_betas-{args.opt_disc_betas}_disc_mode-{args.disc_mode}'
    # criterion = nn.BCELoss()
    
    
    
    writer_fake = SummaryWriter(f"{args.log_path}/tensorboard_records/fake")
    writer_real = SummaryWriter(f"{args.log_path}/tensorboard_records/real")
    

    
    
    
    
    
    if args.gan_model == 'infogan':
        
        if args.disc_mode == 'vanila':
            disc = InfoGANMLPDiscriminator(input_dim=args.input_dim, hidden_dim= args.hidden_dim, output_dim=args.hidden_dim).to(args.device)
        elif args.disc_mode == 'selfattention':
            disc = InfoGANSelfAttentionMLDiscriminator(input_dim=args.input_dim, height=args.height, width=args.width, embed_dim=args.embed_dim, output_dim=args.hidden_dim, num_heads=args.num_heads).to(args.device)

        if args.gen_mode == 'vanila':
            zc_dim = args.zdim + num_durability_classes + num_weight_classes
            gen = InfoGANMLPGenerator(zc_dim=zc_dim, input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.input_dim).to(args.device)
        elif args.gen_mode == 'selfattention':
            zc_dim = args.zdim + num_durability_classes + num_weight_classes
            gen = InfoGANSelfAttentionMLPGenerator(zc_dim=zc_dim, input_dim=args.input_dim, height=args.height, width=args.width, output_dim=args.input_dim, num_heads=args.num_heads).to(args.device)
        
        dhead = InfoGANMLPDHead(input_dim=args.hidden_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, mode=args.loss_mode).to(args.device) ##input_dim은 InfoGANSelfAttentionMLDiscriminator의 아웃풋이기 때문에 hidden_dim이 된다
        qhead = InfoGANMLPQHead_discrete(input_dim=args.hidden_dim, hidden_dim=args.hidden_dim, num_class_c1=num_durability_classes, num_class_c2=num_weight_classes).to(args.device)

        gen.apply(initialize_weights)
        disc.apply(initialize_weights)
        dhead.apply(initialize_weights)
        qhead.apply(initialize_weights)##똑같다                  
        

    # elif args.gan_model == 'wgan' or args.gan_model == 'wgangp':
    elif args.gan_model == 'vanila':
        
        if args.disc_mode == 'vanila':
            disc = Discriminator_discrete_wgan(mode=args.loss_mode, input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim).to(args.device) ##args
        elif args.disc_mode == 'selfattention':
            disc = SelfAttentionMLPDiscriminator(input_dim=args.input_dim, height=args.height, width=args.width, embed_dim=args.embed_dim, num_heads=args.num_heads, output_dim=args.output_dim, mode=args.loss_mode).to(args.device)  ##need to update

        if args.gen_mode == 'vanila':
            zc_dim = args.zdim
            gen = Generator_discrete_wgan(zdim=zc_dim, input_dim=args.input_dim, output_dim=args.input_dim, num_durability_classes=num_durability_classes, num_weight_classes=num_weight_classes).to(args.device)
        elif args.gen_mode == 'selfattention':
            # zc_dim = args.zdim + num_durability_classes + num_weight_classes
            zc_dim = args.zdim + 1 + 1
            gen = SelfAttentionMLPGenerator(zc_dim=zc_dim, input_dim=args.input_dim, height=args.height, width=args.width, output_dim=args.input_dim, embed_dim=args.embed_dim, num_heads=args.num_heads).to(args.device) ##need to update

        gen.apply(initialize_weights)
        disc.apply(initialize_weights)
            
        
    # elif args.gan_model == 'mlp':
    #     # elif args.disc_mode == 'mlp_sum':
    #     gen.apply(initialize_weights)
    #     disc.apply(initialize_weights)
    #     pass
    
    else:
        raise ValueError('You must choose gan_model among infogan, wga, wgangp, mlp')
        

    


    # Loss for discrimination between real and fake images.
    criterionD = nn.BCELoss()
    # Loss for discrete latent code.
    criterionQ_dis = nn.CrossEntropyLoss()
    # Loss for continuous latent code.
    criterionQ_con = NormalNLLLoss()
    
    

    
    
    
    if args.gan_model == 'infogan':
        if args.optimizer == 'adam':    
            opt_gen = optim.Adam([{'params': gen.parameters()}, {'params': qhead.parameters()}], lr=args.learning_rate_gen, betas=(args.opt_beta1, args.opt_beta2))
            opt_disc = optim.Adam([{'params': disc.parameters()}, {'params': dhead.parameters()}], lr=args.learning_rate_disc, betas=(args.opt_beta1, args.opt_beta2))    
        elif args.optimizer == 'adamw':    
            opt_gen = optim.AdamW([{'params': gen.parameters()}, {'params': qhead.parameters()}], lr=args.learning_rate_gen, betas=(args.opt_beta1, args.opt_beta2))
            opt_disc = optim.AdamW([{'params': disc.parameters()}, {'params': dhead.parameters()}], lr=args.learning_rate_disc, betas=(args.opt_beta1, args.opt_beta2))    
        elif args.optimizer == 'rmsprop':    
            opt_gen = optim.RMSprop([{'params': gen.parameters()}, {'params': qhead.parameters()}], lr=args.learning_rate_gen)
            opt_disc = optim.RMSprop([{'params': disc.parameters()}, {'params': dhead.parameters()}], lr=args.learning_rate_disc)    
        
    # elif args.gan_model == 'wgan' or 'wgangp':
    elif args.gan_model == 'vanila':
        if args.optimizer == 'adam':    
            opt_gen = optim.Adam(gen.parameters(), lr=args.learning_rate_gen, betas=(args.opt_beta1, args.opt_beta2))
            opt_disc = optim.Adam(disc.parameters(), lr=args.learning_rate_disc, betas=(args.opt_beta1, args.opt_beta2))
        elif args.optimizer == 'adamw':    
            opt_gen = optim.AdamW(gen.parameters(), lr=args.learning_rate_gen, betas=(args.opt_beta1, args.opt_beta2))
            opt_disc = optim.AdamW(disc.parameters(), lr=args.learning_rate_disc, betas=(args.opt_beta1, args.opt_beta2))
        elif args.optimizer == 'rmsprop':    
            opt_gen = optim.RMSprop(gen.parameters(), lr=args.learning_rate_gen)
            opt_disc = optim.RMSprop(disc.parameters(), lr=args.learning_rate_disc)
        
        



    # discrete_c1_dim = num_durability_classes
    # discrete_c2_dim = num_weight_classes
    # idx = np.arange(discrete_c1_dim).repeat(10)
    
    # dis_c1 = torch.zeros(100, params['num_dis_c'], params['dis_c_dim'], device=device)
    # for i in range(params['num_dis_c']):
    #     dis_c[torch.arange(0, 100), i, idx] = 1.0

    # dis_c = dis_c.view(100, -1, 1, 1)

    # fixed_noise = torch.cat((fixed_noise, dis_c), dim=1)
    

    

    

    # fixed_noise = torch.randn((args.batch_size, args.zdim)).to(args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    torch.cuda.is_available()
    # jw = jewon.detach().cpu().numpy()
    print('training start!')
    logger.info('training start!') ##logging all argparse information
    logger.info('='*64)
    s=time.time()
    
    generated_image_path = f'{args.output_path}'
    
    if args.gan_model == 'infogan' and args.loss_mode == 'vanila':
        logger.info('InfoGAN model with vanila loss term training initiated!')
        for epoch in tqdm(range(args.epochs)):
            for batch_idx, (real, durability, weight, jewon, mask, imagename) in tqdm(enumerate(loader)):
            #     break
            # break
                ##taking out inputs
                real = real.view(-1, args.input_dim).to(args.device)
                durability = durability.long().to(args.device)
                weight = weight.long().to(args.device)
                jewon = jewon.view(-1, args.input_dim).to(args.device)
                mask = mask.view(-1, args.input_dim).to(args.device)
                batch_size = real.shape[0]
                # real.shape
                # durability.shape
                
                gen.train()
                disc.train()

    
                ##generating noise and codes
                noise = torch.randn(batch_size, args.zdim).to(args.device)
                
                # Ground truth labels for Q model
                noise_durability = np.random.randint(0, num_durability_classes, args.batch_size)
                durability_labels = torch.tensor(noise_durability, requires_grad=False).to(args.device)
                durability_labels = F.one_hot(durability_labels.to(torch.int64), num_classes= num_durability_classes)*(1-0.1) + 0.1/num_durability_classes # label smoothning
                
                #for Q
                noise_weight = np.random.randint(0, num_weight_classes, args.batch_size)
                weight_labels = torch.tensor(noise_weight, requires_grad=False).to(args.device)
                weight_labels = F.one_hot(weight_labels.to(torch.int64), num_classes= num_weight_classes)*(1-0.1) + 0.1/num_weight_classes # label smoothning 
                
                final_noise = torch.cat((noise, durability_labels, weight_labels), dim =1)
                
                ## generating fake image
                fake = gen(final_noise, jewon)
                
                
                ##loss for real data discriminator
                opt_disc.zero_grad()
                disc_real_intermediate = disc(real)
                disk_real_output = dhead(disc_real_intermediate).view(-1)
                lossD_real = criterionD(disk_real_output, torch.ones_like(disk_real_output))
                lossD_real.backward()
                
                ##loss for fake data discriminator
                disc_fake_intermediate = disc(fake)
                disk_fake_output = dhead(disc_fake_intermediate).view(-1)
                lossD_fake = criterionD(disk_fake_output, torch.ones_like(disk_fake_output))
                lossD_fake.backward()
                
                # Total Loss for the discriminator
                D_loss = lossD_real + lossD_fake
                # Update parameters
                opt_disc.step()
                
                opt_gen.zero_grad()
            
                ##generating noise and codes
                noise = torch.randn(batch_size, args.zdim).to(args.device)
                
                # Ground truth labels for Q model
                noise_durability = np.random.randint(0, num_durability_classes, args.batch_size)
                durability_labels = torch.tensor(noise_durability, requires_grad=False).to(args.device)
                durability_labels = F.one_hot(durability_labels.to(torch.int64), num_classes= num_durability_classes)*(1-0.1) + 0.1/num_durability_classes # label smoothning
                
                #for Q
                noise_weight = np.random.randint(0, num_weight_classes, args.batch_size)
                weight_labels = torch.tensor(noise_weight, requires_grad=False).to(args.device)
                weight_labels = F.one_hot(weight_labels.to(torch.int64), num_classes= num_weight_classes)*(1-0.1) + 0.1/num_weight_classes # label smoothning 
                
                final_noise = torch.cat((noise, durability_labels, weight_labels), dim =1)
                
                gen_fake = gen(final_noise, jewon)
                
                # Fake data treated as real.
                fake_to_be_real_intermediate = disc(gen_fake)
                fake_to_be_real_output = dhead(fake_to_be_real_intermediate).view(-1)
                lossG = criterionD(fake_to_be_real_output, torch.ones_like(fake_to_be_real_output))
    
                durability_logits, weight_logits = qhead(fake_to_be_real_intermediate)
                
                
                # target = torch.LongTensor(idx).to(device)
                # Calculating loss for discrete latent code.
                discrete_loss = 0
                discrete_loss += criterionQ_dis(durability_logits, durability)
                discrete_loss += criterionQ_dis(weight_logits, weight)
    
                # total loss for generator.
                ##https://aigong.tistory.com/433
                G_loss = lossG + discrete_loss - args.constraint_lambda*torch.norm(jewon - torch.mul(mask, gen_fake)) #+ con_loss 
                # Calculate gradients.
                G_loss.backward()
                # Update parameters.
                opt_gen.step()
    
                if (batch_idx % 100 == 0 and batch_idx > 0) or (batch_idx == len(loader) -1 and batch_idx > 0):
                    logger.info(f"Epoch [{epoch}/{args.epochs}] Batch {batch_idx}/{len(loader)} \
                                          Loss D: {D_loss:.4f}, loss G: {G_loss:.4f}"
                                )
                    # print(
                    #     f"Epoch [{epoch}/{args.epochs}] Batch {batch_idx}/{len(loader)} \
                    #           Loss D: {D_loss:.4f}, loss G: {G_loss:.4f}"
                    # )
            
    
                    with torch.no_grad():
                        
                        gen.eval()
                        disc.eval()
            
                        fake = gen(final_noise, jewon)#.reshape(-1, 3, 10, 30)
                        fake = fake.reshape(-1, 3, args.height, args.width)
                        real = real.reshape(-1, 3, args.height, args.width)
                        jewon=jewon.reshape(-1, 3, args.height, args.width)
                        

                        if args.datashape == 'dropempty':
                            
                            jewon=jewon.reshape(-1, 3, args.height, args.width)
                            ##억지로 실제 데이터의 fixed pixel들을 생성해낸 이미지에도 붙여넣고 싶으면 아래 코드를 언커맨트하면된다.
                            # fake[:,:, [7, 5], [7, 10]] = jewon[:,:, [7, 5], [7, 10]]
                            
                            unnormalize_real = UnNormalize((0.5,), (0.5,))(real)
                            untotensored_real = UnToTensor(unnormalize_real)
                            
                            final_original_real = np.zeros((batch_size, 3, 10, 30), dtype=np.uint8)
                            final_original_real[:, :, :, 0:6] = untotensored_real[:, :, :, 0:6]
                            final_original_real[:, :, :, 10:16] = untotensored_real[:, :, :, 6:12]
                            final_original_real[:, :, :, 20:22] = untotensored_real[:, :, :, 12:14]
                            final_original_real[:,:,[7, 5], [11, 14]]

                
                            unnormalize_fake = UnNormalize((0.5,), (0.5,))(fake)
                            untotensored_fake = UnToTensor(unnormalize_fake)
    
                            final_original_fake = np.zeros((batch_size, 3, 10, 30), dtype=np.uint8)
                            final_original_fake[:, :, :, 0:6] = untotensored_fake[:, :, :, 0:6]
                            final_original_fake[:, :, :, 10:16] = untotensored_fake[:, :, :, 6:12]
                            final_original_fake[:, :, :, 20:22] = untotensored_fake[:, :, :, 12:14]
                            

                            unnormalize_jewon = UnNormalize((0.5,), (0.5,))(jewon)
                            untotensored_jewon = UnToTensor(unnormalize_jewon)
    
                            final_original_jewon = np.zeros((batch_size, 3, 10, 30), dtype=np.uint8)
                            final_original_jewon[:, :, :, 0:6] = untotensored_jewon[:, :, :, 0:6]
                            final_original_jewon[:, :, :, 10:16] = untotensored_jewon[:, :, :, 6:12]
                            final_original_jewon[:, :, :, 20:22] = untotensored_jewon[:, :, :, 12:14]
                            
                            
                            
                        elif args.datashape == 'original':
                            # real = real.reshape(-1, 3, args.height, args.width)
                            # # real[:,:, [7, 5], [11, 14]]
                            
                            # fake = fake.reshape(-1, 3, args.height, args.width)
                            # # fake[:,:, [7, 5], [11, 14]]
                            
                            # jewon=jewon.reshape(-1, 3, args.height, args.width)
                            # jewon[:,:, [7, 5], [11, 14]]
                            
                            ##억지로 실제 데이터의 fixed pixel들을 생성해낸 이미지에도 붙여넣고 싶으면 아래 코드를 언커맨트하면된다.
                            # fake[:,:, [7, 5], [11, 14]] = jewon[:,:, [7, 5], [11, 14]]
                            
                            
                            unnormalize_real = UnNormalize((0.5,), (0.5,))(real)
                            untotensored_real = UnToTensor(unnormalize_real)
                            final_original_real = untotensored_real
                            
                            unnormalize_fake = UnNormalize((0.5,), (0.5,))(fake)
                            untotensored_fake = UnToTensor(unnormalize_fake)
                            final_original_fake = untotensored_fake
                            
                            unnormalize_jewon = UnNormalize((0.5,), (0.5,))(jewon)
                            untotensored_jewon = UnToTensor(unnormalize_jewon)
                            final_original_jewon = untotensored_jewon
                            
                        final_original_real = torch.tensor(final_original_real, dtype=torch.float32)
                        final_original_fake = torch.tensor(final_original_fake, dtype=torch.float32)     
                        final_original_jewon = torch.tensor(final_original_jewon, dtype=torch.float32)  
                        
                        
                        img_grid_fake = torchvision.utils.make_grid(final_original_fake[20:30], normalize=True)#, normalize=True)
                        img_grid_real = torchvision.utils.make_grid(final_original_real[20:30], normalize=True)#, normalize=True)
        
                        writer_fake.add_image(
                            "Hyundai Fake Images", img_grid_fake, global_step=step
                        )
                        writer_real.add_image(
                            "Hyundai Real Images", img_grid_real, global_step=step
                        )
                        step += 1
                        
                        if epoch==args.epochs-1 and batch_idx == len(loader) -1:
                            print('generation is working')
                            
                            GenDirisExist = os.path.exists(generated_image_path)
                            FakeDirisExist = os.path.exists(f'{generated_image_path}/fake')
                            RealDirisExist = os.path.exists(f'{generated_image_path}/real')
                            JewonDirisExist = os.path.exists(f'{generated_image_path}/jewon')
                            if not GenDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(generated_image_path)
                                logger.info("The generated image output directory is created!")
                                
                            if not FakeDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(f'{generated_image_path}/fake')
                                logger.info("The generated image output directory of fake images is created!")
                                
                            
                            if not RealDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(f'{generated_image_path}/real')
                                logger.info("The generated image output directory of real images is created!")
                                
                                                            
                            if not JewonDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(f'{generated_image_path}/jewon')
                                logger.info("The generated image output directory of jewon images is created!")
                                
                            final_original_real = final_original_real.numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
                            final_original_fake = final_original_fake.numpy().transpose((0, 2, 3, 1)).astype(np.uint8)  
                            final_original_jewon = final_original_jewon.numpy().transpose((0, 2, 3, 1)).astype(np.uint8)  
                            
                            
                            
                            for i, fake in enumerate(final_original_fake):
                                fake = Image.fromarray(fake)
                                fake.save(f'{generated_image_path}/fake/fake_{i}.bmp')
                                
                            logger.info("The generated image output directory is created!")
                            for i, real in enumerate(final_original_real):
                                real = Image.fromarray(real)
                                real.save(f'{generated_image_path}/real/real_{i}.bmp')
                                
                            for i, jewon in enumerate(final_original_jewon):
                                jewon = Image.fromarray(jewon)
                                jewon.save(f'{generated_image_path}/jewon/jewon_{i}.bmp')
    
        if args.gan_model == 'infogan':
            model_saving_full_path = f"{args.log_path}/infogan_model.pt"
            
            torch.save({
                        'disc_state_dict': disc.state_dict(),
                        'gen_state_dict': gen.state_dict(),
                        'dhead_state_dict' : dhead.state_dict(),
                        'qhead_state_dict' : qhead.state_dict(),
                        'opt_disc': opt_disc.state_dict(),
                        'opt_gen': opt_gen.state_dict(),
                        # 'norm_mean': args.norm_mean,
                        # 'norm_stdv': args.norm_stdv,
                        'learning_rate_gen': args.learning_rate_gen,
                        'learning_rate_disc': args.learning_rate_disc,
                        'batch_size': args.batch_size,
                        'epochs': args.epochs,
                        'constraint_lambda': args.constraint_lambda,
                        'lambda_gp': args.lambda_gp,
                        'critic_iter': args.critic_iter,
                        'opt_beta1': args.opt_beta1,
                        'opt_beta2': args.opt_beta2,
                        'gan_model': args.gan_model,
                        'disc_mode': args.disc_mode,
                        'gen_mode': args.gen_mode,
                        'num_durability_classes': num_durability_classes,
                        'num_weight_classes': num_weight_classes,
                        'zc_dim': zc_dim,
                        'input_dim': args.input_dim,
                        'hidden_dim': args.hidden_dim,
                        'embed_dim': args.embed_dim,
                        'output_dim': args.output_dim,
                        'num_heads': args.num_heads,
                        'height': args.height,
                        'width': args.width,
                        'loss_mode': args.loss_mode,
                        'datashape': args.datashape,
                        'optimizer': args.optimizer,
                        # 'model_path': args.model_path,
                        # 'model_name': args.model_name,
                        'log_path': args.log_path}
                        , model_saving_full_path)
            
        elif args.gan_model == 'vanila':
            model_saving_full_path = f"{args.log_path}/vanila_model.pt"
            
            torch.save({
                        'disc_state_dict': disc.state_dict(),
                        'gen_state_dict': gen.state_dict(),
                        'opt_disc': opt_disc.state_dict(),
                        'opt_gen': opt_gen.state_dict(),
                        # 'norm_mean': args.norm_mean,
                        # 'norm_stdv': args.norm_stdv,
                        'learning_rate_gen': args.learning_rate_gen,
                        'learning_rate_disc': args.learning_rate_disc,
                        'batch_size': args.batch_size,
                        'epochs': args.epochs,
                        'constraint_lambda': args.constraint_lambda,
                        'lambda_gp': args.lambda_gp,
                        'critic_iter': args.critic_iter,
                        'opt_beta1': args.opt_beta1,
                        'opt_beta2': args.opt_beta2,
                        'gan_model': args.gan_model,
                        'disc_mode': args.disc_mode,
                        'gen_mode': args.gen_mode,
                        'num_durability_classes': num_durability_classes,
                        'num_weight_classes': num_weight_classes,
                        'zc_dim': zc_dim,
                        'input_dim': args.input_dim,
                        'hidden_dim': args.hidden_dim,
                        'embed_dim': args.embed_dim,
                        'output_dim': args.output_dim,
                        'num_heads': args.num_heads,
                        'height': args.height,
                        'width': args.width,
                        'loss_mode': args.loss_mode,
                        'datashape': args.datashape,
                        'optimizer': args.optimizer,
                        # 'model_path': args.model_path,
                        # 'model_name': args.model_name,
                        'log_path': args.log_path}
                        , model_saving_full_path)
        
        
        print('training took ', f'{time.time()-s}', ' seconds')
        logger.info(f'training took {time.time()-s} seconds') 
        print('Job done ')
        logger.info('training loop done!') 
    
                    
                        
                        
    elif args.gan_model == 'infogan' and args.loss_mode == 'wgan':
        logger.info('InfoGAN model with wgan loss term training initiated!')
        for epoch in tqdm(range(args.epochs)):
            for batch_idx, (real, durability, weight, jewon, mask, imagename) in tqdm(enumerate(loader)):
            #     break
            # break
                ##taking out inputs
                real = real.view(-1, args.input_dim).to(args.device)
                durability = durability.long().to(args.device)
                weight = weight.long().to(args.device)
                jewon = jewon.view(-1, args.input_dim).to(args.device)
                mask = mask.view(-1, args.input_dim).to(args.device)
                batch_size = real.shape[0]
                # real.shape
                # durability.shape
                
                gen.train()
                disc.train()
        
    

                for _ in range(args.critic_iter):
                    # break
                    ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                    
                    ##for normal model
                    # noise = torch.randn(batch_size, args.zdim).to(args.device)
                    # final_noise = torch.cat((noise, durability.unsqueeze(1), weight.unsqueeze(1)), dim =1)
                    # fake = gen(noise, durability, weight, noise)
                    
                        
                    ##generating noise and codes
                    noise = torch.randn(batch_size, args.zdim).to(args.device)
                    
                    # Ground truth labels for Q model
                    noise_durability = np.random.randint(0, num_durability_classes, args.batch_size)
                    durability_labels = torch.tensor(noise_durability, requires_grad=False).to(args.device)
                    durability_labels = F.one_hot(durability_labels.to(torch.int64), num_classes= num_durability_classes)*(1-0.1) + 0.1/num_durability_classes # label smoothning
                    
                    #for Q
                    noise_weight = np.random.randint(0, num_weight_classes, args.batch_size)
                    weight_labels = torch.tensor(noise_weight, requires_grad=False).to(args.device)
                    weight_labels = F.one_hot(weight_labels.to(torch.int64), num_classes= num_weight_classes)*(1-0.1) + 0.1/num_weight_classes # label smoothning 
                    
                    final_noise = torch.cat((noise, durability_labels, weight_labels), dim =1)
                    fake = gen(final_noise, jewon)
                    

                    disc_real = disc(real).view(-1)
                    disc_fake = disc(fake).view(-1)
                    
                    
                    loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake))
                    # lossD_real = criterion(disc_real, torch.ones_like(disc_real))        
                    # lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                    # lossD = (lossD_real + lossD_fake) / 2
                    
                    # gp = gradient_penalty(disc, durability_weight, jewon, real_image, fake_image, device=device)
                    disc.zero_grad()
                    loss_disc.backward(retain_graph=True)
                    opt_disc.step()
                        # clip critic weights between -0.01, 0.01
                    for p in disc.parameters():
                        p.data.clamp_(-args.weight_clip, args.weight_clip)

                
                ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                # where the second option of maximizing doesn't suffer from
                # saturating gradients     
                # gen_fake = gen(final_noise, jewon)
                
                # Fake data treated as real.
                
                noise = torch.randn(batch_size, args.zdim).to(args.device)
                
                # Ground truth labels for Q model
                noise_durability = np.random.randint(0, num_durability_classes, args.batch_size)
                durability_labels = torch.tensor(noise_durability, requires_grad=False).to(args.device)
                durability_labels = F.one_hot(durability_labels.to(torch.int64), num_classes= num_durability_classes)*(1-0.1) + 0.1/num_durability_classes # label smoothning
                
                #for Q
                noise_weight = np.random.randint(0, num_weight_classes, args.batch_size)
                weight_labels = torch.tensor(noise_weight, requires_grad=False).to(args.device)
                weight_labels = F.one_hot(weight_labels.to(torch.int64), num_classes= num_weight_classes)*(1-0.1) + 0.1/num_weight_classes # label smoothning 
                
                final_noise = torch.cat((noise, durability_labels, weight_labels), dim =1)
                
                gen_fake = gen(final_noise, jewon)
                
                
                
                fake_to_be_real_intermediate = disc(gen_fake)  
                fake_to_be_real_output = dhead(fake_to_be_real_intermediate).view(-1)
                
                # fake_to_be_real_output.shape
                # lossG = criterionD(fake_to_be_real_output, torch.ones_like(fake_to_be_real_output))
    
                durability_logits, weight_logits = qhead(fake_to_be_real_intermediate)
                
                # target = torch.LongTensor(idx).to(device)
                # Calculating loss for discrete latent code.
                discrete_loss = 0
                discrete_loss += criterionQ_dis(durability_logits, durability)
                discrete_loss += criterionQ_dis(weight_logits, weight)
    
                # total loss for generator.
                ##https://aigong.tistory.com/433
                loss_gen = -(torch.mean(disc(gen_fake))) + discrete_loss  -args.constraint_lambda*torch.norm(jewon - torch.mul(mask, gen_fake))
                # G_loss = lossG + discrete_loss - args.constraint_lambda*torch.norm(jewon - torch.mul(mask, gen_fake)) #+ con_loss 
                # Calculate gradients.
                loss_gen.backward()
                # Update parameters.
                opt_gen.step()
                gen.zero_grad()
                
                
                # # loss_gen = -(torch.mean(disc(gen_fake)) + LAMDA*torch.norm(real - torch.mul(noised_jewon, fake)))       
                # loss_gen = -(torch.mean(disc(gen_fake)))  -args.constraint_lambda*torch.norm(jewon - torch.mul(mask, gen_fake))    
                # # lossG = criterion(output, torch.ones_like(output)) + 1*torch.norm(real - torch.mul(noised_jewon, fake))
                # gen.zero_grad()
                # loss_gen.backward()
                # opt_gen.step()

                if (batch_idx % 100 == 0 and batch_idx > 0) or (batch_idx == len(loader) -1 and batch_idx > 0):
                    print(
                        f"Epoch [{epoch}/{args.epochs}] Batch {batch_idx}/{len(loader)} \
                              Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                    )
                    with torch.no_grad():
                        
                        gen.eval()
                        disc.eval()
            
                        fake = gen(final_noise, jewon)#.reshape(-1, 3, 10, 30)
                        fake = fake.reshape(-1, 3, args.height, args.width)
                        real = real.reshape(-1, 3, args.height, args.width)
                        jewon=jewon.reshape(-1, 3, args.height, args.width)
                        

                        if args.datashape == 'dropempty':
                            
                            jewon=jewon.reshape(-1, 3, args.height, args.width)
                            ##억지로 실제 데이터의 fixed pixel들을 생성해낸 이미지에도 붙여넣고 싶으면 아래 코드를 언커맨트하면된다.
                            # fake[:,:, [7, 5], [7, 10]] = jewon[:,:, [7, 5], [7, 10]]
                            
                            unnormalize_real = UnNormalize((0.5,), (0.5,))(real)
                            untotensored_real = UnToTensor(unnormalize_real)
                            
                            final_original_real = np.zeros((batch_size, 3, 10, 30), dtype=np.uint8)
                            final_original_real[:, :, :, 0:6] = untotensored_real[:, :, :, 0:6]
                            final_original_real[:, :, :, 10:16] = untotensored_real[:, :, :, 6:12]
                            final_original_real[:, :, :, 20:22] = untotensored_real[:, :, :, 12:14]
                            final_original_real[:,:,[7, 5], [11, 14]]

                
                            unnormalize_fake = UnNormalize((0.5,), (0.5,))(fake)
                            untotensored_fake = UnToTensor(unnormalize_fake)
    
                            final_original_fake = np.zeros((batch_size, 3, 10, 30), dtype=np.uint8)
                            final_original_fake[:, :, :, 0:6] = untotensored_fake[:, :, :, 0:6]
                            final_original_fake[:, :, :, 10:16] = untotensored_fake[:, :, :, 6:12]
                            final_original_fake[:, :, :, 20:22] = untotensored_fake[:, :, :, 12:14]
                            

                            unnormalize_jewon = UnNormalize((0.5,), (0.5,))(jewon)
                            untotensored_jewon = UnToTensor(unnormalize_jewon)
    
                            final_original_jewon = np.zeros((batch_size, 3, 10, 30), dtype=np.uint8)
                            final_original_jewon[:, :, :, 0:6] = untotensored_jewon[:, :, :, 0:6]
                            final_original_jewon[:, :, :, 10:16] = untotensored_jewon[:, :, :, 6:12]
                            final_original_jewon[:, :, :, 20:22] = untotensored_jewon[:, :, :, 12:14]
                            
                            
                            
                        elif args.datashape == 'original':
                            # real = real.reshape(-1, 3, args.height, args.width)
                            # # real[:,:, [7, 5], [11, 14]]
                            
                            # fake = fake.reshape(-1, 3, args.height, args.width)
                            # # fake[:,:, [7, 5], [11, 14]]
                            
                            # jewon=jewon.reshape(-1, 3, args.height, args.width)
                            # jewon[:,:, [7, 5], [11, 14]]
                            
                            ##억지로 실제 데이터의 fixed pixel들을 생성해낸 이미지에도 붙여넣고 싶으면 아래 코드를 언커맨트하면된다.
                            # fake[:,:, [7, 5], [11, 14]] = jewon[:,:, [7, 5], [11, 14]]
                            
                            
                            unnormalize_real = UnNormalize((0.5,), (0.5,))(real)
                            untotensored_real = UnToTensor(unnormalize_real)
                            final_original_real = untotensored_real
                            
                            unnormalize_fake = UnNormalize((0.5,), (0.5,))(fake)
                            untotensored_fake = UnToTensor(unnormalize_fake)
                            final_original_fake = untotensored_fake
                            
                            unnormalize_jewon = UnNormalize((0.5,), (0.5,))(jewon)
                            untotensored_jewon = UnToTensor(unnormalize_jewon)
                            final_original_jewon = untotensored_jewon
                            
                        final_original_real = torch.tensor(final_original_real, dtype=torch.float32)
                        final_original_fake = torch.tensor(final_original_fake, dtype=torch.float32)     
                        final_original_jewon = torch.tensor(final_original_jewon, dtype=torch.float32)  
                        
                        
                        img_grid_fake = torchvision.utils.make_grid(final_original_fake[20:30], normalize=True)#, normalize=True)
                        img_grid_real = torchvision.utils.make_grid(final_original_real[20:30], normalize=True)#, normalize=True)
        
                        writer_fake.add_image(
                            "Hyundai Fake Images", img_grid_fake, global_step=step
                        )
                        writer_real.add_image(
                            "Hyundai Real Images", img_grid_real, global_step=step
                        )
                        step += 1
                        
                        if epoch==args.epochs-1 and batch_idx == len(loader) -1:
                            print('generation is working')
                            
                            GenDirisExist = os.path.exists(generated_image_path)
                            FakeDirisExist = os.path.exists(f'{generated_image_path}/fake')
                            RealDirisExist = os.path.exists(f'{generated_image_path}/real')
                            JewonDirisExist = os.path.exists(f'{generated_image_path}/jewon')
                            if not GenDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(generated_image_path)
                                logger.info("The generated image output directory is created!")
                                
                            if not FakeDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(f'{generated_image_path}/fake')
                                logger.info("The generated image output directory of fake images is created!")
                                
                            
                            if not RealDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(f'{generated_image_path}/real')
                                logger.info("The generated image output directory of real images is created!")
                                
                                                            
                            if not JewonDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(f'{generated_image_path}/jewon')
                                logger.info("The generated image output directory of jewon images is created!")
                                
                            final_original_real = final_original_real.numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
                            final_original_fake = final_original_fake.numpy().transpose((0, 2, 3, 1)).astype(np.uint8)  
                            final_original_jewon = final_original_jewon.numpy().transpose((0, 2, 3, 1)).astype(np.uint8)  
                            
                            
                            
                            for i, fake in enumerate(final_original_fake):
                                fake = Image.fromarray(fake)
                                fake.save(f'{generated_image_path}/fake/fake_{i}.bmp')
                                
                            logger.info("The generated image output directory is created!")
                            for i, real in enumerate(final_original_real):
                                real = Image.fromarray(real)
                                real.save(f'{generated_image_path}/real/real_{i}.bmp')
                                
                            for i, jewon in enumerate(final_original_jewon):
                                jewon = Image.fromarray(jewon)
                                jewon.save(f'{generated_image_path}/jewon/jewon_{i}.bmp')
                                
        
        if args.gan_model == 'infogan':
            model_saving_full_path = f"{args.log_path}/infogan_model.pt"
            
            torch.save({
                        'disc_state_dict': disc.state_dict(),
                        'gen_state_dict': gen.state_dict(),
                        'dhead_state_dict' : dhead.state_dict(),
                        'qhead_state_dict' : qhead.state_dict(),
                        'opt_disc': opt_disc.state_dict(),
                        'opt_gen': opt_gen.state_dict(),
                        # 'norm_mean': args.norm_mean,
                        # 'norm_stdv': args.norm_stdv,
                        'learning_rate_gen': args.learning_rate_gen,
                        'learning_rate_disc': args.learning_rate_disc,
                        'batch_size': args.batch_size,
                        'epochs': args.epochs,
                        'constraint_lambda': args.constraint_lambda,
                        'lambda_gp': args.lambda_gp,
                        'critic_iter': args.critic_iter,
                        'opt_beta1': args.opt_beta1,
                        'opt_beta2': args.opt_beta2,
                        'gan_model': args.gan_model,
                        'disc_mode': args.disc_mode,
                        'gen_mode': args.gen_mode,
                        'num_durability_classes': num_durability_classes,
                        'num_weight_classes': num_weight_classes,
                        'zc_dim': zc_dim,
                        'input_dim': args.input_dim,
                        'hidden_dim': args.hidden_dim,
                        'embed_dim': args.embed_dim,
                        'output_dim': args.output_dim,
                        'num_heads': args.num_heads,
                        'height': args.height,
                        'width': args.width,
                        'loss_mode': args.loss_mode,
                        'datashape': args.datashape,
                        'optimizer': args.optimizer,
                        # 'model_path': args.model_path,
                        # 'model_name': args.model_name,
                        'log_path': args.log_path}
                        , model_saving_full_path)
            
        elif args.gan_model == 'vanila':
            model_saving_full_path = f"{args.log_path}/vanila_model.pt"
            
            torch.save({
                        'disc_state_dict': disc.state_dict(),
                        'gen_state_dict': gen.state_dict(),
                        'opt_disc': opt_disc.state_dict(),
                        'opt_gen': opt_gen.state_dict(),
                        # 'norm_mean': args.norm_mean,
                        # 'norm_stdv': args.norm_stdv,
                        'learning_rate_gen': args.learning_rate_gen,
                        'learning_rate_disc': args.learning_rate_disc,
                        'batch_size': args.batch_size,
                        'epochs': args.epochs,
                        'constraint_lambda': args.constraint_lambda,
                        'lambda_gp': args.lambda_gp,
                        'critic_iter': args.critic_iter,
                        'opt_beta1': args.opt_beta1,
                        'opt_beta2': args.opt_beta2,
                        'gan_model': args.gan_model,
                        'disc_mode': args.disc_mode,
                        'gen_mode': args.gen_mode,
                        'num_durability_classes': num_durability_classes,
                        'num_weight_classes': num_weight_classes,
                        'zc_dim': zc_dim,
                        'input_dim': args.input_dim,
                        'hidden_dim': args.hidden_dim,
                        'embed_dim': args.embed_dim,
                        'output_dim': args.output_dim,
                        'num_heads': args.num_heads,
                        'height': args.height,
                        'width': args.width,
                        'loss_mode': args.loss_mode,
                        'datashape': args.datashape,
                        'optimizer': args.optimizer,
                        # 'model_path': args.model_path,
                        # 'model_name': args.model_name,
                        'log_path': args.log_path}
                        , model_saving_full_path)
        
        
        print('training took ', f'{time.time()-s}', ' seconds')
        logger.info(f'training took {time.time()-s} seconds') 
        print('Job done ')
        logger.info('training loop done!') 
        
                            

    elif args.gan_model == 'infogan' and args.loss_mode == 'wgangp':
        logger.info('InfoGAN model with wgangp loss term training initiated!')
        for epoch in tqdm(range(args.epochs)):
            for batch_idx, (real, durability, weight, jewon, mask, imagename) in tqdm(enumerate(loader)):
            #     break
            # break
                ##taking out inputs
                real = real.view(-1, args.input_dim).to(args.device)
                durability = durability.long().to(args.device)
                weight = weight.long().to(args.device)
                jewon = jewon.view(-1, args.input_dim).to(args.device)
                mask = mask.view(-1, args.input_dim).to(args.device)
                batch_size = real.shape[0]
                # real.shape
                # durability.shape
        
                gen.train()
                disc.train()

                for _ in range(args.critic_iter):
                    # break
                    ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                    
                    ##for normal model
                    # noise = torch.randn(batch_size, args.zdim).to(args.device)
                    # final_noise = torch.cat((noise, durability.unsqueeze(1), weight.unsqueeze(1)), dim =1)
                    # fake = gen(noise, durability, weight, noise)
                    
                        
                    ##generating noise and codes
                    noise = torch.randn(batch_size, args.zdim).to(args.device)
                    
                    # Ground truth labels for Q model
                    noise_durability = np.random.randint(0, num_durability_classes, args.batch_size)
                    durability_labels = torch.tensor(noise_durability, requires_grad=False).to(args.device)
                    durability_labels = F.one_hot(durability_labels.to(torch.int64), num_classes= num_durability_classes)*(1-0.1) + 0.1/num_durability_classes # label smoothning
                    
                    #for Q
                    noise_weight = np.random.randint(0, num_weight_classes, args.batch_size)
                    weight_labels = torch.tensor(noise_weight, requires_grad=False).to(args.device)
                    weight_labels = F.one_hot(weight_labels.to(torch.int64), num_classes= num_weight_classes)*(1-0.1) + 0.1/num_weight_classes # label smoothning 
                    
                    final_noise = torch.cat((noise, durability_labels, weight_labels), dim =1)
                    fake = gen(final_noise, jewon)
                    

                    disc_real = disc(real).view(-1)
                    disc_fake = disc(fake).view(-1)
                    gp = gradient_penalty_onlyimage1d(disc, real, fake, device=args.device)
                    
                    loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + args.lambda_gp * gp
                    # lossD_real = criterion(disc_real, torch.ones_like(disc_real))        
                    # lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                    # lossD = (lossD_real + lossD_fake) / 2
                    
                    # gp = gradient_penalty(disc, durability_weight, jewon, real_image, fake_image, device=device)
                    disc.zero_grad()
                    loss_disc.backward(retain_graph=True)
                    opt_disc.step()
                        # clip critic weights between -0.01, 0.01
                    # for p in disc.parameters():
                    #     p.data.clamp_(-args.weight_clip, args.weight_clip)

                
                ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                # where the second option of maximizing doesn't suffer from
                # saturating gradients     
                # gen_fake = gen(final_noise, jewon)
                
                # Fake data treated as real.
                
                noise = torch.randn(batch_size, args.zdim).to(args.device)
                
                # Ground truth labels for Q model
                noise_durability = np.random.randint(0, num_durability_classes, args.batch_size)
                durability_labels = torch.tensor(noise_durability, requires_grad=False).to(args.device)
                durability_labels = F.one_hot(durability_labels.to(torch.int64), num_classes= num_durability_classes)*(1-0.1) + 0.1/num_durability_classes # label smoothning
                
                #for Q
                noise_weight = np.random.randint(0, num_weight_classes, args.batch_size)
                weight_labels = torch.tensor(noise_weight, requires_grad=False).to(args.device)
                weight_labels = F.one_hot(weight_labels.to(torch.int64), num_classes= num_weight_classes)*(1-0.1) + 0.1/num_weight_classes # label smoothning 
                
                final_noise = torch.cat((noise, durability_labels, weight_labels), dim =1)
                
                gen_fake = gen(final_noise, jewon)
                
                
                
                fake_to_be_real_intermediate = disc(gen_fake)  
                fake_to_be_real_output = dhead(fake_to_be_real_intermediate).view(-1)
                
                # fake_to_be_real_output.shape
                # lossG = criterionD(fake_to_be_real_output, torch.ones_like(fake_to_be_real_output))
    
                durability_logits, weight_logits = qhead(fake_to_be_real_intermediate)
                
                # target = torch.LongTensor(idx).to(device)
                # Calculating loss for discrete latent code.
                discrete_loss = 0
                discrete_loss += criterionQ_dis(durability_logits, durability)
                discrete_loss += criterionQ_dis(weight_logits, weight)
    
                # total loss for generator.
                ##https://aigong.tistory.com/433
                loss_gen = -(torch.mean(disc(gen_fake))) + discrete_loss  -args.constraint_lambda*torch.norm(jewon - torch.mul(mask, gen_fake))
                # G_loss = lossG + discrete_loss - args.constraint_lambda*torch.norm(jewon - torch.mul(mask, gen_fake)) #+ con_loss 
                # Calculate gradients.
                loss_gen.backward()
                # Update parameters.
                opt_gen.step()
                gen.zero_grad()
                
                
                # # loss_gen = -(torch.mean(disc(gen_fake)) + LAMDA*torch.norm(real - torch.mul(noised_jewon, fake)))       
                # loss_gen = -(torch.mean(disc(gen_fake)))  -args.constraint_lambda*torch.norm(jewon - torch.mul(mask, gen_fake))    
                # # lossG = criterion(output, torch.ones_like(output)) + 1*torch.norm(real - torch.mul(noised_jewon, fake))
                # gen.zero_grad()
                # loss_gen.backward()
                # opt_gen.step()

                if (batch_idx % 100 == 0 and batch_idx > 0) or (batch_idx == len(loader) -1 and batch_idx > 0):
                    print(
                        f"Epoch [{epoch}/{args.epochs}] Batch {batch_idx}/{len(loader)} \
                              Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                    )
                    with torch.no_grad():
                        
                        gen.eval()
                        disc.eval()
            
                        fake = gen(final_noise, jewon)#.reshape(-1, 3, 10, 30)
                        fake = fake.reshape(-1, 3, args.height, args.width)
                        real = real.reshape(-1, 3, args.height, args.width)
                        jewon=jewon.reshape(-1, 3, args.height, args.width)
                        

                        if args.datashape == 'dropempty':
                            
                            jewon=jewon.reshape(-1, 3, args.height, args.width)
                            ##억지로 실제 데이터의 fixed pixel들을 생성해낸 이미지에도 붙여넣고 싶으면 아래 코드를 언커맨트하면된다.
                            # fake[:,:, [7, 5], [7, 10]] = jewon[:,:, [7, 5], [7, 10]]
                            
                            unnormalize_real = UnNormalize((0.5,), (0.5,))(real)
                            untotensored_real = UnToTensor(unnormalize_real)
                            
                            final_original_real = np.zeros((batch_size, 3, 10, 30), dtype=np.uint8)
                            final_original_real[:, :, :, 0:6] = untotensored_real[:, :, :, 0:6]
                            final_original_real[:, :, :, 10:16] = untotensored_real[:, :, :, 6:12]
                            final_original_real[:, :, :, 20:22] = untotensored_real[:, :, :, 12:14]
                            final_original_real[:,:,[7, 5], [11, 14]]

                
                            unnormalize_fake = UnNormalize((0.5,), (0.5,))(fake)
                            untotensored_fake = UnToTensor(unnormalize_fake)
    
                            final_original_fake = np.zeros((batch_size, 3, 10, 30), dtype=np.uint8)
                            final_original_fake[:, :, :, 0:6] = untotensored_fake[:, :, :, 0:6]
                            final_original_fake[:, :, :, 10:16] = untotensored_fake[:, :, :, 6:12]
                            final_original_fake[:, :, :, 20:22] = untotensored_fake[:, :, :, 12:14]
                            

                            unnormalize_jewon = UnNormalize((0.5,), (0.5,))(jewon)
                            untotensored_jewon = UnToTensor(unnormalize_jewon)
    
                            final_original_jewon = np.zeros((batch_size, 3, 10, 30), dtype=np.uint8)
                            final_original_jewon[:, :, :, 0:6] = untotensored_jewon[:, :, :, 0:6]
                            final_original_jewon[:, :, :, 10:16] = untotensored_jewon[:, :, :, 6:12]
                            final_original_jewon[:, :, :, 20:22] = untotensored_jewon[:, :, :, 12:14]
                            
                            
                            
                        elif args.datashape == 'original':
                            # real = real.reshape(-1, 3, args.height, args.width)
                            # # real[:,:, [7, 5], [11, 14]]
                            
                            # fake = fake.reshape(-1, 3, args.height, args.width)
                            # # fake[:,:, [7, 5], [11, 14]]
                            
                            # jewon=jewon.reshape(-1, 3, args.height, args.width)
                            # jewon[:,:, [7, 5], [11, 14]]
                            
                            ##억지로 실제 데이터의 fixed pixel들을 생성해낸 이미지에도 붙여넣고 싶으면 아래 코드를 언커맨트하면된다.
                            # fake[:,:, [7, 5], [11, 14]] = jewon[:,:, [7, 5], [11, 14]]
                            
                            
                            unnormalize_real = UnNormalize((0.5,), (0.5,))(real)
                            untotensored_real = UnToTensor(unnormalize_real)
                            final_original_real = untotensored_real
                            
                            unnormalize_fake = UnNormalize((0.5,), (0.5,))(fake)
                            untotensored_fake = UnToTensor(unnormalize_fake)
                            final_original_fake = untotensored_fake
                            
                            unnormalize_jewon = UnNormalize((0.5,), (0.5,))(jewon)
                            untotensored_jewon = UnToTensor(unnormalize_jewon)
                            final_original_jewon = untotensored_jewon
                            
                        final_original_real = torch.tensor(final_original_real, dtype=torch.float32)
                        final_original_fake = torch.tensor(final_original_fake, dtype=torch.float32)     
                        final_original_jewon = torch.tensor(final_original_jewon, dtype=torch.float32)  
                        
                        
                        img_grid_fake = torchvision.utils.make_grid(final_original_fake[20:30], normalize=True)#, normalize=True)
                        img_grid_real = torchvision.utils.make_grid(final_original_real[20:30], normalize=True)#, normalize=True)
        
                        writer_fake.add_image(
                            "Hyundai Fake Images", img_grid_fake, global_step=step
                        )
                        writer_real.add_image(
                            "Hyundai Real Images", img_grid_real, global_step=step
                        )
                        step += 1
                        
                        if epoch==args.epochs-1 and batch_idx == len(loader) -1:
                            print('generation is working')
                            
                            GenDirisExist = os.path.exists(generated_image_path)
                            FakeDirisExist = os.path.exists(f'{generated_image_path}/fake')
                            RealDirisExist = os.path.exists(f'{generated_image_path}/real')
                            JewonDirisExist = os.path.exists(f'{generated_image_path}/jewon')
                            if not GenDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(generated_image_path)
                                logger.info("The generated image output directory is created!")
                                
                            if not FakeDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(f'{generated_image_path}/fake')
                                logger.info("The generated image output directory of fake images is created!")
                                
                            
                            if not RealDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(f'{generated_image_path}/real')
                                logger.info("The generated image output directory of real images is created!")
                                
                                                            
                            if not JewonDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(f'{generated_image_path}/jewon')
                                logger.info("The generated image output directory of jewon images is created!")
                                
                            final_original_real = final_original_real.numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
                            final_original_fake = final_original_fake.numpy().transpose((0, 2, 3, 1)).astype(np.uint8)  
                            final_original_jewon = final_original_jewon.numpy().transpose((0, 2, 3, 1)).astype(np.uint8)  
                            
                            
                            
                            for i, fake in enumerate(final_original_fake):
                                fake = Image.fromarray(fake)
                                fake.save(f'{generated_image_path}/fake/fake_{i}.bmp')
                                
                            logger.info("The generated image output directory is created!")
                            for i, real in enumerate(final_original_real):
                                real = Image.fromarray(real)
                                real.save(f'{generated_image_path}/real/real_{i}.bmp')
                                
                            for i, jewon in enumerate(final_original_jewon):
                                jewon = Image.fromarray(jewon)
                                jewon.save(f'{generated_image_path}/jewon/jewon_{i}.bmp')
                                
                                
        if args.gan_model == 'infogan':
            model_saving_full_path = f"{args.log_path}/infogan_model.pt"
            
            torch.save({
                        'disc_state_dict': disc.state_dict(),
                        'gen_state_dict': gen.state_dict(),
                        'dhead_state_dict' : dhead.state_dict(),
                        'qhead_state_dict' : qhead.state_dict(),
                        'opt_disc': opt_disc.state_dict(),
                        'opt_gen': opt_gen.state_dict(),
                        # 'norm_mean': args.norm_mean,
                        # 'norm_stdv': args.norm_stdv,
                        'learning_rate_gen': args.learning_rate_gen,
                        'learning_rate_disc': args.learning_rate_disc,
                        'batch_size': args.batch_size,
                        'epochs': args.epochs,
                        'constraint_lambda': args.constraint_lambda,
                        'lambda_gp': args.lambda_gp,
                        'critic_iter': args.critic_iter,
                        'opt_beta1': args.opt_beta1,
                        'opt_beta2': args.opt_beta2,
                        'gan_model': args.gan_model,
                        'disc_mode': args.disc_mode,
                        'gen_mode': args.gen_mode,
                        'num_durability_classes': num_durability_classes,
                        'num_weight_classes': num_weight_classes,
                        'zc_dim': zc_dim,
                        'input_dim': args.input_dim,
                        'hidden_dim': args.hidden_dim,
                        'embed_dim': args.embed_dim,
                        'output_dim': args.output_dim,
                        'num_heads': args.num_heads,
                        'height': args.height,
                        'width': args.width,
                        'loss_mode': args.loss_mode,
                        'datashape': args.datashape,
                        'optimizer': args.optimizer,
                        # 'model_path': args.model_path,
                        # 'model_name': args.model_name,
                        'log_path': args.log_path}
                        , model_saving_full_path)
            
        elif args.gan_model == 'vanila':
            model_saving_full_path = f"{args.log_path}/vanila_model.pt"
            
            torch.save({
                        'disc_state_dict': disc.state_dict(),
                        'gen_state_dict': gen.state_dict(),
                        'opt_disc': opt_disc.state_dict(),
                        'opt_gen': opt_gen.state_dict(),
                        # 'norm_mean': args.norm_mean,
                        # 'norm_stdv': args.norm_stdv,
                        'learning_rate_gen': args.learning_rate_gen,
                        'learning_rate_disc': args.learning_rate_disc,
                        'batch_size': args.batch_size,
                        'epochs': args.epochs,
                        'constraint_lambda': args.constraint_lambda,
                        'lambda_gp': args.lambda_gp,
                        'critic_iter': args.critic_iter,
                        'opt_beta1': args.opt_beta1,
                        'opt_beta2': args.opt_beta2,
                        'gan_model': args.gan_model,
                        'disc_mode': args.disc_mode,
                        'gen_mode': args.gen_mode,
                        'num_durability_classes': num_durability_classes,
                        'num_weight_classes': num_weight_classes,
                        'zc_dim': zc_dim,
                        'input_dim': args.input_dim,
                        'hidden_dim': args.hidden_dim,
                        'embed_dim': args.embed_dim,
                        'output_dim': args.output_dim,
                        'num_heads': args.num_heads,
                        'height': args.height,
                        'width': args.width,
                        'loss_mode': args.loss_mode,
                        'datashape': args.datashape,
                        'optimizer': args.optimizer,
                        # 'model_path': args.model_path,
                        # 'model_name': args.model_name,
                        'log_path': args.log_path}
                        , model_saving_full_path)
        
        
        print('training took ', f'{time.time()-s}', ' seconds')
        logger.info(f'training took {time.time()-s} seconds') 
        print('Job done ')
        logger.info('training loop done!') 
    
              


    elif args.gan_model == 'vanila' and args.loss_mode == 'wgangp':                        

        logger.info('vanila GAN model with WGAN-GP loss term training initiated!')
        for epoch in tqdm(range(args.epochs)):
            for batch_idx, (real, durability, weight, jewon, mask, imagename) in tqdm(enumerate(loader)):
            #     break
            # break
                ##taking out inputs
                real = real.view(-1, args.input_dim).to(args.device)
                durability = durability.long().to(args.device)
                weight = weight.long().to(args.device)
                jewon = jewon.view(-1, args.input_dim).to(args.device)
                mask = mask.view(-1, args.input_dim).to(args.device)
                batch_size = real.shape[0]
                
                gen.train()
                disc.train()
                
                for _ in range(args.critic_iter):
                    # break
                    ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                    
                    if args.gen_mode == 'vanila':
                        noise = torch.randn(batch_size, args.zdim).to(args.device)
                        fake = gen(noise, durability, weight, jewon)
                    
                    
                    #for selfattention models
                    elif args.gen_mode == 'selfattention':
                        noise = torch.randn(batch_size, args.zdim).to(args.device)
                        final_input = torch.cat((noise, durability, weight), dim=1)
                        fake = gen(final_input, jewon)
                    
                    # noise = torch.randn(batch_size, args.zdim).to(args.device)
                    # fake = gen(noise, durability, weight, jewon)
                    disc_real = disc(real).view(-1)
                    disc_fake = disc(fake).view(-1)
                    gp = gradient_penalty_onlyimage1d(disc, real, fake, device=args.device)
                    ##gp = DCGAN_gradient_penalty() if you wanna try wgan gp not only on image data but also other additional data, try this function
                    loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + args.lambda_gp * gp
                    disc.zero_grad()
                    loss_disc.backward(retain_graph=True)
                    opt_disc.step()
        
                ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                # where the second option of maximizing doesn't suffer from
                # saturating gradients     
                
                    # if args.gen_mode == 'vanila':
                    #     # noise = torch.randn(batch_size, args.zdim).to(args.device)
                    #     gen_fake = gen(noise, durability, weight, jewon)
                    
                    
                    # #for selfattention models
                    # elif args.gen_mode == 'selfattention':
                    #     # noise = torch.randn(batch_size, args.zdim).to(args.device)
                    #     # final_input = torch.cat((noise, durability, weight), dim=1)
                    #     gen_fake = gen(final_input, jewon)
                        
                        
                # gen_fake = gen(noise, durability, weight, jewon)   
                # loss_gen = -(torch.mean(disc(gen_fake)))  -args.constraint_lambda*torch.norm(jewon - torch.mul(mask, gen_fake))
                loss_gen = -(torch.mean(disc(fake)))  -args.constraint_lambda*torch.norm(jewon - torch.mul(mask, fake))
        
                gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()
        
                if (batch_idx % 100 == 0 and batch_idx > 0) or (batch_idx == len(loader) -1 and batch_idx > 0):
                    logger.info(f"Epoch [{epoch}/{args.epochs}] Batch {batch_idx}/{len(loader)} \
                                          Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                                )
                    print(
                        f"Epoch [{epoch}/{args.epochs}] Batch {batch_idx}/{len(loader)} \
                              Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                    )
                    with torch.no_grad():
                        
                        gen.eval()
                        disc.eval()
                        
                        if args.gen_mode == 'vanila':
                            # noise = torch.randn(batch_size, args.zdim).to(args.device)
                            fake = gen(noise, durability, weight, jewon)
                        
                        
                        #for selfattention models
                        elif args.gen_mode == 'selfattention':
                            # noise = torch.randn(batch_size, args.zdim).to(args.device)
                            # final_input = torch.cat((noise, durability, weight), dim=1)
                            fake = gen(final_input, jewon)
                            
                        # fake = gen(final_noise, jewon)#.reshape(-1, 3, 10, 30)
                        fake = fake.reshape(-1, 3, args.height, args.width)
                        real = real.reshape(-1, 3, args.height, args.width)
                        jewon=jewon.reshape(-1, 3, args.height, args.width)
                        

                        if args.datashape == 'dropempty':
                            
                            jewon=jewon.reshape(-1, 3, args.height, args.width)
                            ##억지로 실제 데이터의 fixed pixel들을 생성해낸 이미지에도 붙여넣고 싶으면 아래 코드를 언커맨트하면된다.
                            # fake[:,:, [7, 5], [7, 10]] = jewon[:,:, [7, 5], [7, 10]]
                            
                            unnormalize_real = UnNormalize((0.5,), (0.5,))(real)
                            untotensored_real = UnToTensor(unnormalize_real)
                            
                            final_original_real = np.zeros((batch_size, 3, 10, 30), dtype=np.uint8)
                            final_original_real[:, :, :, 0:6] = untotensored_real[:, :, :, 0:6]
                            final_original_real[:, :, :, 10:16] = untotensored_real[:, :, :, 6:12]
                            final_original_real[:, :, :, 20:22] = untotensored_real[:, :, :, 12:14]
                            final_original_real[:,:,[7, 5], [11, 14]]

                
                            unnormalize_fake = UnNormalize((0.5,), (0.5,))(fake)
                            untotensored_fake = UnToTensor(unnormalize_fake)
    
                            final_original_fake = np.zeros((batch_size, 3, 10, 30), dtype=np.uint8)
                            final_original_fake[:, :, :, 0:6] = untotensored_fake[:, :, :, 0:6]
                            final_original_fake[:, :, :, 10:16] = untotensored_fake[:, :, :, 6:12]
                            final_original_fake[:, :, :, 20:22] = untotensored_fake[:, :, :, 12:14]
                            

                            unnormalize_jewon = UnNormalize((0.5,), (0.5,))(jewon)
                            untotensored_jewon = UnToTensor(unnormalize_jewon)
    
                            final_original_jewon = np.zeros((batch_size, 3, 10, 30), dtype=np.uint8)
                            final_original_jewon[:, :, :, 0:6] = untotensored_jewon[:, :, :, 0:6]
                            final_original_jewon[:, :, :, 10:16] = untotensored_jewon[:, :, :, 6:12]
                            final_original_jewon[:, :, :, 20:22] = untotensored_jewon[:, :, :, 12:14]
                            
                            
                            
                        elif args.datashape == 'original':
                            # real = real.reshape(-1, 3, args.height, args.width)
                            # # real[:,:, [7, 5], [11, 14]]
                            
                            # fake = fake.reshape(-1, 3, args.height, args.width)
                            # # fake[:,:, [7, 5], [11, 14]]
                            
                            # jewon=jewon.reshape(-1, 3, args.height, args.width)
                            # jewon[:,:, [7, 5], [11, 14]]
                            
                            ##억지로 실제 데이터의 fixed pixel들을 생성해낸 이미지에도 붙여넣고 싶으면 아래 코드를 언커맨트하면된다.
                            # fake[:,:, [7, 5], [11, 14]] = jewon[:,:, [7, 5], [11, 14]]
                            
                            
                            unnormalize_real = UnNormalize((0.5,), (0.5,))(real)
                            untotensored_real = UnToTensor(unnormalize_real)
                            final_original_real = untotensored_real
                            
                            unnormalize_fake = UnNormalize((0.5,), (0.5,))(fake)
                            untotensored_fake = UnToTensor(unnormalize_fake)
                            final_original_fake = untotensored_fake
                            
                            unnormalize_jewon = UnNormalize((0.5,), (0.5,))(jewon)
                            untotensored_jewon = UnToTensor(unnormalize_jewon)
                            final_original_jewon = untotensored_jewon
                            
                        final_original_real = torch.tensor(final_original_real, dtype=torch.float32)
                        final_original_fake = torch.tensor(final_original_fake, dtype=torch.float32)     
                        final_original_jewon = torch.tensor(final_original_jewon, dtype=torch.float32)  
                        
                        
                        img_grid_fake = torchvision.utils.make_grid(final_original_fake[20:30], normalize=True)#, normalize=True)
                        img_grid_real = torchvision.utils.make_grid(final_original_real[20:30], normalize=True)#, normalize=True)
        
                        writer_fake.add_image(
                            "Hyundai Fake Images", img_grid_fake, global_step=step
                        )
                        writer_real.add_image(
                            "Hyundai Real Images", img_grid_real, global_step=step
                        )
                        step += 1
                        
                        if epoch==args.epochs-1 and batch_idx == len(loader) -1:
                            print('generation is working')
                            
                            GenDirisExist = os.path.exists(generated_image_path)
                            FakeDirisExist = os.path.exists(f'{generated_image_path}/fake')
                            RealDirisExist = os.path.exists(f'{generated_image_path}/real')
                            JewonDirisExist = os.path.exists(f'{generated_image_path}/jewon')
                            if not GenDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(generated_image_path)
                                logger.info("The generated image output directory is created!")
                                
                            if not FakeDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(f'{generated_image_path}/fake')
                                logger.info("The generated image output directory of fake images is created!")
                                
                            
                            if not RealDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(f'{generated_image_path}/real')
                                logger.info("The generated image output directory of real images is created!")
                                
                                                            
                            if not JewonDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(f'{generated_image_path}/jewon')
                                logger.info("The generated image output directory of jewon images is created!")
                                
                            final_original_real = final_original_real.numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
                            final_original_fake = final_original_fake.numpy().transpose((0, 2, 3, 1)).astype(np.uint8)  
                            final_original_jewon = final_original_jewon.numpy().transpose((0, 2, 3, 1)).astype(np.uint8)  
                            
                            
                            
                            for i, fake in enumerate(final_original_fake):
                                fake = Image.fromarray(fake)
                                fake.save(f'{generated_image_path}/fake/fake_{i}.bmp')
                                
                            logger.info("The generated image output directory is created!")
                            for i, real in enumerate(final_original_real):
                                real = Image.fromarray(real)
                                real.save(f'{generated_image_path}/real/real_{i}.bmp')
                                
                            for i, jewon in enumerate(final_original_jewon):
                                jewon = Image.fromarray(jewon)
                                jewon.save(f'{generated_image_path}/jewon/jewon_{i}.bmp')
                                
                                
                                
        
        if args.gan_model == 'infogan':
            model_saving_full_path = f"{args.log_path}/infogan_model.pt"
            
            torch.save({
                        'disc_state_dict': disc.state_dict(),
                        'gen_state_dict': gen.state_dict(),
                        'dhead_state_dict' : dhead.state_dict(),
                        'qhead_state_dict' : qhead.state_dict(),
                        'opt_disc': opt_disc.state_dict(),
                        'opt_gen': opt_gen.state_dict(),
                        # 'norm_mean': args.norm_mean,
                        # 'norm_stdv': args.norm_stdv,
                        'learning_rate_gen': args.learning_rate_gen,
                        'learning_rate_disc': args.learning_rate_disc,
                        'batch_size': args.batch_size,
                        'epochs': args.epochs,
                        'constraint_lambda': args.constraint_lambda,
                        'lambda_gp': args.lambda_gp,
                        'critic_iter': args.critic_iter,
                        'opt_beta1': args.opt_beta1,
                        'opt_beta2': args.opt_beta2,
                        'gan_model': args.gan_model,
                        'disc_mode': args.disc_mode,
                        'gen_mode': args.gen_mode,
                        'num_durability_classes': num_durability_classes,
                        'num_weight_classes': num_weight_classes,
                        'zc_dim': zc_dim,
                        'input_dim': args.input_dim,
                        'hidden_dim': args.hidden_dim,
                        'embed_dim': args.embed_dim,
                        'output_dim': args.output_dim,
                        'num_heads': args.num_heads,
                        'height': args.height,
                        'width': args.width,
                        'loss_mode': args.loss_mode,
                        'datashape': args.datashape,
                        'optimizer': args.optimizer,
                        # 'model_path': args.model_path,
                        # 'model_name': args.model_name,
                        'log_path': args.log_path}
                        , model_saving_full_path)
            
        elif args.gan_model == 'vanila':
            model_saving_full_path = f"{args.log_path}/vanila_model.pt"
            
            torch.save({
                        'disc_state_dict': disc.state_dict(),
                        'gen_state_dict': gen.state_dict(),
                        'opt_disc': opt_disc.state_dict(),
                        'opt_gen': opt_gen.state_dict(),
                        # 'norm_mean': args.norm_mean,
                        # 'norm_stdv': args.norm_stdv,
                        'learning_rate_gen': args.learning_rate_gen,
                        'learning_rate_disc': args.learning_rate_disc,
                        'batch_size': args.batch_size,
                        'epochs': args.epochs,
                        'constraint_lambda': args.constraint_lambda,
                        'lambda_gp': args.lambda_gp,
                        'critic_iter': args.critic_iter,
                        'opt_beta1': args.opt_beta1,
                        'opt_beta2': args.opt_beta2,
                        'gan_model': args.gan_model,
                        'disc_mode': args.disc_mode,
                        'gen_mode': args.gen_mode,
                        'num_durability_classes': num_durability_classes,
                        'num_weight_classes': num_weight_classes,
                        'zc_dim': zc_dim,
                        'input_dim': args.input_dim,
                        'hidden_dim': args.hidden_dim,
                        'embed_dim': args.embed_dim,
                        'output_dim': args.output_dim,
                        'num_heads': args.num_heads,
                        'height': args.height,
                        'width': args.width,
                        'loss_mode': args.loss_mode,
                        'datashape': args.datashape,
                        'optimizer': args.optimizer,
                        # 'model_path': args.model_path,
                        # 'model_name': args.model_name,
                        'log_path': args.log_path}
                        , model_saving_full_path)
        
        
        print('training took ', f'{time.time()-s}', ' seconds')
        logger.info(f'training took {time.time()-s} seconds') 
        print('Job done ')
        logger.info('training loop done!') 
    
             
    elif args.gan_model == 'vanila' and args.loss_mode == 'wgan':                        
        logger.info('vanila GAN model with WGAN loss term training initiated!')
        for epoch in tqdm(range(args.epochs)):
            for batch_idx, (real, durability, weight, jewon, mask, imagename) in tqdm(enumerate(loader)):

                ##taking out inputs
                real = real.view(-1, args.input_dim).to(args.device)
                durability = durability.long().to(args.device)
                weight = weight.long().to(args.device)
                jewon = jewon.view(-1, args.input_dim).to(args.device)
                mask = mask.view(-1, args.input_dim).to(args.device)
                batch_size = real.shape[0]     

                gen.train()
                disc.train()     


                
                for _ in range(args.critic_iter):
                    # break
                    ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                    
                    ##for normal model
                    # noise = torch.randn(batch_size, args.zdim).to(args.device)
                    # final_noise = torch.cat((noise, durability.unsqueeze(1), weight.unsqueeze(1)), dim =1)
                    # fake = gen(noise, durability, weight, noise)
                    
                    if args.gen_mode == 'vanila':
                        noise = torch.randn(batch_size, args.zdim).to(args.device)
                        fake = gen(noise, durability, weight, jewon)
                    
                    
                    #for selfattention models
                    elif args.gen_mode == 'selfattention':
                        noise = torch.randn(batch_size, args.zdim).to(args.device)
                        final_input = torch.cat((noise, durability, weight), dim=1)
                        fake = gen(final_input, jewon)
                        
                    

                    disc_real = disc(real).view(-1)
                    disc_fake = disc(fake).view(-1)
                    
                    loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake))
                    # lossD_real = criterion(disc_real, torch.ones_like(disc_real))        
                    # lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                    # lossD = (lossD_real + lossD_fake) / 2
                    
                    # gp = gradient_penalty(disc, durability_weight, jewon, real_image, fake_image, device=device)
                    disc.zero_grad()
                    loss_disc.backward(retain_graph=True)
                    opt_disc.step()
                        # clip critic weights between -0.01, 0.01
                    for p in disc.parameters():
                        p.data.clamp_(-args.weight_clip, args.weight_clip)

                ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                # where the second option of maximizing doesn't suffer from
                # saturating gradients     
                
                # if args.gen_mode == 'vanila':
                #     # noise = torch.randn(batch_size, args.zdim).to(args.device)
                #     gen_fake = gen(noise, durability, weight, jewon)
                
                
                # #for selfattention models
                # elif args.gen_mode == 'selfattention':
                #     # noise = torch.randn(batch_size, args.zdim).to(args.device)
                #     # final_input = torch.cat((noise, durability, weight), dim=1)
                #     gen_fake = gen(final_input, jewon)
                
                
                
                
                # loss_gen = -(torch.mean(disc(gen_fake)))  -args.constraint_lambda*torch.norm(jewon - torch.mul(mask, gen_fake))    
                loss_gen = -(torch.mean(disc(fake)))  -args.constraint_lambda*torch.norm(jewon - torch.mul(mask, fake))    

                gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()


                if (batch_idx % 100 == 0 and batch_idx > 0) or (batch_idx == len(loader) -1 and batch_idx > 0):
                    logger.info(
                        f"Epoch [{epoch}/{args.epochs}] Batch {batch_idx}/{len(loader)} \
                              Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                    )
                    with torch.no_grad():
                        
                        gen.eval()
                        disc.eval()
                        
                        
                        if args.gen_mode == 'vanila':
                            # noise = torch.randn(batch_size, args.zdim).to(args.device)
                            fake = gen(noise, durability, weight, jewon)
                        
                        
                        #for selfattention models
                        elif args.gen_mode == 'selfattention':
                            # noise = torch.randn(batch_size, args.zdim).to(args.device)
                            # final_input = torch.cat((noise, durability, weight), dim=1)
                            fake = gen(final_input, jewon)
                            
                        # fake = gen(noise, durability, weight, jewon)#.reshape(-1, 3, 10, 30)
                        fake = fake.reshape(-1, 3, args.height, args.width)
                        real = real.reshape(-1, 3, args.height, args.width)
                        jewon=jewon.reshape(-1, 3, args.height, args.width)
                        # z = jewon.detach().cpu().numpy()
                        
                        if args.datashape == 'dropempty':
                            
                            
                            # jewon=jewon.reshape(-1, 3, args.height, args.width)
                            
                            # 억지로 실제 데이터의 fixed pixel들을 생성해낸 이미지에도 붙여넣고 싶으면 아래 코드를 언커맨트하면된다.
                            # fake[:,:, [7, 5], [7, 10]] = jewon[:,:, [7, 5], [7, 10]]
                            
                            unnormalize_real = UnNormalize((0.5,), (0.5,))(real)
                            untotensored_real = UnToTensor(unnormalize_real)
                            
                            final_original_real = np.zeros((batch_size, 3, 10, 30), dtype=np.uint8)
                            final_original_real[:, :, :, 0:6] = untotensored_real[:, :, :, 0:6]
                            final_original_real[:, :, :, 10:16] = untotensored_real[:, :, :, 6:12]
                            final_original_real[:, :, :, 20:22] = untotensored_real[:, :, :, 12:14]
                            
                            unnormalize_fake = UnNormalize((0.5,), (0.5,))(fake)
                            untotensored_fake = UnToTensor(unnormalize_fake)
    
                            final_original_fake = np.zeros((batch_size, 3, 10, 30), dtype=np.uint8)
                            final_original_fake[:, :, :, 0:6] = untotensored_fake[:, :, :, 0:6]
                            final_original_fake[:, :, :, 10:16] = untotensored_fake[:, :, :, 6:12]
                            final_original_fake[:, :, :, 20:22] = untotensored_fake[:, :, :, 12:14]
                            
                            
                            unnormalize_jewon = UnNormalize((0.5,), (0.5,))(jewon)
                            untotensored_jewon = UnToTensor(unnormalize_jewon)
    
                            final_original_jewon = np.zeros((batch_size, 3, 10, 30), dtype=np.uint8)
                            final_original_jewon[:, :, :, 0:6] = untotensored_jewon[:, :, :, 0:6]
                            final_original_jewon[:, :, :, 10:16] = untotensored_jewon[:, :, :, 6:12]
                            final_original_jewon[:, :, :, 20:22] = untotensored_jewon[:, :, :, 12:14]
                            
                                
                            
                        elif args.datashape == 'original':

                            # real = real.reshape(-1, 3, args.height, args.width)
                            # real[:,:, [7, 5], [11, 14]]
                            
                            # fake = fake.reshape(-1, 3, args.height, args.width)
                            # fake[:,:, [7, 5], [11, 14]]
                            
                            # jewon=jewon.reshape(-1, 3, args.height, args.width)
                            # jewon[:,:, [7, 5], [11, 14]]
                            
                            
                            ##억지로 실제 데이터의 fixed pixel들을 생성해낸 이미지에도 붙여넣고 싶으면 아래 코드를 언커맨트하면된다.
                            # fake[:,:, [7, 5], [11, 14]] = jewon[:,:, [7, 5], [11, 14]]
                            
                  
                            unnormalize_real = UnNormalize((0.5,), (0.5,))(real)
                            untotensored_real = UnToTensor(unnormalize_real)
                            final_original_real = untotensored_real
                            
                            unnormalize_fake = UnNormalize((0.5,), (0.5,))(fake)
                            untotensored_fake = UnToTensor(unnormalize_fake)
                            final_original_fake = untotensored_fake
                            
                            unnormalize_jewon = UnNormalize((0.5,), (0.5,))(jewon)
                            untotensored_jewon = UnToTensor(unnormalize_jewon)
                            final_original_jewon = untotensored_jewon
                            
                            
                            
                        final_original_real = torch.tensor(final_original_real, dtype=torch.float32)
                        final_original_fake = torch.tensor(final_original_fake, dtype=torch.float32)   
                        final_original_jewon = torch.tensor(final_original_jewon, dtype=torch.float32)   
                        
                        
                        img_grid_fake = torchvision.utils.make_grid(final_original_fake[20:30], normalize=True)#, normalize=True)
                        img_grid_real = torchvision.utils.make_grid(final_original_real[20:30], normalize=True)#, normalize=True)
                        
                        # img_grid_fake = torchvision.utils.make_grid(fake[20:30])#, normalize=True)
                        # img_grid_real = torchvision.utils.make_grid(real[20:30])#, normalize=True)
                        
                        
                        writer_fake.add_image(
                            "Hyundai Fake Images", img_grid_fake, global_step=step
                        )
                        writer_real.add_image(
                            "Hyundai Real Images", img_grid_real, global_step=step
                        )
                        step += 1
                        
                        if epoch==args.epochs-1 and batch_idx == len(loader) -1:
                            print('generation is working')
                            
                            GenDirisExist = os.path.exists(generated_image_path)
                            FakeDirisExist = os.path.exists(f'{generated_image_path}/fake')
                            RealDirisExist = os.path.exists(f'{generated_image_path}/real')
                            JewonDirisExist = os.path.exists(f'{generated_image_path}/jewon')
                            if not GenDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(generated_image_path)
                                logger.info("The generated image output directory is created!")
                                
                            if not FakeDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(f'{generated_image_path}/fake')
                                logger.info("The generated image output directory of fake images is created!")
                                
                            
                            if not RealDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(f'{generated_image_path}/real')
                                logger.info("The generated image output directory of real images is created!")
                                
                                                            
                            if not JewonDirisExist:
                        
                                # Create a new directory because it does not exist
                                os.makedirs(f'{generated_image_path}/jewon')
                                logger.info("The generated image output directory of jewon images is created!")
                                
                            final_original_real = final_original_real.numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
                            final_original_fake = final_original_fake.numpy().transpose((0, 2, 3, 1)).astype(np.uint8)  
                            final_original_jewon = final_original_jewon.numpy().transpose((0, 2, 3, 1)).astype(np.uint8)  
                            
                            # jewons = jewon.detach().cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)  

                            # unnormalize_fake = UnNormalize((0.5,), (0.5,))(fake)
                            # untotensored_fake = UnToTensor(unnormalize_fake)
                            
                            # unnormalize_real = UnNormalize((0.5,), (0.5,))(real)
                            # untotensored_real = UnToTensor(unnormalize_real)
                            
                            # unnormalize_jewon = UnNormalize((0.5,), (0.5,))(jewon)
                            # untotensored_jewon = UnToTensor(unnormalize_jewon)
                            
                            
                            # real = untotensored_real.transpose((0, 2, 3, 1)).astype(np.uint8)
                            # fake = untotensored_fake.transpose((0, 2, 3, 1)).astype(np.uint8)    
                            # jewon = untotensored_jewon.transpose((0, 2, 3, 1)).astype(np.uint8)  

                            
                            
                            for i, fake in enumerate(final_original_fake):
                                fake = Image.fromarray(fake)
                                fake.save(f'{generated_image_path}/fake/fake_{i}.bmp')
                                
                            logger.info("The generated image output directory is created!")
                            for i, real in enumerate(final_original_real):
                                real = Image.fromarray(real)
                                real.save(f'{generated_image_path}/real/real_{i}.bmp')
                                
                            for i, jewon in enumerate(final_original_jewon):
                                jewon = Image.fromarray(jewon)
                                jewon.save(f'{generated_image_path}/jewon/jewon_{i}.bmp')
                                
                                
                            # for i, f in enumerate(fake):
                            #     f = Image.fromarray(f)
                            #     f.save(f'{generated_image_path}/fake/fake_{i}.bmp')
                                
                            # logger.info("The generated image output directory is created!")
                            # for i, r in enumerate(real):
                            #     r = Image.fromarray(r)
                            #     r.save(f'{generated_image_path}/real/real_{i}.bmp')
                                
                            # for i, j in enumerate(jewon):
                            #     j = Image.fromarray(j)
                            #     j.save(f'{generated_image_path}/jewon/jewon_{i}.bmp')

                            
        if args.gan_model == 'infogan':
            model_saving_full_path = f"{args.log_path}/infogan_model.pt"
            
            torch.save({
                        'disc_state_dict': disc.state_dict(),
                        'gen_state_dict': gen.state_dict(),
                        'dhead_state_dict' : dhead.state_dict(),
                        'qhead_state_dict' : qhead.state_dict(),
                        'opt_disc': opt_disc.state_dict(),
                        'opt_gen': opt_gen.state_dict(),
                        # 'norm_mean': args.norm_mean,
                        # 'norm_stdv': args.norm_stdv,
                        'learning_rate_gen': args.learning_rate_gen,
                        'learning_rate_disc': args.learning_rate_disc,
                        'batch_size': args.batch_size,
                        'epochs': args.epochs,
                        'constraint_lambda': args.constraint_lambda,
                        'lambda_gp': args.lambda_gp,
                        'critic_iter': args.critic_iter,
                        'opt_beta1': args.opt_beta1,
                        'opt_beta2': args.opt_beta2,
                        'gan_model': args.gan_model,
                        'disc_mode': args.disc_mode,
                        'gen_mode': args.gen_mode,
                        'num_durability_classes': num_durability_classes,
                        'num_weight_classes': num_weight_classes,
                        'zc_dim': zc_dim,
                        'input_dim': args.input_dim,
                        'hidden_dim': args.hidden_dim,
                        'embed_dim': args.embed_dim,
                        'output_dim': args.output_dim,
                        'num_heads': args.num_heads,
                        'height': args.height,
                        'width': args.width,
                        'loss_mode': args.loss_mode,
                        'datashape': args.datashape,
                        'optimizer': args.optimizer,
                        # 'model_path': args.model_path,
                        # 'model_name': args.model_name,
                        'log_path': args.log_path}
                        , model_saving_full_path)
            
        elif args.gan_model == 'vanila':
            model_saving_full_path = f"{args.log_path}/vanila_model.pt"
            
            torch.save({
                        'disc_state_dict': disc.state_dict(),
                        'gen_state_dict': gen.state_dict(),
                        'opt_disc': opt_disc.state_dict(),
                        'opt_gen': opt_gen.state_dict(),
                        # 'norm_mean': args.norm_mean,
                        # 'norm_stdv': args.norm_stdv,
                        'learning_rate_gen': args.learning_rate_gen,
                        'learning_rate_disc': args.learning_rate_disc,
                        'batch_size': args.batch_size,
                        'epochs': args.epochs,
                        'constraint_lambda': args.constraint_lambda,
                        'lambda_gp': args.lambda_gp,
                        'critic_iter': args.critic_iter,
                        'opt_beta1': args.opt_beta1,
                        'opt_beta2': args.opt_beta2,
                        'gan_model': args.gan_model,
                        'disc_mode': args.disc_mode,
                        'gen_mode': args.gen_mode,
                        'num_durability_classes': num_durability_classes,
                        'num_weight_classes': num_weight_classes,
                        'zc_dim': zc_dim,
                        'input_dim': args.input_dim,
                        'hidden_dim': args.hidden_dim,
                        'embed_dim': args.embed_dim,
                        'output_dim': args.output_dim,
                        'num_heads': args.num_heads,
                        'height': args.height,
                        'width': args.width,
                        'loss_mode': args.loss_mode,
                        'datashape': args.datashape,
                        'optimizer': args.optimizer,
                        # 'model_path': args.model_path,
                        # 'model_name': args.model_name,
                        'log_path': args.log_path}
                        , model_saving_full_path)
        
        
        # print('training took ', f'{time.time()-s}', ' seconds')
        logger.info(f'training took {time.time()-s} seconds') 
        # print('Job done ')
        logger.info('training loop done!') 
        
                    
if __name__ == "__main__":
    main(args)                    
