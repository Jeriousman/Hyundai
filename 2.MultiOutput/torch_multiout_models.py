# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:43:30 2023

@author: jake_
"""



import torch
import torch.nn as nn
import torch.nn.functional as F





from multiout_layers import PreMLP, PostMLP, SelfAttentionBlock
# Epoch 50 		 Averge Training Loss: 2.9331411497271296e-07
# 100%|██████████| 50/50 [02:05<00:00,  2.51s/it]Epoch 50 		 Averge Validation Loss: 9.207166347853975e-08


class MultOutRegressor(nn.Module):
    def __init__(self, input_dim, target_dim, hidden_dim=64, p=0.3, seed=1234):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim

        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.target_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3)
        # x = F.batch_norm(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3)
        # x = F.batch_norm(x)
        x = self.fc3(x)
        x = F.dropout(x, p=0.3)
        # x = F.batch_norm(x)
        return x
    
    
    



class MultOutChainedRegressor(nn.Module):
    def __init__(self, input_dim, target_dim, order, hidden_dim=64,seed=1234 ):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self._order = order 
        assert len(self._order) == self.target_dim
        assert min(self._order) == 0 

        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_seq = []
        self.nested_dim = self.hidden_dim
        self.output_rank = {}
        for idx , order in enumerate(self._order) : 
            self.output_seq.append(nn.Linear(self.nested_dim, 1))
            self.nested_dim += 1 
            self.output_rank[idx] = order 
        else :
            self.linears = nn.ModuleList(self.output_seq)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        _x = x
        last_vector = torch.zeros(size=(x.size()[0],self.target_dim)).to('cuda')
        for idx , rank in self.output_rank.items() :
            y_hat = self.output_seq[idx](_x)
            last_vector[:,[rank]] += y_hat
            _x = torch.cat([_x,y_hat],axis=1)
        return last_vector
    
    
def predict(model, dataloader):
    predictions = []
    for i, (data, label) in enumerate(dataloader):
        batch_size = data.shape[0]
        preds = model(data.reshape(batch_size, -1).to('cuda'))
        predictions.append(preds.cpu())
    predictions = torch.cat([tensor for tensor in predictions], dim=0) ## putting all together into a tensor
    return predictions





class MultOutRegressorSelfAttentionMLP(nn.Module):
    def __init__(self, img_dim=900, seq_len=300, embed_dim=32, label_dim=189, num_heads=2, batch_first=True):
        super(MultOutRegressorSelfAttentionMLP, self).__init__()
        # self.height=height
        # self.width=width
        # self.channel=channel
        self.img_dim = img_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.label_dim = label_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        
        self.premlp = PreMLP(self.img_dim, self.seq_len, self.embed_dim)
        self.selfattention = SelfAttentionBlock(self.embed_dim, self.num_heads, self.batch_first)
        self.postmlp =  PostMLP(self.seq_len, self.embed_dim, self.label_dim)     
        # self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.premlp(x)
        x, attn_output_weights = self.selfattention(x)
        x = x.reshape(-1, self.seq_len * self.embed_dim)
        x = self.postmlp(x)
        # x = self.tanh(x)
        return x
    
# order =    sorted(set(range(0, args.label_dim)))
# len(order) 
# target_dim = len(order)

# output_seq = []
# selfattention_seq = []



# nested_dim = 64
# output_rank = {}
# for idx , order in enumerate(order) : 
#     selfattention_seq.append(SelfAttentionBlock(nested_dim, num_heads=1, batch_first=True).to('cuda'))
#     output_seq.append(nn.Linear(nested_dim, 1).to('cuda'))
#     nested_dim += 1 
#     output_rank[idx] = order 

# fc1 = nn.Linear(420, 64).to('cuda')
# fc2 = nn.Linear(64, 64).to('cuda')

# x = fc1(data)
# x = F.relu(x)
# x = fc2(x)
# x = F.relu(x)
# _x = x



# last_vector = torch.zeros(size=(data.size()[0], target_dim)).to('cuda')
# last_vector.shape
# for idx , rank in output_rank.items() :
#     intermediate, _  = selfattention_seq[idx](_x) ##intermediate = [B, H]
#     y_hat = output_seq[idx](intermediate) ## [B, H] -> [B, 1]. yhat = [B, 1]
#     last_vector[:,[rank]] += y_hat
#     _x = torch.cat([_x,y_hat],axis=1)



class MultOutChainedSelfAttentionRegressor(nn.Module):
    def __init__(self, input_dim, target_dim, order, hidden_dim=64,seed=1234):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self._order = order 
        assert len(self._order) == self.target_dim
        assert min(self._order) == 0 

        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.selfattention_seq = []
        self.output_seq = []
        self.nested_dim = self.hidden_dim
        self.output_rank = {}
        for idx , order in enumerate(self._order) : 
            self.selfattention_seq.append(SelfAttentionBlock(self.nested_dim, num_heads=1, batch_first=True).to('cuda'))
            self.output_seq.append(nn.Linear(self.nested_dim, 1).to('cuda'))
            self.nested_dim += 1 
            self.output_rank[idx] = order 
        else :
            self.linears = nn.ModuleList(self.output_seq)
            self.selfattentions = nn.ModuleList(self.selfattention_seq)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        _x = x
        last_vector = torch.zeros(size=(x.size()[0],self.target_dim)).to('cuda')
        for idx , rank in self.output_rank.items() :
            intermediate, _ = self.selfattention_seq[idx](_x)
            intermediate = F.dropout(intermediate, p=0.3)
            y_hat = self.output_seq[idx](intermediate)
            last_vector[:,[rank]] += y_hat
            _x = torch.cat([_x,y_hat],axis=1)
        return last_vector