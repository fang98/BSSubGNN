# -*- coding: utf-8 -*-

import random
import numpy as np
import warnings
import torch
from torch import nn
from torch import optim
from torch_geometric.data import DataLoader
from utils import load_data,process,evaluate,graph2data
import argparse
import os
from model import bssubgnn
from torch_geometric import seed_everything
from config import const_args
from tqdm import tqdm
#from data_utils import load_data

# ------------ argument ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='twitch_en', dest = 'dataset', help='dataset name')
# citeseer cora twitch_en cs
parser.add_argument('--seed', type=int, default=2, dest = 'seed', help='data seed')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--batch_size',type=int, default=256)
parser.add_argument('--n_layers',type=int, default=3)
parser.add_argument('--hop',type=str, default=1)
parser.add_argument('--epochs',type=int, default=20)
parser.add_argument('--hidden_dim',type=int, default=32)
parser.add_argument('--hidden_dim2',type=int, default=16)
parser.add_argument('--r_val',type=float, default=0.05)
parser.add_argument('--using_attr', action='store_true', default=False)
# Controls whether to use node attributes:
# - False (default): Used in citeseer and cora datasets.
# - True: Used in twitch_en and cs datasets.
args = parser.parse_args()


args_dict = vars(args)
args_dict.update(const_args)
args = argparse.Namespace(**args_dict)


# ------------- training & evaluation ---------------
def run(args):
    warnings.filterwarnings('ignore')
    print(args)
    seed_everything(seed=args.seed)
    
    if not os.path.exists('./subgnn_model'):
        os.makedirs('./subgnn_model')
    
    dataset = load_data(args.dataset)
    train_edges,test_edges,adj_train,node_attr,neighbor_list = process(args,dataset)
    
    train_data = graph2data(train_edges,adj_train,node_attr,neighbor_list,args.hop)
    test_data = graph2data(test_edges,adj_train,node_attr,neighbor_list,args.hop)
    if args.r_val>0:
        random.shuffle(train_data)
        n_val = int(args.r_val*len(train_data))
        val_data = train_data[:n_val]
        train_data = train_data[n_val:]
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    else:
        # val_data = None
        val_loader = None
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    attr_dim = train_data[0].num_features
    del train_data
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    del test_data
    
    is_self = True
    model = bssubgnn(attr_dim, args.hidden_dim,args.hidden_dim2,args.n_layers,is_self).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_func = nn.CrossEntropyLoss()
    
    n_samples = 0
    best_res = 0
    upd = 1
    
    for epoch in tqdm(range(args.epochs), desc="Training", unit="epoch"):
        model.train()
        total_loss = []
        
        for batch in train_loader:
            batch.to(args.device)
            out = model(batch)
            
            loss = loss_func(out, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss.append( loss.item() * len(batch.y))
            n_samples += len(batch.y)
            
        total_loss = np.array(total_loss)
        avg_loss = np.sum(total_loss, 0) / n_samples
        
        if (epoch + 1) % upd == 0:
            upt_res = evaluate(model,test_loader,args.device)
            
            if upt_res[0]+upt_res[1] > best_res and args.r_val>0:
                upt_res = evaluate(model,val_loader,args.device)
                torch.save(obj=model.state_dict(), f='subgnn_model/'+args.dataset+'_'+str(args.seed+1)+'.pth')
                best_res = upt_res[0]+upt_res[1]
    
    if args.r_val>0:
        new_model = bssubgnn(attr_dim,args.hidden_dim,args.hidden_dim2,args.n_layers,is_self).to(args.device)
        new_model.load_state_dict(torch.load('subgnn_model/'+args.dataset+'_'+str(args.seed+1)+'.pth'))
        upt_res = evaluate(model,test_loader,args.device)
    auc = upt_res[0]
    ap = upt_res[1]
    
    return auc, ap


if __name__ == '__main__':
    auc, ap = run(args)
    print('Final results: AUC =',auc,', AP =',ap)






