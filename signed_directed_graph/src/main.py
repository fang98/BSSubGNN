
import numpy as np
import os
import argparse
from config import const_args
import torch
from torch import optim
from torch import nn
from torch_geometric.data import DataLoader
from model import bssubgnn
from utils import setup_seed,process_dataset,evaluate,graph2data
import warnings
from tqdm import tqdm


# ------------ argument ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='BitcoinAlpha', dest = 'dataset', help='dataset name')
#BitcoinAlpha BitcoinOTC WikiRfA Slashdot Epinions
parser.add_argument('--seed', type=int, default=1, dest = 'seed', help='data seed')#1 2 3 4 5
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--batch_size',type=int, default=256)
parser.add_argument('--n_layers',type=int, default=3)
parser.add_argument('--hop',type=str, default=1)
parser.add_argument('--epochs',type=int, default=20)
parser.add_argument('--hidden_dim',type=int, default=32)
parser.add_argument('--hidden_dim2',type=int, default=16)
parser.add_argument('--out_dim',type=int, default=2)
parser.add_argument('--r_val', type=float, default=0.05)
parser.add_argument('--max_sub_num',type=int, default=1e10)
args = parser.parse_args()


args_dict = vars(args)
args_dict.update(const_args)
args = argparse.Namespace(**args_dict)

# ------------- training & evaluation ---------------
def run(args):
    warnings.filterwarnings('ignore')
    print(args)
    setup_seed(seed=args.seed)
    
    if not os.path.exists('./subgnn_model'):
        os.makedirs('./subgnn_model')
    
    num_nodes, adj_train, neighbor_list, bfs_labels, node_attr, train_edges, test_edges, val_edges = \
        process_dataset(args)
    
    train_data = graph2data(train_edges,adj_train,node_attr,neighbor_list,bfs_labels,args.hop,args.max_sub_num)
    test_data = graph2data(test_edges,adj_train,node_attr,neighbor_list,bfs_labels,args.hop,args.max_sub_num)
    val_data = graph2data(val_edges,adj_train,node_attr,neighbor_list,bfs_labels,args.hop,args.max_sub_num)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    is_self = True
    attr_dim = train_data[0].x.shape[1]#
    model = bssubgnn(attr_dim, args.hidden_dim,args.hidden_dim2,args.out_dim,args.n_layers,2,True,is_self).to(args.device)
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
            
            upt_res = evaluate(model,val_loader,args.device)
            if upt_res[0]+upt_res[1] > best_res:
                torch.save(obj=model.state_dict(), f='subgnn_model/'+args.dataset+'_'+str(args.seed)+'.pth')
                best_res = upt_res[0]+upt_res[1]
    
    new_model = bssubgnn(attr_dim, args.hidden_dim,args.hidden_dim2,args.out_dim,args.n_layers,2,True,is_self).to(args.device)
    new_model.load_state_dict(torch.load('subgnn_model/'+args.dataset+'_'+str(args.seed)+'.pth'))
    upt_res = evaluate(model,test_loader,args.device)
    
    return upt_res


if __name__ == '__main__':
    res = run(args)
    print('AUC =',res[0],'Macro-F1 =',res[1],'Micro-F1 =',res[2],'F1 =',res[3])