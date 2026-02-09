
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_add, scatter_mean,scatter_sum,scatter_max,scatter_min
from torch_geometric.nn import GCNConv


class bssubgnn(torch.nn.Module):
    def __init__(self, attri_dim, hid_dim,hid_dim2,n_layers,is_self):
        super(bssubgnn, self).__init__()
        self.n_layers = n_layers
        self.is_self = is_self
        
        self.lin = nn.ModuleList()
        self.lin.append(nn.Linear(attri_dim, hid_dim))
        for i in range(self.n_layers):
            self.lin.append(nn.Linear(hid_dim, hid_dim))
        
        self.lin_agg = nn.ModuleList()
        for i in range(self.n_layers):
            self.lin_agg.append(GCNConv(hid_dim, hid_dim))#GCNConv GATConv
        
        tmp = 1 if self.is_self else 0
        self.lin_cat = nn.ModuleList()
        for i in range(self.n_layers):
            self.lin_cat.append(nn.Linear((1+tmp)*hid_dim, hid_dim))
        
        self.lin_exist = nn.Sequential(
            nn.Linear(3*self.n_layers*hid_dim,32),
            nn.ReLU(),#ReLU Tanh
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,4))
        self.lin_pool = nn.ModuleList()
        for i in range(3):
            self.lin_pool.append(nn.Linear(self.n_layers*hid_dim,1))
        
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        row,col = edge_index
        num_subgrpahs = data.batch.max().item()+1
        idx3 = x[:,2]==1
        idx4 = x[:,3]==1
        idx5 = x[:,4]==1
        
        z = []
        for i in range(self.n_layers):
            x = self.aggregate(self.lin[i],self.lin_agg[i],self.lin_cat[i],x,edge_index,self.is_self)
            x = F.tanh(x)
            z.append(x)
        x = torch.cat(z,1)
        
        x_out1 = self.pool(x[idx3],batch[idx3],num_subgrpahs,self.lin_pool[0])
        x_out2 = self.pool(x[idx4],batch[idx4],num_subgrpahs,self.lin_pool[1])
        x_out3 = self.pool(x[idx5],batch[idx5],num_subgrpahs,self.lin_pool[2])
        
        x_out = torch.cat((x_out1,x_out2,x_out3),1)
        out_exist = self.lin_exist(x_out)
        
        return out_exist
    
    
    def aggregate(self,w_lin,w_agg,w_cat,x, edge_index,is_self=False):
        xx = []
        x = w_lin(x)
        if is_self:
            xx.append(x)
        out = w_agg(x,edge_index)
        xx.append(out)
        x = torch.cat(xx,1)
        x = w_cat(x)
        return x
    
    
    def pool(self,x,batch,n,lin):
        score = lin(x)
        score = F.tanh(score) #leaky_relu tanh
        score = torch.exp(score)
        
        x = scatter_sum(x*score, batch, dim=0, dim_size=n)
        
        return x

