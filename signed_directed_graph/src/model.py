
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_add, scatter_mean,scatter_sum,scatter_max,scatter_min


class bssubgnn(torch.nn.Module):
    def __init__(self, attri_dim, hid_dim,hid_dim2,out_dim,n_layers,n_rel,is_directed,is_self):
        super(bssubgnn, self).__init__()
        self.n_layers = n_layers
        self.n_rel = n_rel
        self.is_self = is_self
        self.is_directed = is_directed
        if self.is_directed:
            self.n_nei = self.n_rel*2
        
        self.lin = nn.ModuleList()
        self.lin.append(nn.Linear(attri_dim, hid_dim))
        for i in range(self.n_layers):
            self.lin.append(nn.Linear(hid_dim, hid_dim))
        
        self.lin_agg = nn.ModuleList()
        for i in range(self.n_layers):
            for j in range(self.n_nei):
                self.lin_agg.append(nn.Linear(2*hid_dim, 1))
        
        tmp = 1 if self.is_self else 0
        self.lin_cat = nn.ModuleList()
        for i in range(self.n_layers):
            self.lin_cat.append(nn.Linear((self.n_nei+tmp)*hid_dim, hid_dim))
        
        self.lin_exist = nn.Sequential(
            nn.Linear(3*self.n_layers*hid_dim,32),
            nn.ReLU(),#ReLU Tanh
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,out_dim))
        self.lin_pool = nn.ModuleList()
        for i in range(3):
            self.lin_pool.append(nn.Linear(self.n_layers*hid_dim,1))
        
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_index_list = []
        for i in range(self.n_rel):
            edge_index_list.append(edge_index[:,edge_attr==i+1])
        num_subgrpahs = data.batch.max().item()+1
        
        idx3 = x[:,2]==1
        idx4 = x[:,3]==1
        idx5 = x[:,4]==1
        
        z = []
        for i in range(self.n_layers):
            x = self.sumattagg(self.lin[i],self.lin_agg[self.n_nei*i:self.n_nei*(i+1)],self.lin_cat[i],x,edge_index_list,self.is_self)
            x = F.tanh(x)
            z.append(x)
        x = torch.cat(z,1)
        
        x_out1 = self.pool(x[idx3],batch[idx3],num_subgrpahs,self.lin_pool[0])
        x_out2 = self.pool(x[idx4],batch[idx4],num_subgrpahs,self.lin_pool[1])
        x_out3 = self.pool(x[idx5],batch[idx5],num_subgrpahs,self.lin_pool[2])
        
        x_out = torch.cat((x_out1,x_out2,x_out3),1)
        out_exist = self.lin_exist(x_out)
        
        return out_exist
    
    
    def sumagg(self,w_lin,w_cat,x, edge_index_list,is_self=False):
        xx = []
        x = w_lin(x)
        if is_self:
            xx.append(x)
        for i in range(len(edge_index_list)):
            edge_index = edge_index_list[i]
            row, col = edge_index
            out = scatter_sum(x[col], row, dim=0, dim_size=x.size(0))
            xx.append(out)
            out = scatter_sum(x[row], col, dim=0, dim_size=x.size(0))
            xx.append(out)
        x = torch.cat(xx,1)
        x = w_cat(x)
        return x
    
    
    def sumattagg(self,w_lin,w_agg,w_cat,x, edge_index_list,is_self=False):
        xx = []
        x = w_lin(x)
        if is_self:
            xx.append(x)
        for i in range(len(edge_index_list)):
            edge_index = edge_index_list[i]
            row, col = edge_index
            
            ee = torch.cat((x[row],x[col]),1)
            att_score = w_agg[2*i](ee)
            att_score = F.tanh(att_score)
            att_score = torch.exp(att_score)
            ee = x[col]*att_score
            out = scatter_sum(ee, row, dim=0, dim_size=x.size(0))
            xx.append(out)
            
            ee = torch.cat((x[col],x[row]),1)
            att_score2 = w_agg[2*i+1](ee)
            att_score2 = F.tanh(att_score2)
            att_score2 = torch.exp(att_score2)
            ee = x[row]*att_score2
            out = scatter_sum(ee, col, dim=0, dim_size=x.size(0))
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

