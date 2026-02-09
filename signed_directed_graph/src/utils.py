
import random
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
import scipy.sparse as sp
from torch_geometric.data import Data
from functools import partial
from tqdm import tqdm



def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def get_graph(adj):
    G = nx.DiGraph()
    G.add_nodes_from(list(range( np.size(adj,0) )))
    tmp = sp.coo_matrix(adj)
    row,col,data = tmp.row,tmp.col,tmp.data
    
    for u,v,s in zip(row,col,data):
        G.add_edge(u, v, weight = s)
        
    return G
    
    
def process_dataset(args,node_attr=None):
    dict_num_nodes = {'BitcoinAlpha':3782,'BitcoinOTC':5881,'WikiRfA':11259,'Slashdot':82140,'Epinions':131827}
    dict_path = {'BitcoinAlpha':'soc-sign-bitcoinalpha.csv','BitcoinOTC':'soc-sign-bitcoinotc.csv',\
                 'WikiRfA':'wiki-RfA.txt','Slashdot':'soc-sign-Slashdot090221.txt','Epinions':'soc-sign-epinions.txt'}
    
    datapath = dict_path[args.dataset][:-4]
    seed = args.seed
    train_path = 'input/'+datapath+'_'+'train'+'_'+str(seed)+'.npy'
    train_edges = np.load(train_path).astype(int)
    test_path = 'input/'+datapath+'_'+'test'+'_'+str(seed)+'.npy'
    test_edges = np.load(test_path).astype(int)
    
    train_edges[:,:2] = train_edges[:,:2]#-1
    test_edges[:,:2] = test_edges[:,:2]#-1
    num_nodes = dict_num_nodes[args.dataset]
    
    if args.r_val>0:
        n_train = train_edges.shape[0]
        idx = np.random.permutation(n_train)
        n_val = int(n_train*args.r_val)
        n_train = n_train-n_val
        val_edges = train_edges[idx[:n_val],:]
        train_edges = train_edges[idx[n_val:],:]
    else:
        val_edges = None
    
    adj_train = sp.csr_matrix((train_edges[:,2].astype(float), (train_edges[:,0], train_edges[:,1])),
                   shape = (num_nodes, num_nodes))
    G2 = get_graph(adj_train)
    
    neighbor_list = []
    for vi in range(num_nodes):
        nei1 = set(list(G2.successors(vi)))
        nei2 = set(list(G2.predecessors(vi)))
        neighbor_list.append([nei1|nei2])
    if args.hop>1:
        for vi in range(num_nodes):
            nei2 = set()
            for vj in neighbor_list[vi][0]:
                nei2 = nei2.union(neighbor_list[vj][0])
            nei2.discard(vi)
            nei2 = nei2-neighbor_list[vi][0]
            neighbor_list[vi].append(nei2)
    
    bfs_labels = []
    for vi in range(num_nodes):
        # dict_bfs = {vi:0}
        dict_bfs = {}
        for h in range(args.hop):
            for vj in neighbor_list[vi][h]:
                dict_bfs[vj] = h+1
        bfs_labels.append(dict_bfs)
    
    return num_nodes, adj_train, neighbor_list, bfs_labels, node_attr, train_edges, test_edges, val_edges


def evaluate(model,loader,device):
    model.eval()
    all_targets = []
    all_scores = []
    
    for batch in loader:
        batch.to(device)
        out = model(batch)
        all_scores.append(F.softmax(out,dim=1)[:, 1].cpu().detach())
        all_targets.extend(batch.y.tolist())
    all_scores = torch.cat(all_scores).cpu().numpy()
    pred_labels = np.zeros(np.size(all_scores))
    pred_labels[np.where(all_scores>0.5)] = 1
    upt_res = [0]*4
    upt_res[0] = roc_auc_score(all_targets,all_scores)
    upt_res[1] = f1_score(all_targets,pred_labels, average='macro')
    upt_res[2] = f1_score(all_targets,pred_labels, average='micro')
    upt_res[3] = f1_score(all_targets,pred_labels)
    return upt_res


def subgraph2data(link,adj_train,node_attr,neighbor_list,bfs_labels,hop,max_sub_num):
    u,v,y_label = link
    
    nei_u = set(bfs_labels[u].keys())
    nei_v = set(bfs_labels[v].keys())
    if len(nei_u)>max_sub_num-2:
        nei_u = random.sample(nei_u,max_sub_num-2)
        nei_u = set(nei_u)
    if len(nei_v)>max_sub_num-2:
        nei_v = random.sample(nei_v,max_sub_num-2)
        nei_v = set(nei_v)
    
    V_K = nei_u | nei_v
    V_K.add(u)
    V_K.add(v)
    
    dict_lu = {vi:hop+1 for vi in V_K}
    dict_lv = {vi:hop+1 for vi in V_K}
    dict_lu[u] = 0
    dict_lv[v] = 0
    for vi in nei_u:
        dict_lu[vi] = bfs_labels[u][vi]
    for vi in nei_v:
        dict_lv[vi] = bfs_labels[v][vi]
    V_K.discard(u)
    V_K.discard(v)
    V_K = [u,v]+list(V_K)
    
    du = np.array([dict_lu[vi] for vi in V_K])
    dv = np.array([dict_lv[vi] for vi in V_K])
    bs_labels = np.minimum(du,1)*np.minimum(dv,1)*(3*np.minimum(du,dv)\
                        +np.minimum(np.maximum(du-dv,-1),1)-1)+np.minimum(du,1)#+1
    x_bs = np.eye(3*hop+2)[bs_labels.astype(int)]
    if node_attr is not None:
        x = np.hstack((x_bs,node_attr[V_K,:]))
    else:
        x = x_bs
    
    adj_sub = adj_train[V_K,:][:,V_K]
    adj_sub[0,1] = 0
    adj_sub.eliminate_zeros()
    adj_sub = sp.coo_matrix(adj_sub)
    edge_index = np.array([adj_sub.row,adj_sub.col])
    edge_attr = np.array(adj_sub.data)
    edge_attr[np.where(edge_attr==-1)] = 2
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(x, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    data.edge_attr = edge_attr
    data.y = torch.LongTensor([max(0,y_label)])
    return data


def graph2data(edgelist,A,x0,neighbor_list,bfs_labels,hop,max_sub_num):
    partial_worker = partial(subgraph2data,adj_train=A,node_attr=x0,\
                             neighbor_list=neighbor_list,bfs_labels=bfs_labels,
                             hop=hop,max_sub_num=max_sub_num)
    
    data_features = []
    for edge in tqdm(edgelist, desc="Processing edges", unit="edge"):
        res = partial_worker(edge)
        data_features.append(res)
    return data_features

