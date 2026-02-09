
import random
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
import scipy.sparse as sp
from torch_geometric.data import Data
from functools import partial
from tqdm import tqdm
import scipy.io as scio


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def read_data(data_name):
    adj = scio.loadmat('../datasets/'+data_name)['net'] #dataset data
    G = nx.from_numpy_matrix(adj.A)
    if data_name=='NS':
        deg = np.array([nx.degree(G,i) for i in G.nodes()])
        idx = np.where(deg!=0)[0]
        adj = adj[idx,:][:,idx]
        G = nx.from_numpy_matrix(adj.A)
    return adj,G


def training_graph(G,p,hop,is_sampled=True,r=1):
    edges = list(G.edges())
    num_edges = len(edges)
    num_test_edges = round(p*num_edges)
    num_train_edges = num_edges-num_test_edges
    G2 = G.copy()
    del_egdes = random.sample(edges,num_test_edges)
    G2.remove_edges_from(del_egdes)
    train_pos_edges = list(G2.edges())
    test_pos_edges = del_egdes
    
    non_edges = list(nx.non_edges(G))
    ind = list(np.random.permutation(len(non_edges)))
    ind_train_non_edges = ind[:int(r*num_train_edges)]
    if is_sampled:
        ind_test_non_edges = ind[int(r*num_train_edges):int(r*num_train_edges)+int(num_test_edges)]
    else:
        ind_test_non_edges = ind[int(r*num_train_edges):]
    train_non_edges = [non_edges[i] for i in ind_train_non_edges]
    test_non_edges = [non_edges[i] for i in ind_test_non_edges]
    
    train_edges = [[u,v,1] for u,v in train_pos_edges]+[[u,v,0] for u,v in train_non_edges]
    test_edges = [[u,v,1] for u,v in test_pos_edges]+[[u,v,0] for u,v in test_non_edges]
    
    adj_train = nx.adjacency_matrix(G2)
    
    num_nodes = G2.number_of_nodes()
    neighbor_list = []
    for vi in range(num_nodes):
        nei = set(list(G2.neighbors(vi)))
        neighbor_list.append([nei])
    if hop>1:
        for vi in range(num_nodes):
            nei2 = set()
            for vj in neighbor_list[vi][0]:
                nei2 = nei2.union(neighbor_list[vj][0])
            nei2.discard(vi)
            nei2 = nei2-neighbor_list[vi][0]
            neighbor_list[vi].append(nei2)
    
    # node_attr = None
    node_attr = []
    for vi in range(num_nodes):
        node_attr.append([len(neighbor_list[vi][0]),len(neighbor_list[vi][1])])
    node_attr = np.array(node_attr)/100
    
    return G2, adj_train, train_edges, test_edges, neighbor_list, node_attr


def subgraph2data(link,adj_train,node_attr,neighbor_list,hop):
    u,v,y_label = link
    dist_dict = {}
    V_K = set()
    for h in range(hop):
        V_K = V_K | neighbor_list[u][h]
        V_K = V_K | neighbor_list[v][h]
    V_K.discard(u)
    V_K.discard(v)
    dist_dict[u] = np.array([np.inf,np.inf])
    dist_dict[v] = np.array([np.inf,np.inf])
    for vi in V_K:
        dist_dict[vi] = np.array([np.inf,np.inf])
    
    for h in range(hop):
        for vi in neighbor_list[u][h]:
            dist_dict[vi][0] = h+1
        for vi in neighbor_list[v][h]:
            dist_dict[vi][1] = h+1
    dist_dict[u][0] = 0
    dist_dict[v][1] = 0
    V_K = [u,v]+list(V_K)
    dist_label = np.array([dist_dict[vi] for vi in V_K])
    
    du = dist_label[:,0]
    dv = dist_label[:,1]
    bs_labels = np.minimum(du,1)*np.minimum(dv,1)*(3*np.minimum(du,dv)\
                        +np.minimum(np.maximum(du-dv,-1),1)-1)+np.minimum(du,1)#+1
    x_bs = np.eye(3*hop+2)[bs_labels.astype(int)]
    if node_attr is not None:
        x = np.hstack((x_bs,node_attr[V_K,:]))
    else:
        x = x_bs
    
    a = adj_train[V_K,:][:,V_K]
    a[0,1] = 0
    a[1,0] = 0
    a.eliminate_zeros()
    tmp = sp.coo_matrix(a)
    row = tmp.row
    col = tmp.col
    edge_index = np.array([row,col])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(x, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    data.y = torch.LongTensor([y_label])
    return data


def graph2data(edgelist,A,x0,neighbor_list,hop):
    partial_worker = partial(subgraph2data,adj_train=A,node_attr=x0,\
                             neighbor_list=neighbor_list,hop=hop)
    
    data_features = []
    for edge in tqdm(edgelist, desc="Processing edges", unit="edge"):
        res = partial_worker(edge)
        data_features.append(res)
    return data_features


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
    auc = roc_auc_score(all_targets,all_scores)
    ap = average_precision_score(all_targets,all_scores)
    return auc,ap