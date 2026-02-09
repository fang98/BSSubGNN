

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
import scipy.sparse as sp
from torch_geometric.data import Data
from functools import partial
from tqdm import tqdm
from torch_geometric.utils import from_networkx, train_test_split_edges, add_self_loops, negative_sampling, k_hop_subgraph
from torch_geometric.datasets import Planetoid, CitationFull, Flickr, Twitch, Coauthor
from torch_geometric.data import InMemoryDataset
import pandas as pd
import os.path as osp
import pickle


class Disease(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['disease.pt']

    def process(self):
        path = '../data/disease_lp/'
        edges = pd.read_csv(path + 'disease_lp.edges.csv')
        labels = np.load(path + 'disease_lp.labels.npy')
        features = sp.load_npz(path + 'disease_lp.feats.npz').todense()
        dataset = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=torch.tensor(edges.values).t().contiguous(),
            y=torch.tensor(labels)
        )
        torch.save(dataset, self.processed_paths[0])


class Airport(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['airport.pt']

    def process(self):
        data_path = '../data/airport'
        dataset_str = 'airport'
        graph = pickle.load(open(osp.join(data_path, dataset_str + '.p'), 'rb'))
        dataset = from_networkx(graph)
        dataset.x = dataset.feat
        dataset.feat = None
        torch.save(dataset, self.processed_paths[0])


def load_data(data_name):
    if data_name == 'cora':
        dataset = Planetoid('../datasets/Planetoid', name='Cora')
    elif data_name == 'cora_ml':
        dataset = CitationFull('../datasets/CitationFull', name='Cora_Ml')
    elif data_name == 'citeseer':
        dataset = CitationFull('../datasets/CitationFull', name='CiteSeer')
    elif data_name == 'pubmed':
        dataset = Planetoid('../datasets/Planetoid', name='PubMed')
    elif data_name == 'airport':
        dataset = Airport('../datasets/Airport')
    elif data_name == 'disease':
        dataset = Disease('../datasets/Disease')
    elif data_name == 'twitch_en':
        dataset = Twitch('../datasets/Twitch', name='EN')
    elif data_name == 'cs':
        dataset = Coauthor('../datasets/Coauthor', name='cs')
    else:
        raise ValueError('Invalid dataset!')
    return dataset


def process(args,dataset):
    hop = args.hop
    data = dataset[0]
    data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.10)
    edge_index = data.train_pos_edge_index
    data.train_neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1)
    )
    edge_index = data.train_pos_edge_index
    train_pos_edge_index = data.train_pos_edge_index
    train_neg_edge_index = data.train_neg_edge_index
    val_pos_edge_index = data.val_pos_edge_index
    val_neg_edge_index = data.val_neg_edge_index
    test_pos_edge_index = data.test_pos_edge_index
    test_neg_edge_index = data.test_neg_edge_index
    
    train_pos_edge_index = torch.cat([train_pos_edge_index,val_pos_edge_index],1)
    train_neg_edge_index = torch.cat([train_neg_edge_index,val_neg_edge_index],1)
    edge_index = train_pos_edge_index
    
    G = nx.Graph()
    G.add_nodes_from(list(range(data.num_nodes)))
    for u,v in train_pos_edge_index.t().tolist():
        G.add_edge(u,v)
    num_nodes = G.number_of_nodes()
    neighbor_list = []
    for vi in range(num_nodes):
        nei = set(list(G.neighbors(vi)))
        neighbor_list.append([nei])
    if hop>1:
        for vi in range(num_nodes):
            nei2 = set()
            for vj in neighbor_list[vi][0]:
                nei2 = nei2.union(neighbor_list[vj][0])
            nei2.discard(vi)
            nei2 = nei2-neighbor_list[vi][0]
            neighbor_list[vi].append(nei2)
    adj_train = nx.adjacency_matrix(G)
    if args.using_attr:
        node_attr = data.x.cpu().numpy()
    else:
        node_attr = None
    
    train_edges = [[u,v,1] for u,v in train_pos_edge_index.t().tolist()]+[[u,v,0] for u,v in train_neg_edge_index.t().tolist()]
    test_edges = [[u,v,1] for u,v in test_pos_edge_index.t().tolist()]+[[u,v,0] for u,v in test_neg_edge_index.t().tolist()]
    return train_edges,test_edges,adj_train,node_attr,neighbor_list


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





