import dgl
import pandas as pd
import os
from config import const_args as args
import argparse
import sys
sys.path.append('../')
import numpy as np
import torch
import scipy.sparse as sp
import networkx as nx
try:
    from stellargraph.data import EdgeSplitter
except:
    print('no stellargraph')



def undirected_label2directed_label(adj, edge_pairs, task, ratio):
    """
    Convert undirected edge labels to directed edge labels based on the specified task and ratio.

    Parameters:
    - adj (torch.Tensor): Adjacency matrix representing the graph.
    - edge_pairs (torch.Tensor): Tensor containing edge pairs in the format (source, destination, label).
    - task (str or int): Specifies the task to perform. 'train_di' for training directed graphs, 0 for undirected existence,
      1 for existence prediction, 2 for direction prediction and 3 for four-type classification.
    - ratio (float): Ratio used for adjusting the balance between positive and negative edges.

    Returns:
    - new_edge_pairs (torch.Tensor): Filtered tensor containing directed edge pairs after label conversion.
    - new_labels (torch.Tensor): Corresponding labels for the filtered directed edge pairs.
    """
    
    new_edge_pairs = edge_pairs.clone()
    new_edge_pairs = new_edge_pairs.unique(dim=0)# no duplicates
    
    # Create a sparse tensor to mark the presence of edges
    values = torch.ones(len(new_edge_pairs))# select
    h = torch.sparse_coo_tensor(new_edge_pairs[:,:2].T, values, adj.shape)
    m1 = adj.mul(h)+adj.T.mul(h)
    m2 = adj.mul(h)-adj.T.mul(h)
    
    # here the order of edges no more exist
    type_1 = m1.coalesce().indices().T[m1.coalesce().values()==2.] # bidirected edge pairs(N,2)
    type_2 = m2.coalesce().indices().T[m2.coalesce().values()==1.] # positive edges
    type_3 = m2.coalesce().indices().T[m2.coalesce().values()==-1.] # reverse edges
    type_4 = new_edge_pairs[new_edge_pairs[:,2]==0][:,:2] # non-existent
    df = pd.DataFrame({
        'src':type_1.min(dim=1).values.tolist()+type_2[:,0].tolist()+type_3[:,1].tolist()+type_4.min(dim=1).values.tolist(),
        'dst':type_1.max(dim=1).values.tolist()+type_2[:,1].tolist()+type_3[:,0].tolist()+type_4.max(dim=1).values.tolist(),
        'type':[2]*len(type_1)+[1]*len(type_2)+[1]*len(type_3)+[3]*len(type_4) # (min, max) and positive only
    })
    df2 = df.drop_duplicates(subset=['src','dst'])
    new_edge_pairs = torch.zeros((len(df2),3),dtype=int)
    new_edge_pairs[:,0] = torch.tensor(list(df2["src"]))
    new_edge_pairs[:,1] = torch.tensor(list(df2["dst"]))
    new_edge_pairs[:,2] = torch.tensor(list(df2["type"]))
    new_edge_pairs = new_edge_pairs.unique(dim=0)
    labels = new_edge_pairs[:,2]
    new_labels = labels.clone()
    
    # Adjust samples based on the task and ratio
    if task == 'train_di':
        # For training directed graphs, convert bidirectional edges to positive edges
        bi_edges = new_edge_pairs[new_edge_pairs[:,2] == 2].clone()
        bi_edges[:,2] = 1
        new_edge_pairs[:,2][new_edge_pairs[:,2] == 2] = 1
        bi_edges = torch.index_select(bi_edges, 1, torch.tensor([1,0,2]))
        new_edge_pairs = torch.cat((new_edge_pairs, bi_edges), dim=0)
        new_labels = new_edge_pairs[:,2].clone()
        new_labels[new_labels==0] = -1 
        new_labels[new_labels==3] = -1 
        
    elif task == 0: # undirected existence
        new_labels[labels == 3] = 0 # non-existent
        new_labels[labels == 2] = 1 # bi-directional
        pos_num = (new_labels == 1).sum()
        neg_num = (new_labels == 0).sum()

        if  ratio*pos_num > neg_num: # pos>neg
            pos = np.where(new_labels == 1)[0] # pos
            rng = np.random.default_rng(1)
            pos_half = rng.choice(pos, size= (pos_num-int(neg_num/ratio)).item(), replace=False) # balance
            new_labels[pos_half] = -1
        elif ratio*pos_num < neg_num: # neg>pos
            neg = np.where(new_labels == 0)[0] # neg
            rng = np.random.default_rng(1)
            neg_half = rng.choice(neg, size= (neg_num-int(ratio*pos_num)).item(), replace=False) # balance
            new_labels[neg_half] = -1

    elif task==1: # Existence Prediction
        # Randomly turn 1/3 (positive + bidirectional) of the positive edges into reverse edges, and randomly select 1/3 (positive + bidirectional) of non-existent edges.
        pos_num = (labels == 1).sum()
        neg_num = (labels == 3).sum()
        bi_num = (labels == 2).sum()
        new_labels[labels == 3] = -1
        pos = np.where(labels == 1)[0] # pos 
        rng = np.random.default_rng(1)
        pos_half = rng.choice(pos, size= int(1/3*(pos_num+bi_num)), replace=False)
        new_labels[pos_half] = 0
        src = new_edge_pairs[pos_half,0]
        dst = new_edge_pairs[pos_half,1]
        new_edge_pairs[pos_half,0] = dst
        new_edge_pairs[pos_half,1] = src

        neg = np.where(new_labels == -1)[0] # neg 
        rng = np.random.default_rng(1)
        neg_half = rng.choice(neg, size= int(1/3*(pos_num+bi_num)), replace=False) 
        new_labels[neg_half] = 0
        
        new_labels[labels == 2] = 1
    else:
        # step1: ensure the number of pos and neg edges
        pos_num = (labels == 1).sum()
        neg_num = (labels == 3).sum()
        bi_num = (labels == 2).sum()
        if  0.5*ratio*pos_num > neg_num: 
            pos = np.where(labels == 1)[0] 
            rng = np.random.default_rng(1)
            pos_half = rng.choice(pos, size= int(pos_num-int(neg_num/(0.5*ratio))), replace=False) 
            new_labels[pos_half] = -1
        elif 0.5*ratio*pos_num < neg_num: 
            neg = np.where(labels == 3)[0]
            rng = np.random.default_rng(1)
            neg_half = rng.choice(neg, size= int(neg_num-0.5*ratio*pos_num), replace=False)
            new_labels[neg_half] = -1
        if bi_num >= pos_num:
            bi = np.where(labels == 2)[0] # bi
            rng = np.random.default_rng(1)
            bi_half = rng.choice(bi, size= (bi_num-pos_num).item(), replace=False)
            new_labels[bi_half] = -1

        # step2: get reverse
        pos = np.where(labels == 1)[0]
        new_labels[pos[int(len(pos)/2):]] = 0
        s = new_edge_pairs[pos[int(len(pos)/2):],0]
        t = new_edge_pairs[pos[int(len(pos)/2):],1]
        new_edge_pairs[pos[int(len(pos)/2):],0] = t
        new_edge_pairs[pos[int(len(pos)/2):],1] = s
        
        if task == 2: # if direction prediction
            new_labels[(new_labels==2)|(new_labels==3)] = -1
    
    new_edge_pairs[:,2] = new_labels
    return new_edge_pairs[new_labels >= 0], new_labels[new_labels >= 0]


def split_data(args, graph, save_path, seed, task):
    '''
    Split the input graph into training, testing, and validation edges with labels.

    Parameters:
    - args: Arguments from the command line.
    - graph: The input graph to be split.
    - save_path: The directory path where the split data will be saved.
    - seed: Random seed for reproducibility.
    - task: Integer indicating the type of task (1, 2, 3).

    Returns:
    None, saves the split data to files.

    Steps:
    1. Save the whole graph if not already saved.
    2. Convert the graph to a scipy sparse matrix for manipulation.
    3. Split test edges and labels, and save them.
    4. Split validation edges and labels, and save them.
    5. Generate training edges with labels and save them.

    Note:
    - If task is in [1, 2, 3], additional processing is done to convert undirected labels to directed labels.
    - Negative edges are sampled for training based on the positive edges.
    '''
    
    pn_ratio = 1.0 # Set the positive-negative ratio for edge sampling
    
    # Create directories if they don't exist
    if not os.path.exists(save_path+str(seed)):
        os.makedirs(save_path+str(seed))
    if not os.path.exists(save_path+'whole.graph.txt'):
        dgl.data.utils.save_graphs(save_path+'whole.graph', graph)
        edges = torch.zeros(graph.num_edges(),2)
        edges[:,0] = graph.edges()[0]
        edges[:,1] = graph.edges()[1]
        np.savetxt(save_path+'whole.graph.txt',edges, fmt='%i')
    
    # Convert the graph to a scipy sparse matrix
    A = graph.adj()
    row=A._indices()[0]
    col=A._indices()[1]
    data=A._values()
    shape=A.size()
    A_sp=sp.csr_matrix((data, (row, col)), shape=shape)
    
    G = nx.from_scipy_sparse_array(A_sp) # create an undirected graph based on the adjacency

# ---- test -----
    edge_splitter_test = EdgeSplitter(G)
    G_test, test_edges, test_labels = edge_splitter_test.train_test_split(p=float(args.test_val_ratio[0]), method="global", keep_connected=True, seed = seed)
    if task in [1,2,3]:
        test_edges = np.hstack((test_edges,test_labels.reshape(-1,1)))
        di_test_edges, _ = undirected_label2directed_label(A, torch.tensor(test_edges), task, pn_ratio)
        print('sampled test edges',np.unique(di_test_edges[:,2],return_counts=True))
        np.savetxt(save_path+str(seed)+'/test_{}.txt'.format(task), di_test_edges, delimiter=',', fmt='%i')

# --- val ----
    edge_splitter_val = EdgeSplitter(G_test)
    G_val, val_edges, val_labels = edge_splitter_val.train_test_split(p=float(args.test_val_ratio[1]), method="global", keep_connected=True, seed = seed)
    
    if task in [1,2,3]:
        val_edges = np.hstack((val_edges, val_labels.reshape(-1,1)))
        di_val_edges, _ = undirected_label2directed_label(A, torch.tensor(val_edges), task, pn_ratio)
        print('sampled validation edges',np.unique(di_val_edges[:,2],return_counts=True))
        np.savetxt(save_path+str(seed)+'/val_{}.txt'.format(task), di_val_edges, delimiter=',', fmt='%i')

# ---- train ----
    train_graph = dgl.from_networkx(G_val)
    train_src = train_graph.edges()[0]
    train_dst = train_graph.edges()[1]
    train_labels = torch.ones(train_dst.shape)
    train_edges = torch.hstack((train_src.reshape(-1,1),train_dst.reshape(-1,1),train_labels.reshape(-1,1))) # undirected edges

# undirected edges to directed edges
    if task in [1,2,3]:
        pos_train_edges, _ = undirected_label2directed_label(A, train_edges, 'train_di', pn_ratio)
        print('sampled train edges',np.unique(pos_train_edges[:,2],return_counts=True))
        np.savetxt(save_path+str(seed)+'/train_di.txt', pos_train_edges, delimiter=',', fmt='%i')
        
        train_eid = graph.edge_ids(pos_train_edges[:,0], pos_train_edges[:,1])
        bi_graph = dgl.add_edges(graph, graph.reverse().edges()[0], graph.reverse().edges()[1])
        neg_sampler = dgl.dataloading.negative_sampler.GlobalUniform(2)
        neg_edges = neg_sampler(bi_graph, train_eid) 
        neg_train_edges = torch.hstack((neg_edges[0].reshape(-1,1),neg_edges[1].reshape(-1,1),torch.zeros(neg_edges[0].shape).reshape(-1,1)))
        all_train_edges = torch.vstack((pos_train_edges, neg_train_edges))
        di_train_edges, di_labels_train = undirected_label2directed_label(A, all_train_edges, task, pn_ratio)
        print('sampled train edges',np.unique(di_train_edges[:,2],return_counts=True))
        np.savetxt(save_path+str(seed)+'/train_{}.txt'.format(task), di_train_edges, delimiter=',', fmt='%i')


def unique(x, dim=None):
    """
    Returns the unique elements of x along with the indices of those unique elements.

    Parameters:
    - x: Input tensor.
    - dim: Dimension along which to compute uniqueness. If None, the uniqueness is computed over the entire tensor.

    Returns:
    - unique: Tensor containing the unique elements of x.
    - inverse: Indices of the unique elements in the original tensor x.

    Reference:
    - https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810
    """
    
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)



if not os.path.exists('./edge_data'):
    os.makedirs('./edge_data')

twitter = pd.read_csv('../datasets/out.munmun_twitter_social',skiprows=2, delimiter=' ', header=None)
twitter_graph = dgl.graph((twitter[0], twitter[1]))
dgl.save_graphs('./edge_data/twitter/whole.graph',twitter_graph)

cora = pd.read_csv('../datasets/out.subelj_cora_cora',skiprows=2, delimiter=' ', header=None)
cora_graph = dgl.graph((cora[0], cora[1]))
dgl.save_graphs('./edge_data/cora/whole.graph',cora_graph)

epinions = pd.read_csv('../datasets/soc-Epinions1.txt',skiprows=4, delimiter='\t', header=None)
epinions_graph = dgl.graph((epinions[0], epinions[1]))
dgl.save_graphs('./edge_data/epinions/whole.graph',epinions_graph)


args = argparse.Namespace(**args)

for dataset in ['cora','epinions','twitter']:
    for task in [1,2,3]:
        for seed in range(10):
            print(dataset,task,seed)
            graph = dgl.load_graphs('./edge_data/%s/whole.graph'%(dataset))[0][0]
            save_path = './edge_data/%s/'%(dataset)
            split_data(args, graph, save_path, seed, task)



