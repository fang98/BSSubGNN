# Subgraph Encoding with Bicentric Sphere Node Labeling and Pooling for Link Prediction

## Requirements
- Python 3.7
- PyTorch 1.8.1
- CUDA 11.1
-dgl 0.9.1
-torch_geometric 2.0.2


## Plain Networks

Instructions for reproducing the experiments reported in plain networks.

### Data Preparation

To preprocess data, follow these steps:

1. Navigate to the `./plain_graph/src/` directory using the command `cd ./plain_graph/src/`.
2. Execute `python preprocessing.py` to create the train/validation/test sets for the link prediction task.

### Training & Evaluation

To conduct experiments, follow these steps:

1. Navigate to the `./plain_graph/src/` directory using the command `cd ./plain_graph/src/`.
2. Execute `python main.py` for link prediction in plain networks.

Example command:
```bash
python main.py --dataset=Celegans --seed=1
``` 


## Attribute Networks

Instructions for reproducing the experiments reported in attribute networks.

### Data Preparation

To preprocess data, follow these steps:

1. Navigate to the `./attr_graph/datases/` directory using the command `cd ./attr_graph/datases/`.
2. Execute `unzip datasets.zip` to unzip the datasets.
3. Navigate to the `./attr_graph/src/` directory using the command `cd ./attr_graph/src/`.
4. Execute `python preprocessing.py` to create the train/validation/test sets for the link prediction task.

### Training & Evaluation

To conduct experiments, follow these steps:

1. Navigate to the `./attr_graph/src/` directory using the command `cd ./attr_graph/src/`.
2. Execute `python main.py` for link prediction in plain networks.

Example command:
```bash
python main.py --dataset=citeseer
``` 


## Directed Networks

Instructions for reproducing the experiments reported in directed networks.

### Data Preparation

To preprocess data, follow these steps:

1. Navigate to the `./directed_graph/src/` directory using the command `cd ./directed_graph/src/`.
2. Execute `python preprocessing.py` to create the train/validation/test sets for the link prediction task.

### Training & Evaluation

To conduct experiments, follow these steps:

1. Navigate to the `./src/` directory using the command `cd ./src/`.
2. Execute `python main.py` for link prediction in directed networks.

Example command:
```bash
python main.py --dataset=cora --task=1 --seed=0
``` 


## Signed Directed Networks

Instructions for reproducing the experiments reported in signed directed networks.

### Data Preparation

To preprocess data, follow these steps:

1. Navigate to the `./signed_directed_graph/src/` directory using the command `cd ./signed_directed_graph/src/`.
2. Execute `python preprocessing.py` to create the train/test sets for the link prediction task.

### Training & Evaluation

To conduct experiments, follow these steps:

1. Navigate to the `./signed_directed_graph/src/` directory using the command `./signed_directed_graph/src/`.
2. Execute `python main.py` for link prediction in directed networks.

Example command:
```bash
python main.py --dataset=BitcoinAlpha --seed=1
``` 


