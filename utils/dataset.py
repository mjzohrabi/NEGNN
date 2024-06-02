import dgl
import torch
import numpy as np
import networkx as nx

def load_dataset(dataset_name, verbose=False):
    if dataset_name == "cora": 
        dataset = dgl.data.CoraGraphDataset(verbose=False)
    elif dataset_name == "citeseer":
        dataset = dgl.data.CiteseerGraphDataset(verbose=False)
    elif dataset_name == "pubmed":
        dataset = dgl.data.PubmedGraphDataset(verbose=False)
    elif dataset_name == "amazonphoto":
        dataset = dgl.data.AmazonCoBuyPhotoDataset(verbose=False)
    elif dataset_name == "amazoncomputer":
        dataset = dgl.data.AmazonCoBuyComputerDataset(verbose=False)
    else: 
      raise("Not defined!")

    graph_nx = dgl.to_networkx(dataset[0])
    graph_undirected = nx.Graph(graph_nx)
    g = dataset[0]

    if (dataset_name == 'amazonphoto') or (dataset_name == 'amazoncomputer'): 
        length = g.num_nodes()
        train_num = int(np.floor(0.7*length))
        valid_num = int(np.floor(0.1*length))
        test_num = int(np.ceil(0.2*length))
        a = (np.zeros(train_num))
        a_1 = (np.ones(valid_num))
        a_2 = (np.ones(test_num)*2)

        mask = np.hstack((a, a_1))
        mask = np.hstack((mask, a_2))

        np.random.shuffle(mask)

        train_mask = torch.tensor(np.where(mask==0,True,False))
        val_mask = torch.tensor( np.where(mask==1,True,False))
        test_mask = torch.tensor(np.where(mask==2,True,False))

        features = g.ndata['feat']
        labels = g.ndata['label']
        num_feats = features.shape[1]
        n_classes = dataset.num_classes
        n_edges = g.num_edges()

    else:
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        num_feats = features.shape[1]
        n_classes = dataset.num_classes 
        n_edges = g.number_of_edges()

    # if verbose:
    #     print("""----Data statistics------'
    #       #Edges %d
    #       #Classes %d
    #       #Train samples %d
    #       #Val samples %d
    #       #Test samples %d""" %
    #           (n_edges, n_classes,
    #             train_mask.int().sum().item(),
    #             val_mask.int().sum().item(),
    #             test_mask.int().sum().item()))
        
    return g, graph_nx, graph_undirected, features, labels, train_mask, val_mask, test_mask, num_feats, n_classes, n_edges


def load_dataset_link_prediction(dataset_name):
    match dataset_name:
        case "cora": 
            dataset = dgl.data.CoraGraphDataset(verbose=False)
        case "citeseer":
            dataset = dgl.data.CiteseerGraphDataset(verbose=False)
        case "pubmed":
            dataset = dgl.data.PubmedGraphDataset(verbose=False)
        case "amazonphoto":
            dataset = dgl.data.AmazonCoBuyPhotoDataset(verbose=False)
        case "amazoncomputer":
            dataset = dgl.data.AmazonCoBuyComputerDataset(verbose=False)
        case _:
            raise("Not defined!")
    
    return dataset