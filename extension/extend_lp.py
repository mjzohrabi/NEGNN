import networkx as nx
import torch
import heapq
import numpy as np
import dgl
from numpy.linalg import inv
import math
from tqdm import tqdm
import random
import numpy as np
import dgl.function as fn
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]
        
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        # Computes a scalar score for each edge of the given graph.
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
        
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats, h_feats//2)
        self.W2 = nn.Linear(h_feats//2, 1)

    def apply_edges(self, edges):
        # Computes a scalar score for each edge of the given graph.
        # Tested: max, min, cat, add, mean(add/2)
        h = torch.mul(edges.src['h'], edges.dst['h'])
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']



def centrality_based(centrality_metric, graph, graph_undirected, alpha, beta, dataset):

    # these ones return a Dictionary of nodes with centrality as the value.
    if centrality_metric == 'degree':
        centrality = nx.degree_centrality(graph)
    elif centrality_metric == 'eigenvector':
        graph = nx.DiGraph(graph)
        centrality = nx.eigenvector_centrality(graph)

    important_nodes = heapq.nlargest(alpha, centrality, key=centrality.get)
    
    # TODO, should the selected nodes change for each imp_node?
    new_dataset = dataset
    for item in important_nodes:
        tgt_nodes = set(graph_undirected) - set(graph_undirected[item])
        selected_nodes = torch.tensor(np.random.choice(list(tgt_nodes), beta))
        important_node = torch.ones(len(selected_nodes), dtype=int)*torch.tensor(item)
        new_dataset = dgl.add_edges(new_dataset, selected_nodes, important_node)
        new_dataset = dgl.add_edges(new_dataset, important_node, selected_nodes)    

    return new_dataset


def non_edges_important_nodes(graph, alpha, explore):
    # nodes = set(graph)
    # selected_src_nodes = np.random.choice(list(nodes), int(0.2*len(nodes)), replace=False)
    # centrality = nx.eigenvector_centrality(graph)
    centrality = nx.degree_centrality(graph)
    selected_src_nodes = heapq.nlargest(alpha, centrality, key=centrality.get)

    edges_list = []
    for u in tqdm(selected_src_nodes):
        tgt_nodes = set(graph) - set(graph[u])
        selected_tgt_nodes = np.random.choice(list(tgt_nodes), int(explore*len(tgt_nodes)), replace=False)

        edges = list(zip(np.ones(len(selected_tgt_nodes), dtype=int)*u, selected_tgt_nodes))
        edges_list.extend(edges)

    return edges_list


def resource_allocation_index(G, dataset, gamma, alpha, explore, ebunch=None):
    if ebunch is None:
        ebunch = non_edges_important_nodes(G, alpha, explore)
    
    def predict(u, v):
        return sum(1 / G.degree(w) for w in nx.common_neighbors(G, u, v))

    edges = []
    scores = []
    for edge in tqdm(ebunch):
        u, v = edge
        try:
          pred_val = predict(u, v)
          if pred_val > 0:
              edges.append((u,v))
              scores.append(pred_val)
        except:
          continue

    scores = np.array(scores)
    sum_scores = scores.sum()
    scores /= sum_scores

    rnd_indices = np.random.choice(len(edges), gamma, p=scores, replace=True)
    selected_edges = [edges[i] for i in rnd_indices]

    new_edges = torch.tensor(selected_edges)

    new_edges_1 = new_edges[:,0]
    new_edges_2 = new_edges[:,1]

    new_dataset = dgl.add_edges(dataset, new_edges_1, new_edges_2)
    new_dataset = dgl.add_edges(new_dataset, new_edges_2, new_edges_1)

    return new_dataset


def jaccard_coefficient_index(G, dataset, gamma, alpha, explore, ebunch=None):
    if ebunch is None:
        ebunch = non_edges_important_nodes(G, alpha, explore)
    
    def predict(u, v):
        cnbors = list(nx.common_neighbors(G, u, v))
        union_size = len(set(G[u]) | set(G[v]))
        if union_size == 0:
            return 0
        else:
            return len(cnbors) / union_size

    edges = []
    scores = []
    for edge in tqdm(ebunch):
        u, v = edge
        try:
          pred_val = predict(u, v)
          if pred_val > 0:
              edges.append((u,v))
              scores.append(pred_val)
        except:
          continue

    scores = np.array(scores)
    sum_scores = scores.sum()
    scores /= sum_scores

    rnd_indices = np.random.choice(len(edges), gamma, p=scores, replace=True)
    selected_edges = [edges[i] for i in rnd_indices]

    new_edges = torch.tensor(selected_edges)

    new_edges_1 = new_edges[:,0]
    new_edges_2 = new_edges[:,1]

    new_dataset = dgl.add_edges(dataset, new_edges_1, new_edges_2)
    new_dataset = dgl.add_edges(new_dataset, new_edges_2, new_edges_1)

    return new_dataset


def adamic_adar_index(G, dataset, gamma, alpha, explore, ebunch=None):
    if ebunch is None:
        ebunch = non_edges_important_nodes(G, alpha, explore)

    def predict(u, v):
        return sum(1 / math.log(G.degree(w))
                   for w in nx.common_neighbors(G, u, v))

    edges = []
    scores = []
    for edge in tqdm(ebunch):
        u, v = edge
        try:
          pred_val = predict(u, v)
          if pred_val > 0:
              edges.append((u,v))
              scores.append(pred_val)
        except:
          continue

    scores = np.array(scores)
    sum_scores = scores.sum()
    scores /= sum_scores

    rnd_indices = np.random.choice(len(edges), gamma, p=scores, replace=True)
    selected_edges = [edges[i] for i in rnd_indices]

    new_edges = torch.tensor(selected_edges)

    new_edges_1 = new_edges[:,0]
    new_edges_2 = new_edges[:,1]

    new_dataset = dgl.add_edges(dataset, new_edges_1, new_edges_2)
    new_dataset = dgl.add_edges(new_dataset, new_edges_2, new_edges_1)

    return new_dataset


def extend_neighborhood(dataset, graph_nx, graph_undirected, alpha, beta, gamma, explore, extend_metric):
    if extend_metric == 'adamic_adar':
        print("---Adamic Adar index---")
        extended_graph = adamic_adar_index(graph_undirected, dataset=dataset, gamma=gamma, alpha=alpha, explore=explore, ebunch=None)
    elif extend_metric == 'resource_alloc':
        print("---Resouce Allocation index---")
        extended_graph = resource_allocation_index(graph_undirected, dataset=dataset, gamma=gamma, alpha=alpha, explore=explore, ebunch=None)
    elif extend_metric == 'jaccard':
        print("---Jaccard Coefficient index---")
        extended_graph = jaccard_coefficient_index(graph_undirected, dataset=dataset, gamma=gamma, alpha=alpha, explore=explore, ebunch=None)
    elif extend_metric == 'degree':
        print("---Degree index---")
        extended_graph = centrality_based(centrality_metric='degree', graph=graph_nx, graph_undirected=graph_undirected, alpha=alpha, beta=beta, dataset=dataset)
    else: 
        sys.exit("Not a valid metric")

    return extended_graph