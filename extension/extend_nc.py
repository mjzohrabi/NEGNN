import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import networkx as nx
import argparse
import math 
import heapq
import sys 

class ExtendNeighborhood():

    def __init__(self, graph, graph_undirected, alpha, beta, gamma, explore, extend_metric, ebunch=None):
        self.graph = graph
        self.graph_undirected = graph_undirected
        self.extend_metric = extend_metric
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.explore = explore
        self.ebunch = ebunch

    def calc_score_aa(self, u, v):         
        common_neighs = list(nx.common_neighbors(self.graph_undirected, u, v))
        if len(common_neighs) > 0: 
            # scores = map(lambda x: 1 / math.log(self.graph_undirected.degree(x)), common_neighs)
            scores = []
            for x in common_neighs: 
                d = math.log(self.graph_undirected.degree(x))
                if d > 0:
                    scores.append(1/math.log(d))
                else:
                  continue
            return sum(scores)

    def add_edges_centrality_based(self):
        # these ones return a Dictionary of nodes with centrality as the value.
        if self.extend_metric == 'degree':
            centrality = nx.degree_centrality(self.graph_undirected)
        elif self.extend_metric == 'eigenvector':
            di_graph = nx.DiGraph(self.graph_undirected)
            centrality = nx.eigenvector_centrality(di_graph)

        important_nodes = heapq.nlargest(self.alpha, centrality, key=centrality.get)
        
        # TODO, should the selected nodes change for each imp_node?
        new_dataset = self.graph
        for item in important_nodes:
            tgt_nodes = set(self.graph_undirected) - set(self.graph_undirected[item])
            selected_nodes = torch.tensor(np.random.choice(list(tgt_nodes), self.beta))
            important_node = torch.ones(len(selected_nodes), dtype=int)*torch.tensor(item)
            new_dataset = dgl.add_edges(new_dataset, selected_nodes, important_node)
            new_dataset = dgl.add_edges(new_dataset, important_node, selected_nodes) 

        # new_dataset = self.graph
        # for item in important_nodes:
        #     tgt_nodes = set(self.graph_undirected) - set(self.graph_undirected[item])
        #     important_node = torch.ones(len(tgt_nodes), dtype=int)*torch.tensor(item)
        #     candidate_edges = list(zip(important_node, tgt_nodes))
            
        #     edges = []
        #     scores = []
        #     for edge in candidate_edges:
        #         u, v = edge
        #         try:
        #             pred_val = sum(1 / math.log(self.graph_undirected.degree(w)) for w in nx.common_neighbors(self.graph_undirected, u, v))
        #             # pred_val = calc_score(u, v)
        #             if pred_val > 0:
        #                 edges.append((u,v))
        #                 scores.append(pred_val)
        #         except:
        #             continue
        #     scores = np.array(scores)
        #     sum_scores = scores.sum()
        #     scores /= sum_scores

        #     if len(edges) > self.beta:
        #         rnd_indices = np.random.choice(len(edges), int(self.gamma), p=scores, replace=False)
        #         edges = [edges[i] for i in rnd_indices]
                
        #     extra_random = self.beta - len(edges)
        #     if extra_random > 0:
        #         rnd_indices = np.random.choice(len(candidate_edges), int(extra_random), replace=False)
        #         extra_edges = [candidate_edges[i] for i in rnd_indices]
        #         edges.extend(extra_edges)

        #     selected_edges = torch.tensor(edges)
        #     new_edges_1 = selected_edges[:,0]
        #     new_edges_2 = selected_edges[:,1]

        #     new_dataset = dgl.add_edges(new_dataset, new_edges_1, new_edges_2)
        #     new_dataset = dgl.add_edges(new_dataset, new_edges_2, new_edges_1)

        return new_dataset


    def non_edges_important_nodes(self):
        centrality = nx.degree_centrality(self.graph_undirected)
        scores = list( np.array(list(centrality.values())) / np.sum(list(centrality.values())) )
        selected_src_nodes = np.random.choice(list(centrality.keys()), self.alpha, p=scores, replace=False)
        # selected_src_nodes = heapq.nlargest(self.alpha, centrality, key=centrality.get)
        edges_list = []
        # for u in tqdm(selected_src_nodes):
        for u in selected_src_nodes:
            tgt_nodes = set(self.graph_undirected) - set(self.graph_undirected[u])
            selected_tgt_nodes = np.random.choice(list(tgt_nodes), int(self.explore*len(tgt_nodes)), replace=False)
            edges = list(zip(np.ones(len(selected_tgt_nodes), dtype=int)*u, selected_tgt_nodes))
            edges_list.extend(edges)
        return edges_list

    def adamic_adar_index(self, ebunch):
        def calc_score(u, v):         
            common_neighs = list(nx.common_neighbors(self.graph_undirected, u, v))
            if len(common_neighs) > 0: 
                # scores = map(lambda x: 1 / math.log(self.graph_undirected.degree(x)), common_neighs)
                scores = []
                for x in common_neighs: 
                    d = math.log(self.graph_undirected.degree(x))
                    if d > 0:
                        scores.append(1/math.log(d))
                    else:
                      continue
                return sum(scores)
                
        edges = []
        scores = []
        # for edge in tqdm(ebunch):
        for edge in ebunch:
            u, v = edge
            try:
                pred_val = sum(1 / math.log(self.graph_undirected.degree(w)) for w in nx.common_neighbors(self.graph_undirected, u, v))
                # pred_val = calc_score(u, v)

                if pred_val > 0:
                    edges.append((u,v))
                    scores.append(pred_val)
            except:
                continue
        scores = np.array(scores)
        sum_scores = scores.sum()
        scores /= sum_scores

        # print(len(scores), len(edges))
        return edges, scores

    def resource_allocation_index(self, ebunch):
        def calc_score(u, v):         
            common_neighs = list(nx.common_neighbors(self.graph_undirected, u, v))
            if len(common_neighs) > 0: 
                scores = []
                for x in common_neighs: 
                    d = self.graph_undirected.degree(x)
                    if d > 0:
                        scores.append(d)
                    else:
                      continue
                return sum(scores)

        edges = []
        scores = []
        # for edge in tqdm(ebunch):
        for edge in ebunch:
            u, v = edge
            try:
              pred_val = sum(1 / self.graph_undirected.degree(w) for w in nx.common_neighbors(self.graph_undirected, u, v))
              # pred_val = calc_score(u, v)
              if pred_val > 0:
                  edges.append((u,v))
                  scores.append(pred_val)
            except:
              continue
        scores = np.array(scores, dtype='float64')
        sum_scores = scores.sum()
        scores /= sum_scores
        return edges, scores

    def jaccard_coefficient_index(self, ebunch):
        edges = []
        scores = []
        # for edge in tqdm(ebunch):
        for edge in ebunch:
            u, v = edge
            try:
              cnbors = list(nx.common_neighbors(self.graph_undirected, u, v))
              union_size = len(set(self.graph_undirected[u]) | set(self.graph_undirected[v]))
              if union_size == 0:
                  pred_val = 0
              else:
                  pred_val = len(cnbors) / union_size
              
              if pred_val > 0:
                  edges.append((u,v))
                  scores.append(pred_val)
            except:
              continue
        scores = np.array(scores)
        sum_scores = scores.sum()
        scores /= sum_scores
        return edges, scores


    def add_edges_similarity_based(self):
        if self.alpha == self.graph_undirected.number_of_nodes() and self.explore == 1:
            ebunch = nx.non_edges(self.graph_undirected)
        else:
            ebunch = self.non_edges_important_nodes()

        # ss = nx.adamic_adar_index(self.graph_undirected, ebunch)
        # print(list(ss))
        match self.extend_metric:
            case 'resource_allocation':
                edges, scores = self.resource_allocation_index(ebunch)
            case 'jaccard':
                edges, scores = self.jaccard_coefficient_index(ebunch)
            case 'adamic_adar':
                edges, scores = self.adamic_adar_index(ebunch)
            case _:
                sys.exit("Not a similarity metric!")
        # print(len(edges), len(scores))

        if len(edges) > self.gamma:
            rnd_indices = np.random.choice(len(edges), int(self.gamma), p=scores, replace=False)
            edges = [edges[i] for i in rnd_indices]
            
            
        extra_random = int(self.gamma - len(edges))
        if extra_random > 0:
            rnd_indices = np.random.choice(len(ebunch), int(extra_random), replace=False)
            extra_edges = [ebunch[i] for i in rnd_indices]
            edges.extend(extra_edges)

        selected_edges = torch.tensor(edges)
        new_edges_1 = selected_edges[:,0]
        new_edges_2 = selected_edges[:,1]

        new_dataset = dgl.add_edges(self.graph, new_edges_1, new_edges_2)
        new_dataset = dgl.add_edges(new_dataset, new_edges_2, new_edges_1)

        return new_dataset

