import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.transforms import DropEdge
from dgl import add_self_loop

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, drop_edge_prob=0.5):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.drop_edge = DropEdge(p=drop_edge_prob)
  
    def forward(self, g, in_feat):
        if self.training:
            g = self.drop_edge(g)
            g = add_self_loop(g)
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
