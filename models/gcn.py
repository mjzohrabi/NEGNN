import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

class GCN(nn.Module):
  def __init__(self, in_feats, h_feats):
      super(GCN, self).__init__()
      self.conv1 = GraphConv(in_feats, h_feats)
      self.conv2 = GraphConv(h_feats, h_feats)
  
  def forward(self, g, in_feat):
      h = self.conv1(g, in_feat)
      h = F.relu(h)
      h = self.conv2(g, h)
      return h