# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 12:17:29 2022

@author: Shiyu
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from dgl import DGLGraph
import networkx as nx
from dgl.nn.pytorch import GATConv

class HyperNetworkGAT(nn.Module):
    def __init__(self, nhidmlp, nmeta, ngene, nattlayer_learn, nfeatin_learn, nmlplayer_learn, nheads_learn, nhidatt_learn, nhidmlp_learn):
        super(HyperNetworkGAT, self).__init__()
        """
        This is the implementation of hypernetwork.
        
        # Parameters of hypernetwork:
        nhidatt: hidden dimension of graph attention
        nheads: number of heads of graph attention
        nhidmlp: hidden dimension of MLP
        nmeta: dimension of meta information
        ngene: number of genes
        nfeatin: dimention of input feature
        
        # Dimension of the network to be learned
        nattlayer_learn: number of graph attention layers to be learned
        nfeatin_learn: dimension of input features
        nmlplayer_learn: number of MLP layers to be learned
        nheads_learn: dimention of heads of graph attention to be learned
        nhidatt_learn: hidden dimension of graph attention to be learned
        nhidmlp_learn: hidden dimension of MLP to be learned
        
        # Dimenstion of parameters to be learned
        ## Graph attention:
        ### First layer
        nheads_learn x (2 x nhidatt_learn)
        nheads_learn x (nfeatin_learn x nhidatt_learn)
        ### Other layers
        nheads_learn x nattlayer_learn x (2 x nhidatt_learn)
        nheads_learn x nattlayer_learn x (nhidatt_learn x nhidatt_learn)
        ## MLP
        ### First layer
        nhidatt_learn x nhidmlp_learn
        ### Other layers
        nmlplayer_learn x (nhidmlp_learn x nhidmlp_learn)
        ### Last layer
        nhidmlp_learn x 1
        """
        
        # self.nhidatt = nhidatt
        # self.nheads = nheads
        self.nhidmlp = nhidmlp
        self.nmeta = nmeta
        self.ngene = ngene
        
        self.nattlayer_learn = nattlayer_learn
        self.nfeatin_learn = nfeatin_learn
        self.nmlplayer_learn = nmlplayer_learn
        self.nheads_learn = nheads_learn
        self.nhidatt_learn = nhidatt_learn
        self.nhidmlp_learn = nhidmlp_learn
        
        # Graph attention layers to produce graph embedding
        # self.attentions1 = GATConv(self.nfeatin, self.nhidatt, self.nheads).cuda()
        # self.attentions2 = GATConv(self.nhidatt, self.nhidatt, self.nheads).cuda()
        
        # self.out_attention1 = GATConv(self.nhidatt * self.nheads, self.nhidatt, 1).cuda()
        # self.out_attention2 = GATConv(self.nhidatt * self.nheads, 1, 1).cuda()
        
        # MLP to predict kernel
        ## First layer of graph attention parameter
        ### a
        """
        Dimension of a: [nheads_learn] x [(self.nattlayer_learn + 1) * 2 * self.nhidatt_learn]
        """
        self.pred_att_a1 = nn.Sequential(
            nn.Linear(self.nmeta, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nheads_learn * self.nmeta)
            ).cuda()
        self.pred_att_a2 = nn.Sequential(
            nn.Linear(self.nmeta, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, (self.nattlayer_learn + 1) * 2 * self.nhidatt_learn)
            ).cuda()
        
        ### w
        """
        Dimension of w: [nheads_learn] x [(nfeatin_learn + nhidatt_learn * nattlayer_learn) * self.nhidatt_learn]
        """
        self.pred_att_w1 = nn.Sequential(
            nn.Linear(self.nmeta, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nheads_learn * self.nmeta)
            ).cuda()
        
        self.pred_att_w2 = nn.Sequential(
            nn.Linear(self.nmeta, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, (self.nfeatin_learn + self.nhidatt_learn * self.nattlayer_learn) * self.nhidatt_learn)
            ).cuda()
        
        ### b
        """
        Dimension of b: [1] x [nhidatt_learn * (nattlayer_learn + 1) + 1]
        """
        self.pred_att_b1 = nn.Sequential(
            nn.Linear(self.nmeta, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nheads_learn * self.nmeta)
            ).cuda()
        
        self.pred_att_b2 = nn.Sequential(
            nn.Linear(self.nmeta, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidatt_learn * (self.nattlayer_learn + 1))
            ).cuda()
        
        ### a_out
        """
        Dimension of a: [self.nattlayer_learn + 1] x [2 * self.nhidatt_learn]
        """
        self.pred_att_a_out1 = nn.Sequential(
            nn.Linear(self.nmeta, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, (self.nattlayer_learn + 1) * self.nmeta)
            ).cuda()
        self.pred_att_a_out2 = nn.Sequential(
            nn.Linear(self.nmeta, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp,  2 * self.nhidatt_learn)
            ).cuda()
        
        ### w_out
        """
        Dimension of w: [self.nfeatin_learn + self.nhidatt_learn * self.nattlayer_learn] x [self.nhidatt_learn]
        """
        self.pred_att_w_out1 = nn.Sequential(
            nn.Linear(self.nmeta, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, (self.nhidatt_learn * self.nheads_learn + self.nhidatt_learn * self.nheads_learn * self.nattlayer_learn) * self.nmeta)
            ).cuda()
        
        self.pred_att_w_out2 = nn.Sequential(
            nn.Linear(self.nmeta, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidatt_learn)
            ).cuda()
        
        ### b_out
        """
        Dimension of b: [self.nattlayer_learn + 1] x [self.nhidatt_learn]
        """
        self.pred_att_b_out1 = nn.Sequential(
            nn.Linear(self.nmeta, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, (self.nattlayer_learn + 1) * self.nmeta)
            ).cuda()
        
        self.pred_att_b_out2 = nn.Sequential(
            nn.Linear(self.nmeta, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidatt_learn)
            ).cuda()
        
    def forward(self, metadata):
        # Embedding of meta information
        # z = self.attentions1(A, X)
        # z = z.view(z.shape[0], -1)
        # z = F.elu(self.out_attention1(A, z))
        
        # z = self.attentions2(A, z)
        # z = z.view(z.shape[0], -1)
        # z = F.elu(self.out_attention2(A, z))
        # z = z.view(1, -1)
        # z = torch.cat((z, metadata), 1)
        z = metadata
        
        # Predict graph attention parameters
        a = self.pred_att_a1(z)
        a = a.view(self.nheads_learn, -1)
        a = self.pred_att_a2(a)
        
        w = self.pred_att_w1(z)
        w = w.view(self.nheads_learn, -1)
        w = self.pred_att_w2(w)
        
        b = self.pred_att_b1(z)
        b = b.view(self.nheads_learn, -1)
        b = self.pred_att_b2(b)
        
        a_out = self.pred_att_a_out1(z)
        a_out= a_out.view(self.nattlayer_learn + 1, -1)
        a_out = self.pred_att_a_out2(a_out)
        
        w_out = self.pred_att_w_out1(z)
        w_out = w_out.view(self.nhidatt_learn * self.nheads_learn + self.nhidatt_learn * self.nheads_learn * self.nattlayer_learn, -1)
        w_out = self.pred_att_w_out2(w_out)
        
        b_out = self.pred_att_b_out1(z)
        b_out = b_out.view(self.nattlayer_learn + 1, -1)
        b_out = self.pred_att_b_out2(b_out)
        

        return a, w, b, a_out, w_out, b_out
class HyperNetworkMLP(nn.Module):
    def __init__(self, nhidmlp, nmeta, ngene, nattlayer_learn, nfeatin_learn, nmlplayer_learn, nheads_learn, nhidatt_learn, nhidmlp_learn):
        super(HyperNetworkMLP, self).__init__()
        """
        This is the implementation of hypernetwork.
        
        # Parameters of hypernetwork:
        nhidatt: hidden dimension of graph attention
        nheads: number of heads of graph attention
        nhidmlp: hidden dimension of MLP
        nmeta: dimension of meta information
        ngene: number of genes
        nfeatin: dimention of input feature
        
        # Dimension of the network to be learned
        nattlayer_learn: number of graph attention layers to be learned
        nfeatin_learn: dimension of input features
        nmlplayer_learn: number of MLP layers to be learned
        nheads_learn: dimention of heads of graph attention to be learned
        nhidatt_learn: hidden dimension of graph attention to be learned
        nhidmlp_learn: hidden dimension of MLP to be learned
        
        # Dimenstion of parameters to be learned
        ## Graph attention:
        ### First layer
        nheads_learn x (2 x nhidatt_learn)
        nheads_learn x (nfeatin_learn x nhidatt_learn)
        ### Other layers
        nheads_learn x nattlayer_learn x (2 x nhidatt_learn)
        nheads_learn x nattlayer_learn x (nhidatt_learn x nhidatt_learn)
        ## MLP
        ### First layer
        nhidatt_learn x nhidmlp_learn
        ### Other layers
        nmlplayer_learn x (nhidmlp_learn x nhidmlp_learn)
        ### Last layer
        nhidmlp_learn x 1
        """
        
        # self.nhidatt = nhidatt
        # self.nheads = nheads
        self.nhidmlp = nhidmlp
        self.nmeta = nmeta
        self.ngene = ngene
        
        self.nattlayer_learn = nattlayer_learn
        self.nfeatin_learn = nfeatin_learn
        self.nmlplayer_learn = nmlplayer_learn
        self.nheads_learn = nheads_learn
        self.nhidatt_learn = nhidatt_learn
        self.nhidmlp_learn = nhidmlp_learn
        
        # MLP to predict kernel
        # MLP parameter
        ### w
        """
        Dimension of w: [nhidmlp_learn] x [nhidatt_learn + nmlplayer_learn * nhidmlp_learn + 1]
        """
        self.pred_mlp_w1 = nn.Sequential(
            nn.Linear(self.nmeta, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp_learn * self.nmeta)
            ).cuda()
        
        self.pred_mlp_w2 = nn.Sequential(
            nn.Linear(self.nmeta, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidatt_learn + self.nmlplayer_learn * self.nhidmlp_learn + 1)
            ).cuda()
        
        ### b
        """
        Dimension of b: [1] x [nhidmlp_learn * (nmlplayer_learn + 1) + 1]
        """
        self.pred_mlp_b = nn.Sequential(
            nn.Linear(self.nmeta, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp),
            nn.ELU(),
            nn.Linear(self.nhidmlp, self.nhidmlp_learn * (self.nmlplayer_learn + 1) + 1)
            ).cuda()
        
        
    def forward(self, metadata):
        z = metadata
        
        # Predict MLP parameters
        w_mlp = self.pred_mlp_w1(z)
        w_mlp = w_mlp.view(self.nhidmlp_learn, -1)
        w_mlp = self.pred_mlp_w2(w_mlp)
        
        b_mlp = self.pred_mlp_b(z)

        return w_mlp, b_mlp