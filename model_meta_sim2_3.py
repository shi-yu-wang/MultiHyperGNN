# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 23:13:45 2022

@author: Shiyu
"""

# model
import torch.nn as nn
import torch.nn.functional as F
import torch

from hypernetwork_mlp_meta2_3 import *

from dgl import DGLGraph
import networkx as nx
from dgl.nn.pytorch import GATConv

class GCN(nn.Module):
    def __init__(self, ngene_in, ngene_out, nhidatt_in, nhidatt, nhidmlp, nheads, nmlplayer, nattlayer, nhidmlp_hyper, nmeta, alpha):
        super(GCN, self).__init__()
        """
        ngene_in: number of genes from source graph
        ngene_out: number of genes from target graph
        nhidatt: hidden dimension of input and output graph attention network
        nhidmlp: hidden dimension of output MLP
        nheads: number of heads of input and output graph attention network
        nmlplayer: number of output MLP layers
        nattlayer: number of graph attention layers
        
        nhidatt_hyper: hidden dimension of graph attention in hypernetwork
        nheads_hyper: number of heads of graph attention in hypernetwork
        nhidmlp_hyper: hidden dimension of MLP in hypernetwork
        nmeta: dimension of meta data
        """
        self.ngene_in = ngene_in
        self.ngene_out = ngene_out
        self.nheads = nheads
        self.nhidatt = nhidatt
        self.nhidmlp = nhidmlp
        self.nmlplayer = nmlplayer
        self.nattlayer = nattlayer
        self.nhidatt_in = nhidatt_in
        
        # Hypernetwork parameter
        # self.nhidatt_hyper = nhidatt_hyper
        # self.nheads_hyper = nheads_hyper
        self.nhidmlp_hyper = nhidmlp_hyper
        self.nmeta = nmeta
        
        self.alpha = alpha
        
        self.hypernetwork_in = HyperNetworkGAT(nhidmlp = self.nhidmlp_hyper, nmeta = self.nmeta, 
                                         ngene = self.ngene_out, nattlayer_learn = 0, nfeatin_learn = 1, 
                                         nmlplayer_learn = self.nmlplayer, nheads_learn = self.nheads, nhidatt_learn = self.nhidatt_in, nhidmlp_learn = self.nhidmlp)
        
        self.hypernetwork_out1 = HyperNetworkGAT(nhidmlp = self.nhidmlp_hyper, nmeta = self.nmeta, 
                                         ngene = self.ngene_out, nattlayer_learn = self.nattlayer, nfeatin_learn = self.nhidatt_in, 
                                         nmlplayer_learn = self.nmlplayer, nheads_learn = self.nheads, nhidatt_learn = self.nhidatt, nhidmlp_learn = self.nhidmlp)
        
        self.hypernetwork_out2 = HyperNetworkGAT(nhidmlp = self.nhidmlp_hyper, nmeta = self.nmeta, 
                                         ngene = self.ngene_out, nattlayer_learn = 0, nfeatin_learn = 1, 
                                         nmlplayer_learn = self.nmlplayer, nheads_learn = self.nheads, nhidatt_learn = self.nhidatt, nhidmlp_learn = self.nhidmlp)
        # self.hypernetwork_out2 = HyperNetworkGAT(nhidmlp = self.nhidmlp_hyper, nmeta = self.nmeta, 
        #                                  ngene = self.ngene_out, nattlayer_learn = self.nattlayer, nfeatin_learn = self.nhidatt_in, 
        #                                  nmlplayer_learn = self.nmlplayer, nheads_learn = self.nheads, nhidatt_learn = self.nhidatt, nhidmlp_learn = self.nhidmlp)
        
        self.hypernetwork_mlp1 = HyperNetworkMLP(nhidmlp = self.nhidmlp_hyper, nmeta = self.nmeta, 
                                         ngene = self.ngene_out, nattlayer_learn = self.nattlayer, nfeatin_learn = self.nhidatt, 
                                         nmlplayer_learn = self.nmlplayer, nheads_learn = self.nheads, nhidatt_learn = self.nhidatt, nhidmlp_learn = self.nhidmlp)
        
        self.hypernetwork_mlp2 = HyperNetworkMLP(nhidmlp = self.nhidmlp_hyper, nmeta = self.nmeta, 
                                          ngene = self.ngene_out, nattlayer_learn = self.nattlayer, nfeatin_learn = self.nhidatt, 
                                          nmlplayer_learn = self.nmlplayer, nheads_learn = self.nheads, nhidatt_learn = self.nhidatt, nhidmlp_learn = self.nhidmlp)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
        # Innetwork
        self.attentions1 = GATConv(1, self.nhidatt_in, self.nheads)
        # self.attentions2 = GATConv(self.nhidatt_in, self.nhidatt_in, self.nheads)
        self.out_att1 = GATConv(self.nhidatt_in * self.nheads, self.nhidatt_in, 1)
        # self.out_att2 = GATConv(self.nhidatt_in * self.nheads, self.nhidatt_in, 1)
        
        # Outnetwork
        self.attentions3 = GATConv(self.nhidatt, self.nhidatt, self.nheads)
        
        self.attentions4 = GATConv(self.nhidatt, self.nhidatt, self.nheads)
        # for param in self.attentions4.parameters():
        #     param.requires_grad = False
        self.out_att3 = GATConv(self.nhidatt * self.nheads, self.nhidatt, 1)
       
        self.out_att4 = GATConv(self.nhidatt * self.nheads, self.nhidatt, 1)
        # for param in self.out_att4.parameters():
        #     param.requires_grad = False
        
        
        self.attentions5 = GATConv(1, self.nhidatt, self.nheads)
        self.out_att5 = GATConv(self.nhidatt * self.nheads, self.nhidatt, 1)
        
        # Link prediction
        self.attentions_linkpred1 = GATConv(1, self.nhidatt, self.nheads)
        self.attentions_linkpred2 = GATConv(self.nhidatt, self.nhidatt, self.nheads)
        self.out_att_linkpred1 = GATConv(self.nhidatt * self.nheads, self.nhidatt, 1)
        self.out_att_linkpred2 = GATConv(self.nhidatt * self.nheads, self.nhidatt, 1)
        self.pred_link = nn.Sequential(
            nn.Linear(self.nhidatt, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            # nn.Linear(128, 128),
            # nn.ELU(),
            # nn.Linear(128, 128),
            # nn.ELU(),
            # nn.Linear(128, 128),
            # nn.ELU(),
            nn.Linear(128, 128)
            ).cuda()
        
        
    
    def gat(self, g, X, a, w, b, a_out, w_out, b_out, attentions1, attentions_out1, featin, nhidatt):
        # Graph attention
        ## First layer
        a_tmp = a[:, 0, :, :].reshape(2, self.nheads, -1)
        w_tmp = w[:, :featin, :].reshape(-1, featin)
        b_tmp = b[:, :nhidatt].reshape(-1)
        a_out_tmp = a_out[0, :].reshape(2, -1)
        w_out_tmp = w_out[:(nhidatt * self.nheads), :].reshape(-1, nhidatt * self.nheads)
        b_out_tmp = b_out[0, :]
        
        # print(attentions1.fc.weight.shape)
        # print(w_tmp.shape)
        
        attentions1.fc.weight = nn.Parameter(w_tmp)
        attentions1.bias = nn.Parameter(b_tmp)
        attentions1.attn_l = nn.Parameter(a_tmp[0, :, :].view(1, self.nheads, -1))
        attentions1.attn_r = nn.Parameter(a_tmp[1, :, :].view(1, self.nheads, -1))
        
        attentions_out1.fc.weight = nn.Parameter(w_out_tmp)
        attentions_out1.bias = nn.Parameter(b_out_tmp)
        attentions_out1.attn_l = nn.Parameter(a_out_tmp[0, :].view(1, 1, -1))
        attentions_out1.attn_r = nn.Parameter(a_out_tmp[1, :].view(1, 1, -1))
        
        z_2 = attentions1(g, X)
        z_2 = z_2.view(z_2.shape[0], -1)
        z_2 = F.elu(attentions_out1(g, z_2))
        
        z_2 = z_2.view(z_2.shape[0], -1)
        
        return z_2
    
    def forward(self, g_trg_in, g_trg_out, ng_trg, metadata_trg, X_in1, g_in1, metadata_in1, X_in2 = None, g_in2 = None, metadata_in2 = None):
        ########################################################################################################################################### Source encoder
        if g_in1:
            a_in1, w_in1, b_in1, a_out_in1, w_out_in1, b_out_in1 = self.hypernetwork_in(metadata_in1)
            X_in1 = X_in1.view(-1, 1).cuda()
            a_in1 = a_in1.view(self.nheads, -1, 2 * self.nhidatt_in, 1)
            w_in1 = w_in1.view(self.nheads, -1, self.nhidatt_in)
            
            z_in1 = self.gat(g_in1, X_in1[:self.ngene_in], a_in1, w_in1, b_in1, a_out_in1, w_out_in1, b_out_in1, self.attentions1, self.out_att1, 1, self.nhidatt_in)
            
            for param in self.attentions1.parameters():
                param.requires_grad = False
            for param in self.out_att1.parameters():
                param.requires_grad = False
            
        if g_in2:
            a_in2, w_in2, b_in2, a_out_in2, w_out_in2, b_out_in2 = self.hypernetwork_in(metadata_in2)
            X_in2 = X_in2.view(-1, 1).cuda()
            a_in2 = a_in2.view(self.nheads, -1, 2 * self.nhidatt_in, 1)
            w_in2 = w_in2.view(self.nheads, -1, self.nhidatt_in)
            
            z_in2 = self.gat(g_in2, X_in2[:self.ngene_in], a_in2, w_in2, b_in2, a_out_in2, w_out_in2, b_out_in2, self.attentions1, self.out_att1, 1, self.nhidatt_in)
            
            for param in self.attentions1.parameters():
                param.requires_grad = False
            for param in self.out_att1.parameters():
                param.requires_grad = False
        
        
        # Source graph embedding
        if g_in1 and g_in2:
            z_1 = (z_in1 + z_in2) / 2
        elif g_in1:
            z_1 = z_in1
        else:
            z_1 = z_in2
        
        # Output weights via Hypernetwork
        a, w, b, a_out, w_out, b_out = self.hypernetwork_out1(metadata_trg)
        w_mlp, b_mlp = self.hypernetwork_mlp1(metadata_trg)
        """
        Dimension of outputs:
        a: [self.nheads] x [self.nattlayer + 1] x [2 * self.nhidatt] x [1]
        w: [self.nheads] x [self.nfeatin + self.nhidatt * self.nattlayer] x [self.nhidatt]
        b: [self.nheads] x [self.nhidatt * (self.nattlayer + 1)]
        a_out: [self.nattlayer + 1] x [2 * self.nhidatt]
        w_out: [self.nhidatt * self.nheads + self.nhidatt * self.nheads * self.nattlayer] x [self.nhidatt]
        b_out: [self.nattlayer + 1] x [self.nhidatt]
        
        w_mlp: [self.nhidmlp] x [self.nhidatt + self.nmlplayer * self.nhidmlp + 1]
        b_mlp: [1] x [self.nhidmlp * (self.nmlplayer + 1) + 1]
        """
        a = a.view(self.nheads, -1, 2 * self.nhidatt, 1)
        w = w.view(self.nheads, -1, self.nhidatt)
        
        # Graph attention
        ## First layer
        a_tmp = a[:, 0, :, :].reshape(2, self.nheads, -1)
        w_tmp = w[:, :self.nhidatt_in, :].reshape(-1, self.nhidatt_in)
        b_tmp = b[:, :self.nhidatt].reshape(-1)
        a_out_tmp = a_out[0, :].reshape(2, -1)
        w_out_tmp = w_out[:(self.nhidatt * self.nheads), :].reshape(-1, self.nhidatt * self.nheads)
        b_out_tmp = b_out[0, :]
        
        self.attentions3.fc.weight = nn.Parameter(w_tmp)
        self.attentions3.bias = nn.Parameter(b_tmp)
        self.attentions3.attn_l = nn.Parameter(a_tmp[0, :, :].view(1, self.nheads, -1))
        self.attentions3.attn_r = nn.Parameter(a_tmp[1, :, :].view(1, self.nheads, -1))
        
        self.out_att3.fc.weight = nn.Parameter(w_out_tmp)
        self.out_att3.bias = nn.Parameter(b_out_tmp)
        self.out_att3.attn_l = nn.Parameter(a_out_tmp[0, :].view(1, 1, -1))
        self.out_att3.attn_r = nn.Parameter(a_out_tmp[1, :].view(1, 1, -1))
        
        z_2 = self.attentions3(g_trg_in, z_1)
        z_2 = z_2.view(z_2.shape[0], -1)
        z_2 = F.elu(self.out_att3(g_trg_in, z_2))
        
        for param in self.attentions3.parameters():
            param.requires_grad = False
        for param in self.out_att3.parameters():
            param.requires_grad = False
        # print(self.attentions3.fc.weight.requires_grad)
        
        # z_save1 = z_2
        # print(z_2.shape)
        
        # Other layers
        z_2 = z_2.view(-1, self.nhidatt)
        a_tmp2 = a[:, 1, :, :].reshape(2, self.nheads, -1)
        w_tmp2 = w[:, self.nhidatt_in:(self.nhidatt + self.nhidatt_in), :].reshape(-1, self.nhidatt)
        b_tmp2 = b[:, self.nhidatt:(self.nhidatt*2)].reshape(-1)
        a_out_tmp2 = a_out[1, :].reshape(2, -1)
        w_out_tmp2 = w_out[(self.nhidatt * self.nheads):(2*self.nhidatt * self.nheads), :].reshape(-1, self.nhidatt * self.nheads)
        b_out_tmp2 = b_out[1, :]
        
        self.attentions4.fc.weight = nn.Parameter(w_tmp2)
        self.attentions4.bias = nn.Parameter(b_tmp2)
        self.attentions4.attn_l = nn.Parameter(a_tmp2[0, :, :].view(1, self.nheads, -1))
        self.attentions4.attn_r = nn.Parameter(a_tmp2[1, :, :].view(1, self.nheads, -1))
        
        self.out_att4.fc.weight = nn.Parameter(w_out_tmp2)
        self.out_att4.bias = nn.Parameter(b_out_tmp2)
        self.out_att4.attn_l = nn.Parameter(a_out_tmp2[0, :].view(1, 1, -1))
        self.out_att4.attn_r = nn.Parameter(a_out_tmp2[1, :].view(1, 1, -1))

        z_2 = self.attentions4(g_trg_in, z_2)
        z_2 = z_2.view(z_2.shape[0], -1)
        z_2 = F.elu(self.out_att4(g_trg_in, z_2))
        
        z_2 = z_2.view(z_2.shape[0], -1)
        
        for param in self.attentions4.parameters():
            param.requires_grad = False
        for param in self.out_att4.parameters():
            param.requires_grad = False
        
        
        
        # MLP predictor
        ## First layer
        w_mlp_tmp = torch.transpose(w_mlp[:, :self.nhidatt], 0, 1)
        b_mlp_tmp = b_mlp[:, :self.nhidmlp]

        z_2 = F.elu(torch.matmul(z_2, w_mlp_tmp) + b_mlp_tmp)
        
        # Middle layers
        for i in range(self.nmlplayer):
            w_mlp_tmp2 = w_mlp[:, (self.nhidatt + i*self.nhidmlp):(self.nhidatt + i*self.nhidmlp + self.nhidmlp)]
            b_mlp_tmp2 = b_mlp[:, ((i+1)*self.nhidmlp):((i+2)*self.nhidmlp)]
            # bn = nn.BatchNorm1d(self.nhidmlp).cuda()
            z_2 = F.elu(torch.matmul(z_2, w_mlp_tmp2) + b_mlp_tmp2)
        # Last layer
        
        w_mlp_tmp3 = w_mlp[:,-1]
        b_mlp_tmp3 = b_mlp[:,-1]
        y_in_pred = torch.matmul(z_2, w_mlp_tmp3) + b_mlp_tmp3
        
        y_all = torch.cat([y_in_pred.view(-1,1), X_in1[self.ngene_in:].view(-1, 1).cuda()], dim = 0)

        ########################################################################################################################################## Link prediction
        zsemi_lp = self.attentions_linkpred1(g_trg_out, y_all)
        zsemi_lp = zsemi_lp.view(zsemi_lp.shape[0], -1)
        zsemi_lp = F.elu(self.out_att_linkpred1(g_trg_out, zsemi_lp))
        zsemi_lp = self.attentions_linkpred2(g_trg_out, zsemi_lp)
        zsemi_lp = zsemi_lp.view(zsemi_lp.shape[0], -1)
        zsemi_lp = self.out_att_linkpred2(g_trg_out, zsemi_lp)
        zsemi_lp = zsemi_lp.view(zsemi_lp.shape[0], -1)
        zsemi_lp = self.pred_link(zsemi_lp)

        # The whole adjacency matrix
        zsemi_lp1 = zsemi_lp[:ng_trg, :]
        zsemi_lp2 = zsemi_lp[ng_trg:, :]
        A_semi_ori = F.sigmoid(torch.matmul(zsemi_lp1, torch.transpose(zsemi_lp1, 0, 1))).view(zsemi_lp1.shape[0], -1)
        A_semi1 = F.sigmoid(torch.matmul(zsemi_lp1, torch.transpose(zsemi_lp2, 0, 1))).view(zsemi_lp1.shape[0], -1)
        A_semi2 = F.sigmoid(torch.matmul(zsemi_lp2, torch.transpose(zsemi_lp2, 0, 1))).view(zsemi_lp2.shape[0], -1)
        
        ########################################################################################################################################## target decoder
        # y_trg_in = torch.cat([y_in_pred.view(-1, 1), X[self.ngene_in:].view(-1, 1).cuda()], dim = 0)
        
        # # Graph attention
        a_trg, w_trg, b_trg, a_trg_out, w_trg_out, b_trg_out = self.hypernetwork_out2(metadata_trg)
        w_trg_mlp, b_trg_mlp = self.hypernetwork_mlp2(metadata_trg)
        
        
        # ## First layer
        a_trg = a_trg.view(self.nheads, -1, 2 * self.nhidatt, 1)
        w_trg = w_trg.view(self.nheads, -1, self.nhidatt)
        z_trg_2 = self.gat(g_trg_out, y_all, a_trg, w_trg, b_trg, a_trg_out, w_trg_out, b_trg_out, self.attentions5, self.out_att5, 1, self.nhidatt)
        
        for param in self.attentions5.parameters():
            param.requires_grad = False
        for param in self.out_att5.parameters():
            param.requires_grad = False
        
        # MLP predictor
        ## First layer
        w_trg_tmp = torch.transpose(w_trg_mlp[:, :self.nhidatt], 0, 1)
        b_trg_tmp = b_trg_mlp[:, :self.nhidmlp]
        z_trg_2 = F.elu(torch.matmul(z_trg_2, w_trg_tmp) + b_trg_tmp)
        
        # Middle layers
        for i in range(self.nmlplayer):
            w_trg_tmp2 = w_trg_mlp[:, (self.nhidatt + i*self.nhidmlp):(self.nhidatt + i*self.nhidmlp + self.nhidmlp)]
            b_trg_tmp2 = b_trg_mlp[:, ((i+1)*self.nhidmlp):((i+2)*self.nhidmlp)]
            # bn = nn.BatchNorm1d(self.nhidmlp).cuda()
            z_trg_2 = F.elu(torch.matmul(z_trg_2, w_trg_tmp2) + b_trg_tmp2)
        # Last layer
        
        w_trg_tmp3 = w_trg_mlp[:,-1]
        b_trg_tmp3 = b_trg_mlp[:,-1]
        y_out_pred = torch.matmul(z_trg_2, w_trg_tmp3) + b_trg_tmp3
        
        y_pred = torch.cat([y_in_pred.view(-1, 1), y_out_pred[self.ngene_in:].view(-1, 1).cuda()], dim = 0)
        
        return y_in_pred, y_pred, A_semi1, A_semi2, A_semi_ori