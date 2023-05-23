# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:41:12 2023

@author: Shiyu
"""


import pickle
from model_meta_sim2_3 import *
import random
from torch import nn, optim
import numpy as np
import torch.nn.functional as F
from scipy import stats
import pandas as pd
from scipy import stats

import dgl
from dgl import DGLGraph
import networkx as nx

import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

expr_wb = pd.read_csv("expr_in_whole_blood.csv", header = 0)
expr_ms = pd.read_csv("expr_in_muscle.csv", header = 0)
expr_l = pd.read_csv("expr_out_lung.csv", header = 0)
expr_sse = pd.read_csv("expr_out_skin_sun_exposed.csv", header = 0)
expr_nsse = pd.read_csv("expr_out_skin_not_sun_exposed.csv", header = 0)

# Import dataset
# adj matrix of whole blood
adj_wb = pd.read_csv("graph_in_whole_blood.csv", header = 0)
adj_wb = adj_wb.iloc[:, 1:]

adj_ms = pd.read_csv("graph_in_muscle.csv", header = 0)
adj_ms = adj_ms.iloc[:, 1:]

adj_l = pd.read_csv("graph_out_lung.csv", header = 0)
adj_l = adj_l.iloc[:, 1:]

adj_sse = pd.read_csv("graph_out_skin_sun_exposed.csv", header = 0)
adj_sse = adj_sse.iloc[:, 1:]

adj_nsse = pd.read_csv("graph_out_skin_not_sun_exposed.csv", header = 0)
adj_nsse = adj_nsse.iloc[:, 1:]

ng = expr_wb.shape[0]

ng_src = adj_wb.shape[0]
ng_l = adj_l.shape[0]
ng_sse = adj_sse.shape[0]
ng_nsse = adj_nsse.shape[0]

expr_wb = torch.from_numpy(expr_wb.iloc[:, 1:].to_numpy())
expr_ms = torch.from_numpy(expr_ms.iloc[:, 1:].to_numpy())
expr_l = torch.from_numpy(expr_l.iloc[:, 1:].to_numpy())
expr_sse = torch.from_numpy(expr_sse.iloc[:, 1:].to_numpy())
expr_nsse = torch.from_numpy(expr_nsse.iloc[:, 1:].to_numpy())


adj_wb = torch.tensor(adj_wb.values).float()
adj_l = torch.tensor(adj_l.values).float()
adj_ms = torch.tensor(adj_ms.values).float()
adj_sse = torch.tensor(adj_sse.values).float()
adj_nsse = torch.tensor(adj_nsse.values).float()

adj_wb = adj_wb + torch.eye(adj_wb.shape[0])
adj_l = adj_l + torch.eye(adj_l.shape[0])
adj_ms = adj_ms + torch.eye(adj_ms.shape[0])
adj_sse = adj_sse + torch.eye(adj_sse.shape[0])
adj_nsse = adj_nsse + torch.eye(adj_nsse.shape[0])


adj_src, adj_dst = np.nonzero(adj_wb.numpy())
adj_wb = dgl.graph((adj_src, adj_dst)).to(device)
adj_src, adj_dst = np.nonzero(adj_ms.numpy())
adj_ms = dgl.graph((adj_src, adj_dst)).to(device)

adj_src, adj_dst = np.nonzero(adj_l[:ng_src, :ng_src].numpy())
adj_l_in = dgl.graph((adj_src, adj_dst)).to(device)
adj_src, adj_dst = np.nonzero(adj_l.numpy())
adj_all_src = np.append(adj_src, np.arange(ng_l, expr_wb.shape[0]))
adj_all_dst = np.append(adj_dst, np.arange(ng_l, expr_wb.shape[0]))
adj_l_all = dgl.graph((adj_all_src, adj_all_dst)).to(device)

adj_src, adj_dst = np.nonzero(adj_sse[:ng_src, :ng_src].numpy())
adj_sse_in = dgl.graph((adj_src, adj_dst)).to(device)
adj_src, adj_dst = np.nonzero(adj_sse.numpy())
adj_all_src = np.append(adj_src, np.arange(ng_sse, expr_wb.shape[0]))
adj_all_dst = np.append(adj_dst, np.arange(ng_sse, expr_wb.shape[0]))
adj_sse_all = dgl.graph((adj_all_src, adj_all_dst)).to(device)

adj_src, adj_dst = np.nonzero(adj_nsse[:ng_src, :ng_src].numpy())
adj_nsse_in = dgl.graph((adj_src, adj_dst)).to(device)
adj_src, adj_dst = np.nonzero(adj_nsse.numpy())
adj_all_src = np.append(adj_src, np.arange(ng_nsse, expr_wb.shape[0]))
adj_all_dst = np.append(adj_dst, np.arange(ng_nsse, expr_wb.shape[0]))
adj_nsse_all = dgl.graph((adj_all_src, adj_all_dst)).to(device)

idx = [i for i in range(expr_wb.shape[1])]
random.Random(111).shuffle(idx)
# Training sample
idx_train = idx[:180]
idx_test = idx[181:]

idx_train = torch.as_tensor(idx_train)
idx_test = torch.as_tensor(idx_test)

expr_wb_train = expr_wb[:, idx_train].float()
expr_wb_test = expr_wb[:, idx_test].float()
expr_ms_train = expr_ms[:, idx_train].float()
expr_ms_test = expr_ms[:, idx_test].float()

expr_l_train = expr_l[:, idx_train].float()
expr_l_test = expr_l[:, idx_test].float()

expr_sse_train = expr_sse[:, idx_train].float()
expr_sse_test = expr_sse[:, idx_test].float()

expr_nsse_train = expr_nsse[:, idx_train].float()
expr_nsse_test = expr_nsse[:, idx_test].float()

expr_wb_train = torch.log(expr_wb_train + 1)
expr_wb_test = torch.log(expr_wb_test + 1)
expr_ms_train = torch.log(expr_ms_train + 1)
expr_ms_test = torch.log(expr_ms_test + 1)

expr_l_train = torch.log(expr_l_train + 1)
expr_l_test = torch.log(expr_l_test + 1)

expr_sse_train = torch.log(expr_sse_train + 1)
expr_sse_test = torch.log(expr_sse_test + 1)

expr_nsse_train = torch.log(expr_nsse_train + 1)
expr_nsse_test = torch.log(expr_nsse_test + 1)

meta_l = torch.tensor([[1,0,0,1,0,0,0,0,1,0,0,1,1,0,0]])
meta_ms = torch.tensor([[0,1,0,1,1,1,0,1,0,1,0,0,0,1,0]])
meta_sse = torch.tensor([[0,0,1,0,1,0,1,0,0,0,1,0,0,0,1]])
meta_nsse = torch.tensor([[0,0,1,1,0,0,0,0,1,0,1,0,0,0,1]])
meta_wb = torch.tensor([[1,1,1,1,1,1,0,0,1,0,0,1,0,0,0]])



model = GCN(ngene_in = ng_src, ngene_out = ng_src, nhidatt_in = 256, nhidatt = 256, nhidmlp = 800, nheads = 4, nmlplayer= 6, nattlayer = 1, nhidmlp_hyper = 4800, nmeta = 15, alpha = 0.03).cuda()
model.load_state_dict(torch.load("model.pt")['model_state_dict'])
model.eval()

mse_l = []
pc_l = []

mse_sse = []
pc_sse = []

mse_nsse = []
pc_nsse = []

y_test_l = []
y_test_sse = []
y_test_nsse = []

y_test_l_pred = []
y_test_sse_pred = []
y_test_nsse_pred = []

l1_m = None
criterion = nn.MSELoss()
with torch.no_grad():
    for i in range(len(idx_test)):
        X_in1 = expr_wb_test[:, i].float()
        X_in2 = expr_ms_test[:, i].float()
        
        y_l = expr_l_test[:, i].float().cuda()
        y_test_l.append(y_l.detach().cpu().numpy())
        y_in_pred_l, y_pred_l, A_semi1_l, A_semi2_l, A_semi_ori_l = model(g_trg_in=adj_l_in, g_trg_out=adj_l_all, ng_trg = ng_l, metadata_trg = meta_l.float().cuda(), 
                                                    X_in1=X_in1.cuda(), g_in1 = adj_wb, metadata_in1 = meta_wb.float().cuda(), 
                                                    X_in2=X_in2.cuda(), g_in2 = adj_ms, metadata_in2 = meta_ms.float().cuda())
        y_pred_l = y_pred_l.view(-1, 1)
        y_test_l_pred.append(y_pred_l.detach().cpu().numpy())
        y_l = y_l.view(-1, 1)
        
        y_sse = expr_sse_test[:, i].float().cuda()
        y_test_sse.append(y_sse.detach().cpu().numpy())
        y_in_pred_sse, y_pred_sse, A_semi1_sse, A_semi2_sse, A_semi_ori_sse = model(g_trg_in=adj_sse_in, g_trg_out=adj_sse_all, ng_trg = ng_sse, metadata_trg = meta_sse.float().cuda(),
                           X_in1=X_in1.cuda(), g_in1 = adj_wb, metadata_in1 = meta_wb.float().cuda(), 
                           X_in2=X_in2.cuda(), g_in2 = adj_ms, metadata_in2 = meta_ms.float().cuda())
        y_pred_sse = y_pred_sse.view(-1, 1)
        y_test_sse_pred.append(y_pred_sse.detach().cpu().numpy())
        y_sse = y_sse.view(-1, 1)
        
        y_nsse = expr_nsse_test[:, i].float().cuda()
        y_test_nsse.append(y_nsse.detach().cpu().numpy())
        y_in_pred_nsse, y_pred_nsse, A_semi1_nsse, A_semi2_nsse, A_semi_ori_nsse = model(g_trg_in=adj_nsse_in, g_trg_out=adj_nsse_all, ng_trg = ng_nsse, metadata_trg = meta_nsse.float().cuda(),
                            X_in1=X_in1.cuda(), g_in1 = adj_wb, metadata_in1 = meta_wb.float().cuda(), 
                            X_in2=X_in2.cuda(), g_in2 = adj_ms, metadata_in2 = meta_ms.float().cuda())
        y_pred_nsse = y_pred_nsse.view(-1, 1)
        y_test_nsse_pred.append(y_pred_nsse.detach().cpu().numpy())
        y_nsse = y_nsse.view(-1, 1)
        
        print("Iteration: ", i)
        mse_l.append(criterion(y_pred_l, y_l.view(-1, 1).cuda()).detach().cpu().numpy())
        pc_l.append(np.absolute(np.corrcoef(y_pred_l.view(-1).detach().cpu().numpy(), y_l.view(-1).detach().cpu().numpy())[0, 1]))
        
        mse_sse.append(criterion(y_pred_sse, y_sse.view(-1, 1).cuda()).detach().cpu().numpy())
        pc_sse.append(np.absolute(np.corrcoef(y_pred_sse.view(-1).detach().cpu().numpy(), y_sse.view(-1).detach().cpu().numpy())[0, 1]))
        
        mse_nsse.append(criterion(y_pred_nsse, y_nsse.view(-1, 1).cuda()).detach().cpu().numpy())
        pc_nsse.append(np.absolute(np.corrcoef(y_pred_nsse.view(-1).detach().cpu().numpy(), y_nsse.view(-1).detach().cpu().numpy())[0, 1]))

# prediction accuracy
print("MSE in lung: ", sum(mse_l) / len(mse_l))
# Pearson correlation
print("PCC in lung: ", sum(pc_l) / len(pc_l))

print("MSE in SSE: ", sum(mse_sse) / len(mse_sse))
# Pearson correlation
print("PCC in SSE: ", sum(pc_sse) / len(pc_sse))

print("MSE in NSSE: ", sum(mse_nsse) / len(mse_nsse))
# Pearson correlation
print("PCC in NSSE", sum(pc_nsse) / len(pc_nsse))
