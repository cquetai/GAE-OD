"""
Pytorch2.40, cuda12.1,
python -V: Python 3.11.5
pip show torch
Name: torch Version: 2.4.0+cu121
python -c "import torch; print(torch.__version__)"：
2.4.0+cu121
"""
#去掉离群节点聚类
# -*- coding: utf-8 -*-
"""
1) Remove outlier nodes using IsolationForest on structural features
2) Train Graph Attention Auto-Encoder (GAT encoder + inner-product decoder)
3) Extract embeddings, cluster with KMeans
4) Evaluate ACC, NMI, ARI, F1
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx, from_networkx, negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.nn import GATConv
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
from scipy.optimize import linear_sum_assignment
import networkx as nx
from collections import Counter
import random
import os
import subprocess
'''
# 设置环境变量
os.environ['PATH'] = '/usr/local/cuda-11.3/bin:$PATH'
os.environ['LD_LIBRARY_PATH'] ='/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH'

# 获取环境变量

my_variable = os.getenv('PATH')
print(my_variable)
my_variable = os.getenv('LD_LIBRARY_PATH')
print(my_variable)
'''
subprocess.run("bash -c 'source ~/.bashrc'", shell=True, capture_output=True, text=True)
torch.cuda.device_count()

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
print(torch.cuda.device_count())
# -----------------------
# Utilities for evaluation
# -----------------------
def cluster_acc(y_true, y_pred):
    """
    Compute clustering accuracy with Hungarian matching
    y_true, y_pred are 1d arrays of same length
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

# -----------------------
# GAT Auto-Encoder model
# -----------------------
class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.6):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # after multi-head, features multiply by heads; use lin to reduce
        self.lin = nn.Linear(hidden_channels * heads, out_channels)
        self.act = nn.ELU()
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)  # final embedding
        return x

class GATAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, latent_dim=16, heads=4, dropout=0.6):
        super().__init__()
        self.encoder = GATEncoder(in_channels, hidden_channels, latent_dim, heads=heads, dropout=dropout)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return z

# -----------------------
# Training helper: reconstruction loss with negative sampling
# -----------------------
def reconstruction_loss(z, pos_edge_index, num_nodes, device, neg_ratio=1.0):
    # pos_edge_index: [2, E_pos]
    # compute pos scores
    src, dst = pos_edge_index
    z_src = z[src]
    z_dst = z[dst]
    pos_logits = (z_src * z_dst).sum(dim=1)  # inner product

    # negative sampling
    num_neg = int(pos_logits.size(0) * neg_ratio)
    neg_edge = negative_sampling(
        edge_index=pos_edge_index, num_nodes=num_nodes, num_neg_samples=num_neg
    ).to(device)
    nsrc, ndst = neg_edge
    neg_logits = (z[nsrc] * z[ndst]).sum(dim=1)

    # BCE with logits: pos labels = 1, neg labels = 0
    loss_pos = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
    loss_neg = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
    return loss_pos + loss_neg

# -----------------------
# Main pipeline
# -----------------------
def run_pipeline(dataset_name='Cora',
                 remove_outlier=True,
                 contamination=0.05,
                 latent_dim=32,
                 epochs=200,
                 lr=0.005,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 random_seed=42):
    # reproducibility
 
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print('device:', device)
    # load dataset
    from torch_geometric.datasets import WikiCS
    if dataset_name=='WikiCS':
        dataset = WikiCS(root='WikiCS')
    else:
        dataset = Planetoid(root=f'data', name=dataset_name)
    data = dataset[0]
    #print(data)
    num_nodes = data.num_nodes
    #print('num_nodes:', num_nodes)

    # If node features missing, use identity
    if data.x is None:
        x_features = torch.eye(num_nodes, dtype=torch.float)
    else:
        x_features = data.x.clone()
       # exit()
    # Convert to networkx to compute structural features
    G_nx = to_networkx(data, to_undirected=True)
    # ensure numeric node ordering matches torch indexing (0..N-1)
    G_nx = nx.convert_node_labels_to_integers(G_nx, ordering='sorted')

    # compute structural features for outlier detection
    deg = np.array([d for _, d in G_nx.degree()], dtype=float).reshape(-1, 1)
    #print('deg:', deg)
    
    clustering = np.array([nx.clustering(G_nx, n) for n in G_nx.nodes()], dtype=float).reshape(-1, 1)
    #print('clustering:', clustering)
    
    try:
        pagerank = np.array([v for _, v in nx.pagerank(G_nx).items()], dtype=float).reshape(-1, 1)
    except Exception:
        pagerank = np.zeros((num_nodes, 1))
    #print('pagerank:', pagerank)
 
    try:
        betw = np.array([v for _, v in nx.betweenness_centrality(G_nx).items()], dtype=float).reshape(-1, 1)
    except Exception:
        betw = np.zeros((num_nodes, 1))
    #print('betw:', betw)
    
    struct_feats = np.hstack([deg, clustering, pagerank, betw])
    #print('struct_feats:', struct_feats)
    
    scaler = StandardScaler()
    struct_feats_scaled = scaler.fit_transform(struct_feats)
    #print('struct_feats_scaled:', struct_feats_scaled)
    
    # identify outliers
    if remove_outlier:
        iso = IsolationForest(contamination=contamination, random_state=random_seed)
        iso.fit(struct_feats_scaled)
        is_inlier = iso.predict(struct_feats_scaled)  # 1 for inliers, -1 for outliers
        print("is_inlier:",is_inlier)
        #exit()
        
        mask = (is_inlier == 1)
        keep_idx = np.where(mask)[0]
        removed_idx = np.where(~mask)[0]
        print(f"Total nodes: {num_nodes}, kept: {len(keep_idx)}, removed (outliers): {len(removed_idx)}")
    else:
        keep_idx = np.arange(num_nodes)
        removed_idx = np.array([], dtype=int)
        print("Skipping outlier removal.")

    # create new data object with only kept nodes
    # mask node features and labels
    x_keep = x_features[keep_idx]
    y = data.y.numpy()
    y_keep = y[keep_idx]
 
    # filter edges: keep edges where both ends are in keep_idx
    node_id_map = {old: new for new, old in enumerate(keep_idx)}
    edge_index = data.edge_index.numpy()
    srcs = edge_index[0]
    dsts = edge_index[1]
    keep_edge_mask = np.isin(srcs, keep_idx) & np.isin(dsts, keep_idx)
    srcs_kept = srcs[keep_edge_mask]
    dsts_kept = dsts[keep_edge_mask]
    # remap node ids
    srcs_remap = np.array([node_id_map[int(s)] for s in srcs_kept])
    dsts_remap = np.array([node_id_map[int(d)] for d in dsts_kept])
    edge_index_kept = torch.tensor(np.vstack([srcs_remap, dsts_remap]), dtype=torch.long)

    # build new data-like minimal object
    class SimpleData:
        pass
    new_data = SimpleData()
    new_data.x = x_keep
    new_data.edge_index = edge_index_kept
    new_data.y = torch.tensor(y_keep, dtype=torch.long)
    new_data.num_nodes = x_keep.size(0)

    # model
    model = GATAE(in_channels=new_data.x.size(1), hidden_channels=64, latent_dim=latent_dim, heads=4, dropout=0.6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # train
    model.train()
    new_data.x = new_data.x.to(device)
    new_data.edge_index = new_data.edge_index.to(device)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        z = model(new_data.x, new_data.edge_index)
        loss = reconstruction_loss(z, new_data.edge_index, new_data.num_nodes, device, neg_ratio=1.0)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    # extract embeddings
    model.eval()
    with torch.no_grad():
        z = model(new_data.x, new_data.edge_index).cpu().numpy()

    # clustering
    n_clusters = len(np.unique(y_keep))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=20)
    pred = kmeans.fit_predict(z)

    # evaluations
    acc = cluster_acc(y_keep, pred)
    nmi = normalized_mutual_info_score(y_keep, pred)
    ari = adjusted_rand_score(y_keep, pred)
    f1_macro = f1_score(y_keep, pred, average='macro')

    print("\nClustering Results on kept nodes:")
    print(f"ACC: {acc:.4f}")
    print(f"NMI: {nmi:.4f}")
    print(f"ARI: {ari:.4f}")
    print(f"F1-macro: {f1_macro:.4f}")

    # return results and objects for further analysis
    results = {
        'removed_idx': removed_idx,
        'keep_idx': keep_idx,
        'embeddings': z,
        'pred': pred,
        'y_true': y_keep,
        'acc': acc, 'nmi': nmi, 'ari': ari, 'f1_macro': f1_macro
    }
    return results
#citeseer  Cora pubmed WikiCS
if __name__ == '__main__':
    res = run_pipeline(
                       dataset_name='Cora',
                       remove_outlier=True,
                       contamination=0.005,
                       latent_dim=32,
                       epochs=200,
                       lr=0.005,
                       device='cuda' if torch.cuda.is_available() else 'cpu',
                       random_seed=42)