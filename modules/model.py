import pickle
import dgl

import torch
import torch.nn as nn
import dgl.function as fn
import os
import sys
from .utils import *
import torch.nn.functional as F
from tqdm import trange


class Dataloader:
    def __init__(self, g, features, k, dataset_name = None):
        self.k = k
        self.g = g
        self.label_zeros = torch.zeros(1, g.number_of_nodes()).to(features.device)
        self.label_ones = torch.ones(1, g.number_of_nodes()).to(features.device)
        self.en = features
        if dataset_name is not None and os.path.isfile(f"./2hop_aggre/{dataset_name}.pickle"):
            print(f"Load precomputed graph emb from ./2hop_aggre/{dataset_name}.pickle")
            with open(f"./2hop_aggre/{dataset_name}.pickle", "rb") as fp:
                precomputed = pickle.load(fp)
                self.weight = precomputed["weight"].to(features.device)
                self.features_weighted = precomputed["features_weighted"].to(features.device)
                self.eg = precomputed["eg"].to(features.device)
        else:
            print("Preprocessing: Aggregrate neighbour embeddings")
            load_or_precompute_two_hop(g, dataset_name)

            with open(f"2_hop_map/{dataset_name}.pkl", "rb") as f:
                self.two_hop_map = pickle.load(f)
            g_2hop = build_two_hop_graph(g.num_nodes(), self.two_hop_map)
            g_2hop = g_2hop.to(features.device)

            self.weight = get_diag(g_2hop, self.k)
            agg_feat = aggregation(g_2hop, features, k=1) 
            self.features_weighted = (features.swapaxes(1, 0) * self.weight).swapaxes(1, 0).detach()
            self.eg = (agg_feat - self.features_weighted).detach()

            if dataset_name is not None:
                print(f"Save graph emb to ./2hop_aggre/{dataset_name}.pickle")
                if not os.path.isdir("./2hop_aggre"):
                    os.makedirs("./2hop_aggre")
                with open(f"./2hop_aggre/{dataset_name}.pickle", "wb") as fp:
                    pickle.dump({
                        "weight": self.weight.to("cpu"),
                        "features_weighted": self.features_weighted.to("cpu"),
                        "eg": self.eg.to("cpu")
                    }, fp)

    def get_data(self, epoch=-1):
        x_ego = self.en
        x_2hop = self.eg
        return x_ego, x_2hop
    

def build_two_hop_graph(num_nodes, two_hop_map):
    src = []
    dst = []

    for v, neighbors in two_hop_map.items():
        for u in neighbors:
            src.append(u)  # 메시지 보내는 쪽
            dst.append(v)  # 메시지 받는 쪽

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    g_2hop = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    return g_2hop


def aggregation(graph, feat, k):
    with graph.local_scope():
        # compute normalization
        degs = graph.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm = norm.to(feat.device).unsqueeze(1)
        # compute (D^-1 A^k D^-1)^k X
        for _ in range(k):
            feat = feat * norm
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_u('h', 'm'),
                             fn.sum('m', 'h'))
            feat = graph.ndata.pop('h')
            feat = feat * norm
        return feat

def get_diag(graph, k):
    aggregated_matrix = aggregation(
        graph,
        torch.eye(graph.num_nodes(), graph.num_nodes()).to(graph.device),
        k
    )
    return torch.diag(aggregated_matrix)


from collections import defaultdict
from tqdm import tqdm

def precompute_two_hop(g):
    two_hop_map = defaultdict(set)
    total_edges = 0
    for v in tqdm(range(g.num_nodes()), desc="Precomputing 2-hop neighbors"):
        one_hop = set(g.successors(v).tolist())
        for u in one_hop:
            two_hop_map[v].update(g.successors(u).tolist())
        two_hop_map[v] -= one_hop  # 1-hop 제거
        two_hop_map[v].discard(v)
        total_edges += len(two_hop_map[v])

    print(f"📏 Total number of 2-hop edges: {total_edges}")
    return dict(two_hop_map)

def load_or_precompute_two_hop(g, dataset_name, base_dir="2_hop_map"):
    os.makedirs(base_dir, exist_ok=True)
    cache_path = os.path.join(base_dir, f"{dataset_name}.pkl")

    if os.path.exists(cache_path):
        print(f"[INFO] Loading cached two-hop map from: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        print(f"[INFO] Cache not found. Precomputing two-hop map for dataset '{dataset_name}'...")
        two_hop_map = precompute_two_hop(g)
        with open(cache_path, "wb") as f:
            pickle.dump(two_hop_map, f)
        print(f"[INFO] Saved two-hop map to: {cache_path}")
        return two_hop_map
   


class Discriminator(nn.Module):
    def __init__(self, in_dim, hid_dim, bn=False):
        super().__init__()

        self.coef = nn.Sequential(
            nn.Linear(in_dim, hid_dim),        
            nn.Linear(hid_dim, hid_dim),
            )
        
        self.mlp_g = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hid_dim) if bn else nn.Identity()
        )
        self.mlp_n = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hid_dim) if bn else nn.Identity()
        )


    def reg_edge(self, emb, eps=1e-8):
        h = emb / (emb.norm(dim=1, keepdim=True) + eps)  
        N = h.size(0)
        if N <= 1:
            return h.new_tensor(0.)

        sv = h.sum(dim=0)                                
        per_node = (h @ sv - 1.0) / (N - 1)             

        loss_reg = per_node.sum() / N 
        return loss_reg

    def forward(self, features, summary, xlx):

        h1 = self.mlp_g(features)     
        h2 = self.mlp_n(summary)       
        h1 = h1*self.coef(xlx)

        # cosine similarity
        s = torch.nn.functional.cosine_similarity(h1, h2)  
        uni_loss = self.reg_edge(h2)
        
        return s.unsqueeze(0), h1, uni_loss

class MCGAD(nn.Module):
    def __init__(self, g, feature, n_in, n_hidden, bn, edge_index, beta):
        super().__init__()
        self.g = g
        self.feature = feature.to(self.g.device)
        self.bn = bn 
        self.discriminator = Discriminator(n_in, n_hidden, bn)
        self.node_num = self.feature.shape[0]
        self.beta = beta

        lap_cpu = get_lap(edge_index, self.feature.shape[0]).coalesce() 
        feat_cpu = self.feature.detach().cpu()                            

        xt_l = torch.sparse.mm(lap_cpu, feat_cpu)                        
        xtlx_diag = torch.sum(feat_cpu * xt_l, dim=0)                    
        xlx_vec = torch.sigmoid(xtlx_diag).to(self.g.device)        
        self.register_buffer("xlx", xlx_vec) 

    def forward(self, target_features, neighbour_features):
        score, h1, uni_loss = self.discriminator(
            target_features.detach(),
            neighbour_features.detach(),
            self.xlx, 
        )

        if not self.training:
            center = h1.mean(dim=0, keepdim=True)
            
            h1_norm = F.normalize(h1, p=2, dim=1)        
            center_norm = F.normalize(center, p=2, dim=1) 

            cos_sim = torch.sum(h1_norm * center_norm, dim=1)  
         
            dist = 1 - cos_sim  
            dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-9)
            score = (score - score.min()) / (score.max() - score.min() + 1e-9)

            score = (1-self.beta)*score - self.beta*dist.unsqueeze(0)
            return score
        
        return score, uni_loss
        