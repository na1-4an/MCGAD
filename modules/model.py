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
        if dataset_name is not None and os.path.isfile(f"./2hop_aggre_standard/{dataset_name}.pickle"):
            print(f"Load precomputed graph emb from ./2hop_aggre_standard/{dataset_name}.pickle")
            with open(f"./2hop_aggre_standard/{dataset_name}.pickle", "rb") as fp:
                precomputed = pickle.load(fp)
                self.weight = precomputed["weight"].to(features.device)
                self.features_weighted = precomputed["features_weighted"].to(features.device)
                self.eg = precomputed["eg"].to(features.device)

            self.empty_2hop_mask = precomputed.get("empty_2hop_mask", None)
            if self.empty_2hop_mask is None:
                self.empty_2hop_mask = torch.zeros(
                    self.eg.shape[0],
                    dtype=torch.bool,
                    device=features.device,
                )
            else:
                self.empty_2hop_mask = self.empty_2hop_mask.to(features.device).bool()
        else:
            print("Preprocessing: Aggregrate neighbour embeddings")
            load_or_precompute_two_hop(g, dataset_name)

            with open(f"2_hop_map/{dataset_name}.pkl", "rb") as f:
                self.two_hop_map = pickle.load(f)
            g_2hop = build_two_hop_graph(g.num_nodes(), self.two_hop_map)
            g_2hop = g_2hop.to(features.device)
            self.empty_2hop_mask = g_2hop.in_degrees().to(features.device).eq(0)
            self.weight = get_diag(g_2hop, self.k)
            agg_feat = aggregation(g_2hop, features, k=1) 
            self.features_weighted = (features.swapaxes(1, 0) * self.weight).swapaxes(1, 0).detach()
            self.eg = (agg_feat - self.features_weighted).detach()

            if dataset_name is not None:
                print(f"Save graph emb to ./2hop_aggre_standard/{dataset_name}.pickle")
                if not os.path.isdir("./2hop_aggre_standard"):
                    os.makedirs("./2hop_aggre_standard")
                with open(f"./2hop_aggre_standard/{dataset_name}.pickle", "wb") as fp:
                    pickle.dump({
                        "weight": self.weight.to("cpu"),
                        "features_weighted": self.features_weighted.to("cpu"),
                        "eg": self.eg.to("cpu"),
                        "empty_2hop_mask": self.empty_2hop_mask.to("cpu")
                    }, fp)

        # ==========================================
        # ★ 여기만 추가
        # Empty 2-hop node → valid 2-hop EG center
        # ==========================================
        if self.empty_2hop_mask.any():

            valid_idx = (
                (~self.empty_2hop_mask)
                .nonzero(as_tuple=False)
                .flatten()
            )

            empty_idx = (
                self.empty_2hop_mask
                .nonzero(as_tuple=False)
                .flatten()
            )

            if valid_idx.numel() > 0:

                eg_center = self.eg[
                    valid_idx
                ].mean(
                    dim=0,
                    keepdim=True,
                )

                self.eg = self.eg.clone()

                self.eg[
                    empty_idx
                ] = eg_center.expand(empty_idx.numel(), -1).contiguous()
        # ---------------------------------------
        # Fill nodes with no strict 2-hop neighbor
        # ONCE when Dataloader is initialized.
        # ---------------------------------------
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

        self.coef1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
        )

        self.coef2 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
        )

        self.mlp_g = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim) if bn else nn.Identity()
        )
        self.mlp_n = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim) if bn else nn.Identity()
        )

    def reg_edge(self, emb, eps=1e-8):
        h = emb / (emb.norm(dim=1, keepdim=True) + eps)  
        N = h.size(0)
        if N <= 1:
            return h.new_tensor(0.)

        sv = h.sum(dim=0)                                
        per_node = (h @ sv - 1.0) / (N - 1)             
        loss_reg = ((per_node + 1) / 2).mean()
        return loss_reg

    def forward(self, features, summary, xl2x):

        h1 = self.mlp_g(features)     
        h2 = self.mlp_n(summary)       
        h1 = h1*self.coef1(xl2x)
        h2 = h2*self.coef2(xl2x)

        s = torch.nn.functional.cosine_similarity(h1, h2)  
        uni_loss = self.reg_edge(h2)
        return s.unsqueeze(0), h1, h2, uni_loss

class MCGAD(nn.Module):
    def __init__(
        self,
        g,
        feature,
        n_in,
        n_hidden,
        bn,
        edge_index,
        gamma,
        dataset_name=None,
        xl2x=None,
    ):
        super().__init__()
        self.g = g
        self.feature = feature.to(self.g.device)
        self.bn = bn 
        self.discriminator = Discriminator(n_in, n_hidden, bn)
        self.node_num = self.feature.shape[0]
        self.gamma = gamma

        self.register_buffer("xl2x", xl2x.to(self.g.device))

    def forward(self, target_features, neighbour_features):
        score, h1, h2, uni_loss = self.discriminator(
            target_features.detach(),
            neighbour_features.detach(),
            self.xl2x
        )

        if not self.training:
            h1_unit = F.normalize(h1, p=2, dim=1)
            pooled_center = h1_unit.mean(
                dim=0,
                keepdim=True,
            )

            local_score = 1 - score[0]
            global_score = 1 - F.cosine_similarity(
                h1,
                pooled_center,
                dim=1,
            )

            eps = 1e-9

            local_score = (
                local_score - local_score.min()
            ) / (
                local_score.max() - local_score.min()
            ).clamp_min(eps)

            global_score = (
                global_score - global_score.min()
            ) / (
                global_score.max() - global_score.min()
            ).clamp_min(eps)

            anomaly_score = (
                local_score + global_score
            )
            return -anomaly_score.unsqueeze(0)

        h1_unit = F.normalize(h1, p=2, dim=1)
        center = h1_unit.mean(dim=0, keepdim=True)
        reliability = torch.sigmoid(2.0 * score[0])
        gad_loss = (
            reliability
            * ((1-self.gamma)*(1 - score[0]) + self.gamma *(1 - F.cosine_similarity(h1, center.detach(), dim=1)))
        ).mean()
        return score, uni_loss, gad_loss

def two_hop_map_to_edge_index(two_hop_map):
    src = []
    dst = []

    for v, neighbors in two_hop_map.items():
        for u in neighbors:
            src.append(int(u))
            dst.append(int(v))

    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor([src, dst], dtype=torch.long)

def compute_xtlx_vec(features, edge_index):
    lap_cpu = get_lap(edge_index, features.shape[0]).coalesce()
    feat_cpu = features.detach().cpu()

    xt_l = torch.sparse.mm(lap_cpu, feat_cpu)
    xtlx_diag = torch.sum(feat_cpu * xt_l, dim=0)
    xtx_diag = torch.sum(feat_cpu * feat_cpu, dim=0).clamp_min(1e-12)
    xtlx_xtx = xtlx_diag / xtx_diag
    return torch.sigmoid(xtlx_xtx)


def load_or_compute_xtlx(features, edge_index, cache_path, device):
    if os.path.exists(cache_path):
        print(f"Load cached XTLX from {cache_path}")
        return torch.load(cache_path, map_location=device)

    print(f"Compute XTLX and save to {cache_path}")
    vec = compute_xtlx_vec(features, edge_index)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(vec.cpu(), cache_path)

    return vec.to(device)
