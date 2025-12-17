import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.utils import add_remaining_self_loops, to_undirected, coalesce, softmax
from torch_geometric.data import Data
from typing import Optional, Union, List

# ============== GGAT Layer ==============
class GGAT(nn.Module):
    def __init__(self, in_features, out_features, heads=1, dropout=0.0, bias=False,
                 init_type='uniform'):
        super().__init__()
        self.in_features = in_features
        self.out_features_total = out_features * heads
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = out_features
        self.scale = 1 / np.sqrt(self.d_k)
        self.init_type = init_type

        self.W_q = nn.Linear(in_features, self.out_features_total, bias=bias)
        self.W_k = nn.Linear(in_features, self.out_features_total, bias=bias)
        self.W_v = nn.Linear(in_features, self.out_features_total, bias=bias)

        self.D = nn.Parameter(torch.zeros(heads, self.d_k))
        # self.D = torch.zeros(heads, self.d_k).to('cuda')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        if self.W_q.bias is not None:
            nn.init.zeros_(self.W_q.bias)
        if self.W_k.bias is not None:
            nn.init.zeros_(self.W_k.bias)
        if self.W_v.bias is not None:
            nn.init.zeros_(self.W_v.bias)

        if self.init_type == 'bernoulli':
            for h in range(self.heads):
                p = h / (self.heads - 1) if self.heads > 1 else 0.5
                mask = torch.bernoulli(torch.full((self.d_k,), p))
                self.D.data[h] = 2 * mask - 1
        elif self.init_type == 'uniform':
            nn.init.uniform_(self.D, -1, 1)
        elif self.init_type == 'normal':
            nn.init.normal_(self.D, mean=0, std=0.5)
        elif self.init_type == 'ones':
            nn.init.ones_(self.D)
        elif self.init_type == 'xavier':
            nn.init.xavier_uniform_(self.D)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)

        Q = self.W_q(x).view(num_nodes, self.heads, self.d_k)
        K = self.W_k(x).view(num_nodes, self.heads, self.d_k)
        V = self.W_v(x).view(num_nodes, self.heads, self.d_k)

        row, col = edge_index
        self_mask = (row == col)
        edge_mask = ~self_mask

        scores = torch.zeros(edge_index.size(1), self.heads, device=x.device)

        if edge_mask.any():
            K_i = K[row[edge_mask]]
            Q_j = Q[col[edge_mask]]
            diff = K_i - Q_j
            scores[edge_mask] = torch.sum(diff * self.D * diff, dim=-1)
        if self_mask.any():
            K_self = K[row[self_mask]]
            Q_self = Q[row[self_mask]]
            scores[self_mask] = torch.sum(K_self * self.D * Q_self, dim=-1)

        scores = F.leaky_relu(scores * self.scale, 0.2)
        alpha = softmax(scores, row, num_nodes=num_nodes)
        alpha = self.dropout(alpha)

        out = torch.zeros_like(V)
        out.index_add_(0, row, alpha.unsqueeze(-1) * V[col])
        return out.view(num_nodes, self.out_features_total)

class GGAT_FullMetric(nn.Module):
    def __init__(self, in_features, out_features, heads=1, dropout=0.0, bias=False,
                 init_type='uniform'):
        super().__init__()
        self.in_features = in_features
        self.out_features_total = out_features * heads
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = out_features
        self.scale = 1 / np.sqrt(self.d_k)

        self.W_q = nn.Linear(in_features, self.out_features_total, bias=bias)
        self.W_k = nn.Linear(in_features, self.out_features_total, bias=bias)
        self.W_v = nn.Linear(in_features, self.out_features_total, bias=bias)

        # Full symmetric matrix per head (핵심 변경!)
        # S_raw: (heads, d_k, d_k)
        self.S_raw = nn.Parameter(torch.zeros(heads, self.d_k, self.d_k))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        
        # S를 identity로 초기화 (또는 다른 방법)
        for h in range(self.heads):
            nn.init.eye_(self.S_raw.data[h])
            # 약간의 noise 추가
            self.S_raw.data[h] += 0.1 * torch.randn(self.d_k, self.d_k)

    @property
    def S(self):
        # Symmetric하게 만들기 (positive definite 강제 X)
        return (self.S_raw + self.S_raw.transpose(-1, -2)) / 2

    def forward(self, x, edge_index):
        num_nodes = x.size(0)

        Q = self.W_q(x).view(num_nodes, self.heads, self.d_k)
        K = self.W_k(x).view(num_nodes, self.heads, self.d_k)
        V = self.W_v(x).view(num_nodes, self.heads, self.d_k)

        row, col = edge_index
        
        # Difference
        diff = K[row] - Q[col]  # (E, heads, d_k)
        
        # Full quadratic form: diff^T @ S @ diff for each head
        # diff: (E, heads, d_k)
        # S: (heads, d_k, d_k)
        S = self.S  # (heads, d_k, d_k)
        
        # Einstein summation: (E, heads, d_k) @ (heads, d_k, d_k) @ (E, heads, d_k)
        # Step 1: diff @ S → (E, heads, d_k)
        Sd = torch.einsum('ehd,hdk->ehk', diff, S)
        # Step 2: element-wise multiply and sum → (E, heads)
        scores = torch.einsum('ehk,ehk->eh', Sd, diff)
        
        scores = F.leaky_relu(scores * self.scale, 0.2)
        alpha = softmax(scores, row, num_nodes=num_nodes)
        alpha = self.dropout(alpha)

        out = torch.zeros_like(V)
        out.index_add_(0, row, alpha.unsqueeze(-1) * V[col])
        return out.view(num_nodes, self.out_features_total)

    def get_signature(self):
        """학습된 metric의 signature 분석"""
        signatures = []
        S = self.S.detach().cpu().numpy()
        for h in range(self.heads):
            eigs = np.linalg.eigvalsh(S[h])
            n_pos = int(np.sum(eigs > 1e-6))
            n_neg = int(np.sum(eigs < -1e-6))
            n_zero = len(eigs) - n_pos - n_neg
            signatures.append((n_pos, n_zero, n_neg))
        return signatures

class GGAT_SignatureDecomp(nn.Module):
    """
    S = S⁺ - S⁻ where S⁺, S⁻ are positive semi-definite
    
    이론적 의미:
    - S⁺: "같을수록 가까움" (homophilic) 기여
    - S⁻: "다를수록 가까움" (heterophilic) 기여
    - Heterophilic graph에서는 S⁻가 dominant해져야 함
    """
    def __init__(self, in_features, out_features, heads=1, dropout=0.0, bias=False,
                 rank=None):
        super().__init__()
        self.heads = heads
        self.d_k = out_features
        self.out_features_total = out_features * heads
        self.scale = 1 / np.sqrt(self.d_k)
        self.dropout = nn.Dropout(dropout)
        
        # Rank for low-rank approximation (None = full rank)
        self.rank = rank if rank is not None else self.d_k

        self.W_q = nn.Linear(in_features, self.out_features_total, bias=bias)
        self.W_k = nn.Linear(in_features, self.out_features_total, bias=bias)
        self.W_v = nn.Linear(in_features, self.out_features_total, bias=bias)

        # S⁺ = L_pos @ L_pos^T (guaranteed PSD)
        # S⁻ = L_neg @ L_neg^T (guaranteed PSD)
        self.L_pos = nn.Parameter(torch.zeros(heads, self.d_k, self.rank))
        self.L_neg = nn.Parameter(torch.zeros(heads, self.d_k, self.rank))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        
        # L 초기화: S가 처음에 identity에 가깝도록
        # S = L_pos @ L_pos^T - L_neg @ L_neg^T ≈ I
        # L_pos ≈ I^{1/2}, L_neg ≈ 0
        for h in range(self.heads):
            nn.init.eye_(self.L_pos.data[h, :min(self.d_k, self.rank), :min(self.d_k, self.rank)])
            nn.init.zeros_(self.L_neg.data[h])
            # 약간의 noise
            self.L_pos.data[h] += 0.01 * torch.randn_like(self.L_pos.data[h])
            self.L_neg.data[h] += 0.01 * torch.randn_like(self.L_neg.data[h])

    @property
    def S_pos(self):
        # (heads, d_k, rank) @ (heads, rank, d_k) → (heads, d_k, d_k)
        return torch.bmm(self.L_pos, self.L_pos.transpose(-1, -2))

    @property
    def S_neg(self):
        return torch.bmm(self.L_neg, self.L_neg.transpose(-1, -2))

    @property
    def S(self):
        return self.S_pos - self.S_neg

    def forward(self, x, edge_index):
        num_nodes = x.size(0)

        Q = self.W_q(x).view(num_nodes, self.heads, self.d_k)
        K = self.W_k(x).view(num_nodes, self.heads, self.d_k)
        V = self.W_v(x).view(num_nodes, self.heads, self.d_k)

        row, col = edge_index
        diff = K[row] - Q[col]  # (E, heads, d_k)
        
        # Efficient computation using L directly
        # diff^T S⁺ diff = ||L_pos^T diff||²
        # diff^T S⁻ diff = ||L_neg^T diff||²
        
        # (E, heads, d_k) @ (heads, d_k, rank) → (E, heads, rank)
        proj_pos = torch.einsum('ehd,hdr->ehr', diff, self.L_pos)
        proj_neg = torch.einsum('ehd,hdr->ehr', diff, self.L_neg)
        
        # ||proj||² → (E, heads)
        quad_pos = torch.sum(proj_pos ** 2, dim=-1)
        quad_neg = torch.sum(proj_neg ** 2, dim=-1)
        
        # Final score: diff^T (S⁺ - S⁻) diff
        scores = quad_pos - quad_neg
        
        scores = F.leaky_relu(scores * self.scale, 0.2)
        alpha = softmax(scores, row, num_nodes=num_nodes)
        alpha = self.dropout(alpha)

        out = torch.zeros_like(V)
        out.index_add_(0, row, alpha.unsqueeze(-1) * V[col])
        return out.view(num_nodes, self.out_features_total)

    def get_signature(self):
        """학습된 metric의 signature 분석"""
        signatures = []
        S = self.S.detach().cpu().numpy()
        for h in range(self.heads):
            eigs = np.linalg.eigvalsh(S[h])
            n_pos = int(np.sum(eigs > 1e-6))
            n_neg = int(np.sum(eigs < -1e-6))
            n_zero = len(eigs) - n_pos - n_neg
            signatures.append((n_pos, n_zero, n_neg))
        return signatures

    def get_pos_neg_ratio(self):
        """S⁺와 S⁻의 상대적 크기 분석"""
        S_pos_norm = torch.norm(self.S_pos, dim=(-2, -1))  # (heads,)
        S_neg_norm = torch.norm(self.S_neg, dim=(-2, -1))  # (heads,)
        return {
            'S_pos_norm': S_pos_norm.detach().cpu().numpy(),
            'S_neg_norm': S_neg_norm.detach().cpu().numpy(),
            'ratio': (S_neg_norm / (S_pos_norm + 1e-8)).detach().cpu().numpy()
        }

# ============== Scaled Dot Product Attention Layer ==============
class ScaledDotProductGAT(nn.Module):
    def __init__(self, in_features, out_features, heads=1, dropout=0.0, bias=False):
        super().__init__()
        self.heads = heads
        self.d_k = out_features
        self.out_features_total = out_features * heads
        self.scale = 1 / np.sqrt(self.d_k)
        self.W_q = nn.Linear(in_features, self.out_features_total, bias=bias)
        self.W_k = nn.Linear(in_features, self.out_features_total, bias=bias)
        self.W_v = nn.Linear(in_features, self.out_features_total, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        if self.W_q.bias is not None:
            nn.init.zeros_(self.W_q.bias)
        if self.W_k.bias is not None:
            nn.init.zeros_(self.W_k.bias)
        if self.W_v.bias is not None:
            nn.init.zeros_(self.W_v.bias)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)

        Q = self.W_q(x).view(num_nodes, self.heads, self.d_k)
        K = self.W_k(x).view(num_nodes, self.heads, self.d_k)
        V = self.W_v(x).view(num_nodes, self.heads, self.d_k)

        row, col = edge_index
        scores = (Q[row] * K[col]).sum(dim=-1) * self.scale
        alpha = softmax(scores, row, num_nodes=num_nodes)
        alpha = self.dropout(alpha)

        out = torch.zeros_like(V)
        out.index_add_(0, row, alpha.unsqueeze(-1) * V[col])
        return out.view(num_nodes, self.out_features_total)


# ============== Plain GNN ==============
class PlainGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, conv_type='GGAT', heads=4,
                 drop_in=0.5, drop=0.5, init_type='uniform'):
        super().__init__()
        self.enc = nn.Linear(nfeat, nhid)
        self.dec = nn.Linear(nhid, nclass)
        self.drop_in = drop_in
        self.drop = drop
        self.nlayers = nlayers

        self.convs = nn.ModuleList([self._make_conv(conv_type, nhid, nhid // heads, heads, init_type) for i in range(nlayers)])

    def _make_conv(self, conv_type, in_dim, out_dim, heads, init_type):
        if conv_type == 'GGAT':
            return GGAT(in_dim, out_dim, heads=heads, init_type=init_type)
        elif conv_type == 'ScaledDot':
            return ScaledDotProductGAT(in_dim, out_dim, heads=heads)
        elif conv_type == 'GATConv':
            return GATConv(in_dim, out_dim, heads=heads, concat=True)
        elif conv_type == 'GATv2Conv':
            return GATv2Conv(in_dim, out_dim, heads=heads, concat=True)
        elif conv_type == 'GGAT_FullMetric':
            return GGAT_FullMetric(in_dim, out_dim, heads=heads, init_type=init_type)
        elif conv_type == 'GGAT_SignatureDecomp':
            return GGAT_SignatureDecomp(in_dim, out_dim, heads=heads)

    def forward(self, data):
        X, edge_index = data.x, data.edge_index
        X = F.dropout(X, self.drop_in, training=self.training)
        X = F.relu(self.enc(X))

        for conv_layer in self.convs:
            X = F.relu(conv_layer(X, edge_index))
            if X.dim() == 3:
                X = X.mean(dim=1)

        X = F.dropout(X, self.drop, training=self.training)
        return self.dec(X)
