"""
causalnerve.core.engine
=======================
The foundational Sparse Causal Graph Engine (CSC) for CausalNerve.
Provides industrial-grade, O(N*K) complexity causal sparse cognition.
Never materializes dense N×N matrices during forward propagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List, Union
import time

class SparseGraph:
    """
    Industrial Sparse Graph — Never materializes N×N dense matrix.
    Uses indexed propagation for O(N*K) complexity.
    """
    __slots__ = ['values', 'col_indices', 'valid_mask', 'N', 'K', 'entropy',
                 'edge_gates', 'metadata', 'device']

    def __init__(self, values, col_indices, valid_mask, N, K,
                 entropy=None, edge_gates=None, metadata=None):
        self.values = values
        self.col_indices = col_indices
        self.valid_mask = valid_mask
        self.N = N
        self.K = K
        self.entropy = entropy
        self.edge_gates = edge_gates
        self.metadata = metadata or {}
        self.device = values.device

    def __getitem__(self, idx):
        """Slice the batch/time dimension without overhead."""
        return SparseGraph(
            self.values[idx],
            self.col_indices[idx],
            self.valid_mask[idx],
            self.N, self.K,
            self.entropy[idx] if self.entropy is not None else None,
            self.edge_gates[idx] if self.edge_gates is not None else None,
            self.metadata
        )

    def to_sparse_coo(self, batch_idx=0):
        mask = self.valid_mask[batch_idx]
        vals = self.values[batch_idx][mask]
        cols = self.col_indices[batch_idx][mask]
        
        rows = torch.arange(self.N, device=self.device).unsqueeze(1).expand(-1, self.K)[mask]
        indices = torch.stack([rows, cols])
        return torch.sparse_coo_tensor(indices, vals, (self.N, self.N))

    def to_dense(self):
        """Vectorized dense reconstruction (ONLY for loss/viz)."""
        B = self.values.shape[0]
        row_idx = torch.arange(self.N, device=self.device).view(1, self.N, 1).expand(B, self.N, self.K)
        
        flat_idx = (torch.arange(B, device=self.device).view(B, 1, 1) * self.N * self.N +
                    row_idx * self.N +
                    self.col_indices)
        
        dense = torch.zeros(B * self.N * self.N, device=self.device, dtype=self.values.dtype)
        m = self.valid_mask.reshape(-1)
        v = (self.values * self.valid_mask.to(self.values.dtype)).reshape(-1)
        f = flat_idx.reshape(-1)
        
        dense.scatter_add_(0, f[m], v[m])
        return dense.view(B, self.N, self.N)

    def apply_intervention(self, intervention_mask, B):
        """Zero out edges involving intervened nodes."""
        inv_exp = intervention_mask.unsqueeze(-1).expand(B, self.N, self.K)
        new_mask = self.valid_mask & (~inv_exp)
        return SparseGraph(self.values, self.col_indices, new_mask,
                           self.N, self.K, self.entropy, self.edge_gates)

    def edge_count(self):
        return self.valid_mask.sum()

    def mean_weight(self):
        return (self.values * self.valid_mask.float()).sum() / (self.valid_mask.sum() + 1e-8)


class SparseIndexedRouter(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mix = nn.Linear(d_model * 2, d_model)

    def forward(self, x, graph: SparseGraph):
        B, N, D = x.shape
        K = graph.K
        
        flat_indices = graph.col_indices.reshape(B, -1)
        parent_feats = torch.gather(x, 1, flat_indices.unsqueeze(-1).expand(-1, -1, D))
        parent_feats = parent_feats.view(B, N, K, D)
        
        weights = (graph.values * graph.valid_mask.to(graph.values.dtype)).unsqueeze(-1)
        routed = torch.sum(parent_feats * weights, dim=2)
        
        return self.mix(torch.cat([x, routed], dim=-1))


class CausalEdgeGenerator(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.edge_scorer = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, source_feat, target_feat, regime_feat):
        combined = torch.cat([source_feat, target_feat, regime_feat], dim=-1)
        logits = self.edge_scorer(combined).squeeze(-1)
        
        pair = torch.cat([source_feat, target_feat], dim=-1)
        gate = self.edge_gate(pair).mean(dim=-1)
        return logits, gate


class TopKSparseSelector(nn.Module):
    def __init__(self, K: int):
        super().__init__()
        self.K = K

    def forward(self, logits, cand_valid, edge_gates=None, training=True):
        B, N, C = logits.shape
        K = min(self.K, C)

        mask_exp = cand_valid.unsqueeze(0).expand(B, -1, -1)
        masked_logits = logits.masked_fill(~mask_exp, -1e4)

        if training:
            noise = torch.randn_like(masked_logits).abs().clamp(1e-6, 1.0)
            noisy = masked_logits - torch.log(-torch.log(noise + 1e-10) + 1e-10)
        else:
            noisy = masked_logits

        topk_vals, topk_idx = torch.topk(noisy, K, dim=-1)
        
        orig_topk_logits = torch.gather(masked_logits, 2, topk_idx)
        p = torch.sigmoid(orig_topk_logits)
        
        is_valid = (topk_vals > -5000).float()
        values = p + (is_valid - p).detach()

        sel_gates = None
        if edge_gates is not None:
            sel_gates = torch.gather(edge_gates, 2, topk_idx)
            values = values * sel_gates

        entropy = F.binary_cross_entropy_with_logits(
            orig_topk_logits.clamp(-15, 15), p.detach(), reduction='none') * is_valid

        return values, topk_idx, entropy, sel_gates


class CausalGraphBlock(nn.Module):
    """One layer of the CausalNerve Sparse Cognitive Substrate."""
    def __init__(self, d_model, K, window, max_nodes, alpha=0.9, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.window = window
        self.max_nodes = max_nodes
        
        self.norm = nn.LayerNorm(d_model)
        self.edge_gen = CausalEdgeGenerator(d_model)
        self.selector = TopKSparseSelector(K)
        self.router = SparseIndexedRouter(d_model)
        
        self.W_prop = nn.Linear(d_model, d_model, bias=False)
        self.alpha = alpha
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
        self.register_buffer('_cand_indices', None, persistent=False)
        self.register_buffer('_cand_valid', None, persistent=False)

    def _get_candidates(self, N, device):
        if self._cand_indices is not None and self._cand_indices.shape[0] == N:
            return self._cand_indices, self._cand_valid
        
        W = min(self.window, N)
        max_cands = max(min(N - 1, W), 1)
        indices = torch.zeros(N, max_cands, dtype=torch.long, device=device)
        valid = torch.zeros(N, max_cands, dtype=torch.bool, device=device)
        
        for i in range(N):
            start = max(0, i - W)
            parents = list(range(start, i))
            for k, p in enumerate(parents):
                if k < max_cands:
                    indices[i, k] = p
                    valid[i, k] = True
        
        self.register_buffer('_cand_indices', indices)
        self.register_buffer('_cand_valid', valid)
        return indices, valid

    def forward(self, x, regime_state, frozen_graph=None, intervention_mask=None):
        B, N, D = x.shape
        h = self.norm(x)
        
        if frozen_graph is None:
            cand_idx, cand_val = self._get_candidates(N, x.device)
            max_cands = cand_idx.shape[1]
            
            flat_cands = cand_idx.view(-1).clamp(0, N - 1)
            source_feat = h[:, flat_cands, :].view(B, N, max_cands, D)
            target_feat = h.unsqueeze(2).expand(-1, -1, max_cands, -1)
            reg_feat = regime_state.view(B, 1, 1, D).expand(-1, N, max_cands, -1)
            
            logits, gates = self.edge_gen(source_feat, target_feat, reg_feat)
            vals, idx, ent, g = self.selector(logits, cand_val, gates, self.training)
            
            global_idx = torch.gather(cand_idx.unsqueeze(0).expand(B, -1, -1), 2, idx)
            sel_valid = torch.gather(cand_val.unsqueeze(0).expand(B, -1, -1), 2, idx)
            
            graph = SparseGraph(vals, global_idx, sel_valid, N, self.K, ent, g)
        else:
            graph = frozen_graph
            
        if intervention_mask is not None:
            graph = graph.apply_intervention(intervention_mask, B)
            
        routed = self.router(h, graph)
        h_evolve = torch.tanh(self.W_prop(routed) + self.alpha * h)
        out = h + self.ff(h_evolve)
        
        return out, graph


class CausalGraphEngine(nn.Module):
    """
    Core graph discovery and representation engine for CausalNerve.
    """
    def __init__(self, d_model=64, n_layers=3, K=4, window=32,
                 max_nodes=512, alpha=0.9, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            CausalGraphBlock(d_model, K, window, max_nodes, alpha, dropout)
            for _ in range(n_layers)
        ])
        self.regime_proj = nn.Linear(d_model, d_model)
        self.final_norm = nn.LayerNorm(d_model)

    def discover_graphs(self, x: torch.Tensor) -> Tuple[List[SparseGraph], torch.Tensor]:
        if x.dim() == 4:
            B, T, N, D = x.shape
            x_flat = x.reshape(B * T, N, D)
        else:
            x_flat = x
            B, T, N, D = x.shape[0], 1, x.shape[1], x.shape[2]
            
        regime = self.regime_proj(x_flat.mean(dim=1))
        
        all_layers = []
        curr_h = x_flat
        for block in self.blocks:
            _, graph = block(curr_h, regime)
            all_layers.append(graph)
            curr_h = curr_h + 0.1 * block.ff(torch.tanh(block.W_prop(curr_h)))
            
        return all_layers, regime

    def forward(self, h, intervention_mask=None, frozen_graphs=None):
        B, N, D = h.shape
        regime = self.regime_proj(h.mean(dim=1))
        
        graphs = []
        for i, block in enumerate(self.blocks):
            fg = frozen_graphs[i] if frozen_graphs else None
            h, graph = block(h, regime, frozen_graph=fg, intervention_mask=intervention_mask)
            graphs.append(graph)
            
        return {
            'hidden': self.final_norm(h),
            'graphs': graphs,
            'regime_state': regime
        }

    def get_dense_adjacency(self) -> torch.Tensor:
        with torch.no_grad():
            N = getattr(self, 'n_nodes', 14)
            device = next(self.parameters()).device
            d_model = self.blocks[0].d_model
            h = torch.zeros((1, N, d_model), device=device)
            out = self.forward(h)
            graph = out['graphs'][-1]
            dense = graph.to_dense()
            return dense[0].detach()
