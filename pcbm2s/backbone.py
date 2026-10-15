"""Dual-graph encoders retained from the project implementation.

Source provenance and third-party notices are in docs/model_and_provenance.md
and THIRD_PARTY_NOTICES.md. Public training uses PCBM2S in model.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d, ModuleList
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool
from torch_geometric.utils import to_dense_batch, degree
from typing import Optional

try:
    from mamba_ssm import Mamba
except ImportError:
    pass  # The selected encoder raises a clear error if Mamba is unavailable.
    Mamba = None

from .compound_tools import CompoundKit, get_atom_feature_dims, get_bond_feature_dims


# ================== Utility Functions =====================
def _lexsort(keys, descending=False):
    """Stable lexsort on GPU tensors, matching Graph-Mamba's lexsort.

    Sort by the last key first (primary), then second-to-last, etc.
    Reference: Graph-Mamba/graphgps/layer/gps_layer.py :: lexsort
    """
    assert len(keys) >= 1
    out = keys[0].argsort(descending=descending, stable=True)
    for k in keys[1:]:
        index = k[out]
        index = index.argsort(descending=descending, stable=True)
        out = out[index]
    return out

# ================== Feature Dimensions =====================
atom_id_names = list(CompoundKit.atom_vocab_dict.keys())
bond_id_names = list(CompoundKit.bond_vocab_dict.keys())
full_atom_feature_dims = get_atom_feature_dims(atom_id_names)
full_bond_feature_dims = get_bond_feature_dims(bond_id_names)

NUM_INT_FEATURES = len(full_atom_feature_dims)  # 18
NUM_FLOAT_FEATURES = 3  # vdw_radius, partial_charge, mass
ATOM_FLOAT_NAMES = ["van_der_waals_radis", "partial_charge", "mass"]


# =========================
# RBF & Float Atom Encoder
# =========================
class RBF(torch.nn.Module):
    """Radial Basis Function — 不可训练, 使用生物化学经验参数."""

    def __init__(self, centers, gamma, dtype='float32'):
        super(RBF, self).__init__()
        self.register_buffer('centers', centers.reshape([1, -1]))
        self.register_buffer('gamma', gamma.clone().detach())

    def forward(self, x):
        x = x.reshape([-1, 1])
        return torch.exp(-self.gamma * torch.square(x - self.centers))


class AtomFloatRBF(nn.Module):
    """RBF encoding for 3 float atom features: vdw_radius, partial_charge, mass."""
    def __init__(self, embed_dim, rbf_params=None):
        super().__init__()
        self.atom_float_names = ATOM_FLOAT_NAMES

        if rbf_params is None:
            self.rbf_params = {
                'van_der_waals_radis': (torch.arange(1, 3, 0.2), torch.tensor([10.0])),
                'partial_charge': (torch.arange(-1, 4, 0.25), torch.tensor([10.0])),
                'mass': (torch.arange(0, 2, 0.1), torch.tensor([10.0])),
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = torch.nn.ModuleList()
        self.rbf_list = torch.nn.ModuleList()
        for name in self.atom_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = torch.nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

    def forward(self, atom_float_features):
        out_embed = 0
        for i, name in enumerate(self.atom_float_names):
            x = atom_float_features[:, i].reshape(-1, 1)
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed


# =========================
# Shared Embedding Components
# =========================
class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim + 5, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])
        return x_embedding


class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        self.bond_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim + 5, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])
        return bond_embedding


# =========================
# Protein Graph Components
# =========================
class ProteinNodeEncoder(nn.Module):
    def __init__(self, input_dim=31, output_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, prot_node_x):
        return self.encoder(prot_node_x)


class ProteinEdgeEncoder(nn.Module):
    """蛋白边编码器: 距离用 RBF 连续编码, 方向+序列距离用 Linear."""
    def __init__(self, input_dim=5, output_dim=128,
                 use_dist_rbf=False, dist_rbf_centers=20, dist_rbf_gamma=20.0):
        super().__init__()
        self.use_dist_rbf = use_dist_rbf
        if use_dist_rbf:
            self.dist_rbf = RBF(
                torch.linspace(0, 1.0, dist_rbf_centers),
                torch.tensor([dist_rbf_gamma]),
            )
            self.dist_proj = nn.Sequential(
                nn.Linear(dist_rbf_centers, output_dim // 2),
                nn.BatchNorm1d(output_dim // 2),
                nn.ReLU(),
            )
            self.other_encoder = nn.Sequential(
                nn.Linear(input_dim - 1, output_dim // 2),
                nn.BatchNorm1d(output_dim // 2),
                nn.ReLU(),
            )
            self.out_dim = output_dim
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
            )

    def forward(self, prot_edge_attr):
        if self.use_dist_rbf:
            dist = prot_edge_attr[:, 0:1]        # [E, 1] — 距离
            other = prot_edge_attr[:, 1:]          # [E, 4] — dx, dy, dz, seq_dist
            dist_rbf = self.dist_rbf(dist)         # [E, n_centers]
            dist_emb = self.dist_proj(dist_rbf)    # [E, out_dim//2]
            other_emb = self.other_encoder(other)  # [E, out_dim//2]
            return torch.cat([dist_emb, other_emb], dim=-1)
        else:
            return self.encoder(prot_edge_attr)


# =========================
# Adapters
# =========================
class ProteinAdapter(nn.Module):
    def __init__(self, input_dim=1024, output_dim=128, dropout=0.1):
        super().__init__()
        self.prot_adapter = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, target):
        return self.prot_adapter(target)


class Molt5Adapter(nn.Module):
    def __init__(self, input_dim=768, output_dim=128, dropout=0.1):
        super().__init__()
        self.molt5_adapter = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, molt5_emb):
        return self.molt5_adapter(molt5_emb)


# =========================
# RWSE Encoder
# =========================
class RWSENodeEncoder(nn.Module):
    def __init__(self, num_steps, rwse_dim, encoder_type='linear', norm_type='batchnorm'):
        super().__init__()
        if norm_type == 'batchnorm':
            self.norm = nn.BatchNorm1d(num_steps)
        else:
            self.norm = None

        if encoder_type == 'linear':
            self.encoder = nn.Linear(num_steps, rwse_dim)
        elif encoder_type == 'mlp':
            self.encoder = nn.Sequential(
                nn.Linear(num_steps, 2 * rwse_dim),
                nn.ReLU(),
                nn.Linear(2 * rwse_dim, rwse_dim),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(self, rwse_raw):
        if self.norm:
            rwse_raw = self.norm(rwse_raw)
        return self.encoder(rwse_raw)


# =========================
# GINEBlock — 即 GNNPlus 的一层
# =========================
class GINEBlock(nn.Module):
    """GINE message-passing block with residual + FFN.

    Faithfully follows GNNPlus (ICML 2025) GINEConvLayer:
      x = x_in + dropout(relu(conv(x)))   # residual
      x = norm1(x)                         # norm before FFN
      x = x + ffn_dropout(ffn(x))          # FFN with dropout
      x = norm2(x)                         # norm after FFN

    Reference: GNNPlus/GNNPlus/layer/gine_conv_layer.py :: GINEConvLayer
    """
    def __init__(self, dim, dropout=0.1, residual=True, ffn=True, edge_dim=None):
        super().__init__()
        self.residual = residual
        self.ffn = ffn
        self.dropout = dropout

        gin_nn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        if edge_dim is not None:
            self.conv = GINEConv(gin_nn, edge_dim=edge_dim)
        else:
            self.conv = GINEConv(gin_nn)

        if self.ffn:
            # 原始 GNNPlus: norm1 在 FFN 前, norm2 在 FFN 后
            self.norm1 = nn.BatchNorm1d(dim)
            self.ff_linear1 = nn.Linear(dim, dim * 2)
            self.ff_linear2 = nn.Linear(dim * 2, dim)
            self.norm2 = nn.BatchNorm1d(dim)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        # AMP float16 下 GINEConv 在高 degree 节点上聚合溢出，禁用 autocast。
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            edge_attr = edge_attr.float() if edge_attr is not None else None

            h = self.conv(x, edge_index, edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            if self.residual and x.shape[-1] == h.shape[-1]:
                h = x + h

            if self.ffn:
                h = self.norm1(h)
                h = h + self.ff_dropout2(self.ff_linear2(
                    self.ff_dropout1(F.relu(self.ff_linear1(h)))))
                h = self.norm2(h)

            return h


# ==================================================================
# GPSLayer — 按 rampasek/GraphGPS 官方实现
#
# 每一层: GINE (local) + Transformer (global), 两者并行处理同一个 x,
# 然后求和, 再过 MLP.
# ==================================================================
class GPSLayer(nn.Module):
    """Single GraphGPS layer: GINE (local) + Transformer (global).

    Faithfully follows rampasek/GraphGPS post-norm architecture:
      h_local = norm(h_in + dropout(conv(h)))     # local + residual + post-norm
      h_attn  = norm(h_in + dropout(attn(h)))     # global + residual + post-norm
      h = h_local + h_attn                        # combine (h_in added twice)
      h = norm(h + ffn(h))                        # FFN + post-norm

    Reference: GraphGPS/graphgps/layer/gps_layer.py :: GPSLayer
    """
    def __init__(self, dim, heads=4, dropout=0.1, attn_dropout=0.1, edge_dim=None):
        super().__init__()

        # Local: GINE
        gin_nn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        if edge_dim is not None:
            self.conv = GINEConv(gin_nn, edge_dim=edge_dim)
        else:
            self.conv = GINEConv(gin_nn)

        # Global: MultiheadAttention (Transformer)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=attn_dropout, batch_first=True,
        )

        # FFN
        self.ff_linear1 = nn.Linear(dim, dim * 2)
        self.ff_linear2 = nn.Linear(dim * 2, dim)

        # Post-norm: 与原始 GraphGPS 一致
        self.norm1_local = nn.BatchNorm1d(dim)
        self.norm1_attn = nn.BatchNorm1d(dim)
        self.norm2 = nn.BatchNorm1d(dim)

        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        # AMP float16 下 GINEConv 消息聚合和 MultiheadAttention 均可能溢出。
        # 原始 GraphGPS 不使用 AMP，此处禁用 autocast 强制 float32。
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            edge_attr = edge_attr.float() if edge_attr is not None else None
            h_in = x

            h_out_list = []

            # 1. Local: GINE → dropout → residual → post-norm
            h_local = self.conv(x, edge_index, edge_attr)
            h_local = self.dropout_local(h_local)
            h_local = h_in + h_local  # 残差
            h_local = self.norm1_local(h_local)  # post-norm
            h_out_list.append(h_local)

            # 2. Global: Transformer → dropout → residual → post-norm
            h_dense, mask = to_dense_batch(x, batch)
            h_attn, _ = self.attn(
                h_dense, h_dense, h_dense,
                key_padding_mask=~mask, need_weights=False,
            )
            h_attn = h_attn[mask]
            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in + h_attn  # 残差（h_in 第二次被加）
            h_attn = self.norm1_attn(h_attn)  # post-norm
            h_out_list.append(h_attn)

            # 3. Combine local + global
            h = sum(h_out_list)

            # 4. FFN + post-norm
            h = h + self.ff_dropout2(self.ff_linear2(
                self.ff_dropout1(F.relu(self.ff_linear1(h)))))
            h = self.norm2(h)

            return h


# ==================================================================
# GraphMambaLayer — 按 bowang-lab/Graph-Mamba 实现
#
# 核心创新 (v1 丢失的部分):
#   1. Degree-based node permutation — 按节点度数降序排列
#   2. Noise augmentation — 训练时给 degree 加噪声提高鲁棒性
#   3. Mamba SSM 按排列后顺序扫描
# ==================================================================
class GraphMambaLayer(nn.Module):
    """Single GraphMamba layer: GINE (local) + Mamba with degree perm (global).

    Faithfully follows bowang-lab/Graph-Mamba 'Mamba_Hybrid_Degree_Noise' mode:
      - Training: degree + uniform noise → lexsort → Mamba → un-permute
      - Eval: average 5 random permutations for robustness
      - Post-norm architecture matching GraphGPS

    Reference: Graph-Mamba/graphgps/layer/gps_layer.py :: GPSLayer (Mamba_Hybrid_Degree_Noise)
    """
    def __init__(
        self, dim, dropout=0.1, d_state=16, d_conv=4,
        eval_samples=5, edge_dim=None,
    ):
        super().__init__()

        # Local: GINE
        gin_nn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        if edge_dim is not None:
            self.conv = GINEConv(gin_nn, edge_dim=edge_dim)
        else:
            self.conv = GINEConv(gin_nn)

        # Global: Mamba SSM
        if Mamba is None:
            raise ImportError("GraphMamba requires mamba-ssm and a compatible Linux/CUDA PyTorch build. See README.md. No alternative encoder is substituted automatically.")
        self.mamba = Mamba(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=1,
        )

        # FFN
        self.ff_linear1 = nn.Linear(dim, dim * 2)
        self.ff_linear2 = nn.Linear(dim * 2, dim)

        # Post-norm: 与原始 Graph-Mamba / GraphGPS 一致
        self.norm1_local = nn.BatchNorm1d(dim)
        self.norm1_attn = nn.BatchNorm1d(dim)
        self.norm2 = nn.BatchNorm1d(dim)

        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

        self.eval_samples = eval_samples  # 原始代码 eval 时平均 5 次

    def _mamba_with_degree_perm(self, h, edge_index, batch, add_noise=True):
        """Degree-based permutation + optional noise → Mamba scan → un-permute.

        原始 Graph-Mamba 使用 lexsort([deg+noise, batch]) 升序排列,
        noise 为 uniform [0,1) (torch.rand_like).

        注意: Mamba SSM 在 float16 下数值不稳定 (值会跨层爆炸),
        因此强制 float32 计算, 与原始 Graph-Mamba 一致 (不使用 AMP).
        """
        deg = degree(edge_index[0], h.shape[0]).to(torch.float)
        if add_noise:
            # 原始代码: torch.rand_like(deg) — 均匀 [0,1) 噪声
            deg_noise = torch.rand_like(deg)
            deg = deg + deg_noise

        # lexsort: 先按 batch 排序, 再按 deg 排序 (升序)
        h_ind_perm = _lexsort([deg, batch])
        # 强制 float32: Mamba SSM 在 float16 下不稳定
        h_float = h[h_ind_perm].float()
        batch_perm = batch[h_ind_perm]
        h_dense, mask = to_dense_batch(h_float, batch_perm)
        h_ind_perm_reverse = torch.argsort(h_ind_perm)
        h_attn = self.mamba(h_dense)[mask][h_ind_perm_reverse]
        return h_attn.to(h.dtype)  # 转回原始 dtype

    def forward(self, x, edge_index, edge_attr, batch):
        # AMP float16 下 GINEConv 消息聚合和 Mamba SSM 均可能溢出。
        # 原始 Graph-Mamba 不使用 AMP，此处禁用 autocast 强制 float32。
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            edge_attr = edge_attr.float() if edge_attr is not None else None
            h_in = x

            h_out_list = []

            # 1. Local: GINE → dropout → residual → post-norm
            h_local = self.conv(x, edge_index, edge_attr)
            h_local = self.dropout_local(h_local)
            h_local = h_in + h_local
            h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

            # 2. Global: Mamba with degree permutation + noise
            if self.training:
                h_attn = self._mamba_with_degree_perm(x, edge_index, batch, add_noise=True)
            else:
                mamba_arr = []
                for _ in range(self.eval_samples):
                    h_attn_i = self._mamba_with_degree_perm(x, edge_index, batch, add_noise=True)
                    mamba_arr.append(h_attn_i)
                h_attn = sum(mamba_arr) / len(mamba_arr)

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in + h_attn
            h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

            # 3. Combine
            h = sum(h_out_list)

            # 4. FFN + post-norm
            h = h + self.ff_dropout2(self.ff_linear2(
                self.ff_dropout1(F.relu(self.ff_linear1(h)))))
            h = self.norm2(h)

            return h


# ==================================================================
# Encoders
# ==================================================================

class GNNPlusEncoder(nn.Module):
    """Pure GINE message-passing encoder (GNNPlus, ICML 2025).

    GNNPlus 的核心 insight: 经典的 GNN (GINE) + RWSE + 训练技巧 (residual,
    FFN, norm, dropout, edge features) 就能打败 graph transformer,
    不需要 global attention.

    蛋白图 (100-2500 节点) 用 global_mean_pool 会把 binding pocket 信号
    稀释 5-10 倍。改为 attention pooling: 每个节点学习自己的重要性权重,
    加权求和替代等权平均。
    """
    def __init__(self, emb_dim=128, num_layers=5, dropout=0.1,
                 use_edge_attr=True, use_attn_pool=True, use_multiscale=True,
                 gate_hidden_dim=None):
        super().__init__()
        self.use_attn_pool = use_attn_pool
        self.use_multiscale = use_multiscale
        edge_dim = emb_dim if use_edge_attr else None
        self.layers = nn.ModuleList([
            GINEBlock(emb_dim, dropout=dropout, residual=True, ffn=True, edge_dim=edge_dim)
            for _ in range(num_layers)
        ])

        if use_attn_pool:
            # gated attention: sigmoid gate per node, 加权求和
            # 加宽 gate 瓶颈以更好区分 500+ 节点中的结合袋残基
            gate_hidden = gate_hidden_dim if gate_hidden_dim is not None else emb_dim
            self.attn_gate = nn.Sequential(
                nn.Linear(emb_dim, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, 1),
            )

        if use_multiscale:
            # 多尺度: concat 各层输出后再 attention pool
            self.ms_fusion = nn.Linear(emb_dim * num_layers, emb_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        layer_outs = []
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            layer_outs.append(x)

        if self.use_multiscale:
            x = torch.cat(layer_outs, dim=-1)
            x = self.ms_fusion(x)

        if self.use_attn_pool:
            gate = torch.sigmoid(self.attn_gate(x))  # [N, 1]
            num = global_add_pool(x * gate, batch)
            den = global_add_pool(gate, batch) + 1e-8
            return num / den
        else:
            return global_mean_pool(x, batch)


class GraphGPSEncoder(nn.Module):
    """GraphGPS encoder: stacked GPSLayers (GINE + Transformer).

    Reference: rampasek/GraphGPS (NeurIPS 2022).
    """
    def __init__(self, emb_dim=128, num_layers=5, heads=4, dropout=0.1,
                 attn_dropout=0.1, use_edge_attr=True):
        super().__init__()
        edge_dim = emb_dim if use_edge_attr else None
        self.layers = nn.ModuleList([
            GPSLayer(
                dim=emb_dim, heads=heads,
                dropout=dropout, attn_dropout=attn_dropout,
                edge_dim=edge_dim,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_attr, batch):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)
        return global_mean_pool(x, batch)


class GraphMambaEncoder(nn.Module):
    """GraphMamba encoder: stacked GraphMambaLayers (GINE + Mamba + degree perm).

    Reference: bowang-lab/Graph-Mamba (2024).
    Core innovation: degree-based node permutation + noise augmentation.
    """
    def __init__(
        self, emb_dim=128, num_layers=5, dropout=0.1,
        d_state=16, d_conv=4, eval_samples=5, use_edge_attr=True,
    ):
        super().__init__()
        edge_dim = emb_dim if use_edge_attr else None
        self.layers = nn.ModuleList([
            GraphMambaLayer(
                dim=emb_dim, dropout=dropout,
                d_state=d_state, d_conv=d_conv,
                eval_samples=eval_samples, edge_dim=edge_dim,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_attr, batch):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)
        return global_mean_pool(x, batch)


# ==================================================================
# Factory
# ==================================================================

def build_graph_encoder(encoder_type, emb_dim, num_layers, dropout=0.1, **kwargs):
    """Build a graph encoder by type.

    Args:
        encoder_type: "gnnplus" | "graphgps" | "graphmamba"
                      (also accepts "gine" as alias for "gnnplus")
        emb_dim: hidden dimension
        num_layers: number of layers
        dropout: dropout rate
        **kwargs: encoder-specific options
            - graphgps: heads, attn_dropout
            - graphmamba: d_state, d_conv, perm_noise
    """
    use_edge_attr = kwargs.get("use_edge_attr", True)

    if encoder_type in ("gnnplus", "gine"):
        return GNNPlusEncoder(
            emb_dim=emb_dim, num_layers=num_layers,
            dropout=dropout,
            use_edge_attr=use_edge_attr,
            use_attn_pool=kwargs.get("use_attn_pool", True),
            use_multiscale=kwargs.get("use_multiscale", True),
            gate_hidden_dim=kwargs.get("gate_hidden_dim", None),
        )

    elif encoder_type == "graphgps":
        return GraphGPSEncoder(
            emb_dim=emb_dim, num_layers=num_layers,
            heads=kwargs.get("heads", 4),
            dropout=dropout,
            attn_dropout=kwargs.get("attn_dropout", 0.1),
            use_edge_attr=use_edge_attr,
        )

    elif encoder_type == "graphmamba":
        return GraphMambaEncoder(
            emb_dim=emb_dim, num_layers=num_layers,
            dropout=dropout,
            d_state=kwargs.get("d_state", 16),
            d_conv=kwargs.get("d_conv", 4),
            eval_samples=kwargs.get("eval_samples", 5),
            use_edge_attr=use_edge_attr,
        )

    else:
        raise ValueError(
            f"Unknown encoder_type: {encoder_type}. "
            f"Choose from: gnnplus, graphgps, graphmamba"
        )


# =========================
# Attention Fusion
# =========================
class MultiHeadAttentionFusion(nn.Module):
    """Multi-head attention fusion from RAHG.

    原始 RAHG 使用 exp(r) / (exp(r) + exp(s)) 做归一化,
    这在 float32 下没问题, 但在 AMP float16 下 exp() 会溢出 (x>11.09 → inf).
    修复: 用数值等价的 softmax 替代裸 exp, 并强制 float32 计算.

    Reference: RAHG/Tea_Stu/model.py :: AttentionModel
    """
    def __init__(self, dim1, dim2, nhid=256, heads=8, dropout=0.1):
        super().__init__()
        self.nhid = nhid
        self.heads = heads
        self.dropout_rate = dropout

        self.linear_r = nn.Linear(dim1, nhid)
        self.linear_s = nn.Linear(dim2, nhid)
        self.q_r = nn.Parameter(torch.rand(nhid, heads))
        self.q_s = nn.Parameter(torch.rand(nhid, heads))
        self.att_fusion = nn.Linear(heads * nhid, nhid)

    def forward(self, r_feats, s_feats):
        # 强制 float32 防止 AMP float16 下 exp 溢出
        r_feats = r_feats.float()
        s_feats = s_feats.float()

        r_transformed = torch.tanh(self.linear_r(r_feats))
        s_transformed = torch.tanh(self.linear_s(s_feats))

        r_alpha = torch.matmul(r_transformed, self.q_r)  # [B, heads]
        s_alpha = torch.matmul(s_transformed, self.q_s)  # [B, heads]

        # 数值稳定: stack → softmax 等价于 exp(r)/(exp(r)+exp(s))
        # softmax 内部会减去 max 防止溢出
        logits = torch.stack([r_alpha, s_alpha], dim=-1)  # [B, heads, 2]
        weights = F.softmax(logits, dim=-1)               # [B, heads, 2]
        r_alpha_norm = weights[:, :, 0]  # [B, heads]
        s_alpha_norm = weights[:, :, 1]  # [B, heads]

        fusion_x = torch.cat(
            [r_alpha_norm[:, i].unsqueeze(1) * r_transformed +
             s_alpha_norm[:, i].unsqueeze(1) * s_transformed
             for i in range(self.heads)],
            dim=1)
        fusion_x = self.att_fusion(fusion_x)
        return fusion_x, r_alpha_norm, s_alpha_norm


# =========================
# Main DualGraph Model
# =========================
class DTApredictorV2(torch.nn.Module):
    """Dual-graph DTI predictor with configurable encoders for molecule and protein.

    Supports three encoder types per branch:
      - "gnnplus":   Pure GINE (GNNPlus, ICML 2025)
      - "graphgps":  GINE + Transformer (GraphGPS, NeurIPS 2022)
      - "graphmamba": GINE + Mamba with degree perm (GraphMamba, 2024)
    """

    def __init__(self, **config):
        super().__init__()

        # ── Molecule config ──
        mol_encoder_type = config.get("mol_encoder_type", "graphmamba")
        mol_dim = config.get("mol_emb_dim", 128)
        mol_num_layers = config.get("mol_num_layers", 5)
        mol_dropout = config.get("mol_dropout", 0.1)
        mol_use_rwse = config.get("mol_use_rwse", True)
        mol_num_rw_steps = config.get("mol_num_rw_steps", 5)
        mol_rwse_dim = config.get("mol_rwse_dim", 64)
        mol_rwse_encoder_type = config.get("mol_rwse_encoder_type", "linear")
        mol_molt5_out = config.get("mol_molt5_output_dim", 128)

        # ── Protein config ──
        prot_encoder_type = config.get("prot_encoder_type", "gnnplus")
        prot_dim = config.get("prot_emb_dim", 128)
        prot_num_layers = config.get("prot_num_layers", 3)
        prot_dropout = config.get("prot_dropout", 0.1)
        prot_node_in = config.get("prot_node_input_dim", 31)
        prot_edge_in = config.get("prot_edge_input_dim", 5)
        prot_prot5_out = config.get("prot_prot5_output_dim", 256)

        # ── Fusion config ──
        fusion_nhid = config.get("fusion_nhid", 256)
        fusion_heads = config.get("fusion_heads", 8)

        # ── Output config ──
        n_output = config.get("n_output", 1)
        final_dropout = config.get("final_dropout", 0.2)

        self.mol_dim = mol_dim
        self.mol_rwse_dim = mol_rwse_dim
        self.mol_use_rwse = mol_use_rwse
        self.mol_use_float = config.get("mol_use_float", True)

        # ========================
        # Molecule Branch
        # ========================
        atom_emb_dim = mol_dim - mol_rwse_dim if mol_use_rwse else mol_dim
        self.atom_encoder = AtomEncoder(atom_emb_dim)
        self.bond_encoder = BondEncoder(mol_dim)

        if self.mol_use_float:
            # RBF 编码 3 个连续原子特征: vdw_radius, partial_charge, mass
            self.atom_float_rbf = AtomFloatRBF(atom_emb_dim)

        if mol_use_rwse:
            self.rwse_encoder = RWSENodeEncoder(
                num_steps=mol_num_rw_steps,
                rwse_dim=mol_rwse_dim,
                encoder_type=mol_rwse_encoder_type,
                norm_type='batchnorm',
            )

        mol_encoder_kwargs = {}
        if mol_encoder_type == "graphmamba":
            mol_encoder_kwargs.update({
                "d_state": config.get("mol_mamba_d_state", 16),
                "d_conv": config.get("mol_mamba_d_conv", 4),
                "eval_samples": config.get("mol_mamba_eval_samples", 5),
            })
        elif mol_encoder_type == "graphgps":
            mol_encoder_kwargs["heads"] = config.get("mol_gps_heads", 4)

        self.mol_graph_encoder = build_graph_encoder(
            mol_encoder_type, mol_dim, mol_num_layers, mol_dropout,
            use_edge_attr=True, **mol_encoder_kwargs,
        )

        self.molt5_adapter = Molt5Adapter(input_dim=768, output_dim=mol_molt5_out, dropout=0.1)
        self.drug_molt5_fusion = MultiHeadAttentionFusion(
            dim1=mol_dim, dim2=mol_molt5_out,
            nhid=fusion_nhid, heads=fusion_heads, dropout=mol_dropout,
        )

        # ========================
        # Protein Branch
        # ========================
        self.prot_node_encoder = ProteinNodeEncoder(
            input_dim=prot_node_in, output_dim=prot_dim, dropout=prot_dropout,
        )
        self.prot_edge_encoder = ProteinEdgeEncoder(
            input_dim=prot_edge_in, output_dim=prot_dim,
            use_dist_rbf=config.get("prot_use_dist_rbf", False),
        )

        prot_encoder_kwargs = {}
        if prot_encoder_type in ("gnnplus", "gine"):
            gate_hid = config.get("prot_gate_hidden_dim", None)
            if gate_hid is not None:
                prot_encoder_kwargs["gate_hidden_dim"] = gate_hid
        elif prot_encoder_type == "graphmamba":
            prot_encoder_kwargs.update({
                "d_state": config.get("prot_mamba_d_state", 16),
                "d_conv": config.get("prot_mamba_d_conv", 4),
                "eval_samples": config.get("prot_mamba_eval_samples", 5),
            })
        elif prot_encoder_type == "graphgps":
            prot_encoder_kwargs["heads"] = config.get("prot_gps_heads", 4)

        self.prot_graph_encoder = build_graph_encoder(
            prot_encoder_type, prot_dim, prot_num_layers, prot_dropout,
            use_edge_attr=True, **prot_encoder_kwargs,
        )

        self.prot5_adapter = ProteinAdapter(input_dim=1024, output_dim=prot_prot5_out, dropout=prot_dropout)
        self.prot_graph_fusion = MultiHeadAttentionFusion(
            dim1=prot_dim, dim2=prot_prot5_out,
            nhid=fusion_nhid, heads=fusion_heads, dropout=prot_dropout,
        )

        # ========================
        # Output MLP
        # ========================
        self.fc1 = nn.Linear(fusion_nhid * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, n_output)
        self.dropout_layer = nn.Dropout(final_dropout)
        self.relu = nn.ReLU()

    def forward(self, data):
        # ========================
        # Molecule Branch
        # ========================
        int_part = data.x[:, :NUM_INT_FEATURES].long()
        base_emb = self.atom_encoder(int_part)

        if self.mol_use_float and data.x.shape[1] >= NUM_INT_FEATURES + NUM_FLOAT_FEATURES:
            float_part = data.x[:, NUM_INT_FEATURES:NUM_INT_FEATURES + NUM_FLOAT_FEATURES]
            float_emb = self.atom_float_rbf(float_part)
            base_emb = base_emb + float_emb

        if self.mol_use_rwse and hasattr(data, 'pestat_RWSE'):
            rwse_emb = self.rwse_encoder(data.pestat_RWSE)
            x_mol = torch.cat([base_emb, rwse_emb], dim=-1)
        else:
            x_mol = base_emb

        mol_edge_emb = self.bond_encoder(data.edge_attr.long())
        drug_graph_emb = self.mol_graph_encoder(
            x_mol, data.edge_index, mol_edge_emb, data.batch,
        )

        molt5_feat = self.molt5_adapter(data.molt5_emb)
        drug_emb, _, _ = self.drug_molt5_fusion(drug_graph_emb, molt5_feat)

        # ========================
        # Protein Branch
        # ========================
        # 后 3 维是原始 3D 坐标 (非旋转不变, 且半径图边已编码距离+方向)
        x_prot = self.prot_node_encoder(data.prot_node_x[:, :28])
        prot_edge_emb = self.prot_edge_encoder(data.prot_edge_attr)

        prot_batch = getattr(data, 'prot_batch', None)
        if prot_batch is None:
            prot_batch = torch.zeros(data.prot_node_x.shape[0], dtype=torch.long, device=x_prot.device)

        prot_graph_emb = self.prot_graph_encoder(
            x_prot, data.prot_edge_index, prot_edge_emb, prot_batch,
        )

        prot5_feat = self.prot5_adapter(data.target)
        prot_emb, _, _ = self.prot_graph_fusion(prot_graph_emb, prot5_feat)

        # ========================
        # Prediction
        # ========================
        combined = torch.cat([drug_emb, prot_emb], dim=-1)

        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        out = self.out(x)

        return out


