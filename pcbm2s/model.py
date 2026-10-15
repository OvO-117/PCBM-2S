"""Pair encoding, descriptor conditioning and auxiliary supervision."""

from __future__ import annotations

import math
import torch
from torch import nn

from .backbone import DTApredictorV2, NUM_INT_FEATURES, NUM_FLOAT_FEATURES, RWSENodeEncoder
from .data import CONCEPT_NAMES


class PCBM2S(DTApredictorV2):
    """Binary DTI predictor with separate concatenation and concept-gate paths.

    ``forward`` returns logits, not probabilities. ``c_struct`` and ``c_gate``
    in ``head_from_pair`` are standardized descriptors. Keeping them separate
    permits gate-only attribution without changing the concatenation branch.
    """

    def __init__(self, descriptor_stats: dict, **config):
        if config.get("n_output", 1) != 1:
            raise ValueError("This release implements binary DTI (n_output=1)")
        config.setdefault("prot_node_input_dim", 28)
        config.setdefault("prot_num_layers", 6)
        config.setdefault("prot_use_rwse", True)
        if config["prot_node_input_dim"] != 28:
            raise ValueError("Protein node features must use the first 28 columns (no absolute coordinates)")
        if config.get("mol_use_rwse", True) and not 0 < config.get("mol_rwse_dim", 64) < config.get("mol_emb_dim", 128):
            raise ValueError("mol_rwse_dim must be smaller than mol_emb_dim")
        super().__init__(**config)
        self.struct_feature_names = list(CONCEPT_NAMES)
        self.n_concept = len(CONCEPT_NAMES)
        self.use_gate = bool(config.get("use_gate", True))
        self.use_concept_head = bool(config.get("use_concept_head", True))
        self.gate_scale = float(config.get("gate_scale", 1.0))
        if not math.isfinite(self.gate_scale) or self.gate_scale < 0:
            raise ValueError("gate_scale must be a finite nonnegative value")
        self.prot_use_rwse = bool(config.get("prot_use_rwse", False))
        self.prot_num_rw_steps = int(config.get("prot_num_rw_steps", 5))
        if self.prot_use_rwse:
            self.prot_rwse_encoder = RWSENodeEncoder(
                self.prot_num_rw_steps, config.get("prot_emb_dim", 128),
                encoder_type=config.get("prot_rwse_encoder_type", "linear"),
            )
        means, stds = [], []
        for name in CONCEPT_NAMES:
            item = descriptor_stats[name]
            mean, std = float(item["mean"]), float(item["std"])
            if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
                raise ValueError(f"Invalid training statistics for {name}")
            means.append(mean)
            stds.append(std)
        self.register_buffer("struct_mean", torch.tensor(means, dtype=torch.float32))
        self.register_buffer("struct_std", torch.tensor(stds, dtype=torch.float32))
        self.struct_encoder = nn.Sequential(
            nn.Linear(self.n_concept, 32), nn.ReLU(),
            nn.Dropout(config.get("final_dropout", 0.25)),
        )
        pair_dim = config.get("fusion_nhid", 256) * 2
        self.fc1 = nn.Linear(pair_dim + 32, 512)
        if self.use_gate:
            hidden = int(config.get("gate_hidden", 0))
            if hidden > 0:
                self.gate_net = nn.Sequential(nn.Linear(self.n_concept, hidden), nn.ReLU(), nn.Linear(hidden, 512))
                final = self.gate_net[-1]
            else:
                self.gate_net = nn.Linear(self.n_concept, 512)
                final = self.gate_net
            # Initialize the gate to identity modulation.
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        if self.use_concept_head:
            hidden = int(config.get("concept_head_hidden", 128))
            self.concept_head = nn.Sequential(nn.Linear(pair_dim, hidden), nn.ReLU(), nn.Linear(hidden, self.n_concept))

    def normalized_concepts(self, data):
        raw = torch.cat([getattr(data, name).reshape(-1, 1) for name in CONCEPT_NAMES], dim=1)
        return (raw - self.struct_mean) / (self.struct_std + 1e-8)

    def compute_gate(self, c):
        if not self.use_gate:
            return c.new_zeros((c.shape[0], 512))
        return self.gate_scale * torch.tanh(self.gate_net(c))

    def encode_pair(self, data):
        """Encode each entity once; no descriptors enter the pair encoder."""
        base = self.atom_encoder(data.x[:, :NUM_INT_FEATURES].long())
        if self.mol_use_float:
            base = base + self.atom_float_rbf(data.x[:, NUM_INT_FEATURES:NUM_INT_FEATURES + NUM_FLOAT_FEATURES])
        if self.mol_use_rwse:
            rwse = getattr(data, "pestat_RWSE", None)
            if rwse is None:
                raise ValueError("Missing ligand pestat_RWSE: run pcbm2s.preprocess")
            base = torch.cat([base, self.rwse_encoder(rwse)], dim=-1)
        atom_batch = getattr(data, "batch", None)
        if atom_batch is None:
            atom_batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=base.device)
        drug_graph = self.mol_graph_encoder(base, data.edge_index, self.bond_encoder(data.edge_attr.long()), atom_batch)
        drug, _, _ = self.drug_molt5_fusion(drug_graph, self.molt5_adapter(data.molt5_emb))
        protein = self.prot_node_encoder(data.prot_node_x[:, :28])
        if self.prot_use_rwse:
            rwse = getattr(data, "prot_pestat_RWSE", None)
            if rwse is None or rwse.shape != (protein.size(0), self.prot_num_rw_steps):
                raise ValueError("Missing or malformed prot_pestat_RWSE; prepare protein RWSE before training")
            protein = protein + self.prot_rwse_encoder(rwse)
        protein_batch = getattr(data, "prot_node_x_batch", None)
        if protein_batch is None:
            protein_batch = getattr(data, "prot_batch", None)
        if protein_batch is None:
            if drug_graph.shape[0] != 1:
                raise ValueError("Batched protein graphs require make_loader(...), with independent protein membership")
            protein_batch = torch.zeros(protein.shape[0], dtype=torch.long, device=protein.device)
        protein_graph = self.prot_graph_encoder(protein, data.prot_edge_index, self.prot_edge_encoder(data.prot_edge_attr), protein_batch)
        prot, _, _ = self.prot_graph_fusion(protein_graph, self.prot5_adapter(data.target))
        if drug.shape[0] != prot.shape[0]:
            raise ValueError("Ligand/protein batch sizes disagree")
        return torch.cat([drug, prot], dim=-1)

    def head_from_pair(self, z, c_struct, c_gate=None, use_gate=None):
        c_gate = c_struct if c_gate is None else c_gate
        if c_struct.ndim != 2 or c_struct.shape != c_gate.shape or c_struct.shape != (z.shape[0], self.n_concept):
            raise ValueError("Concept tensors must have shape [batch, 5]")
        h = self.fc1(torch.cat([z, self.struct_encoder(c_struct)], dim=-1))
        gate_enabled = self.use_gate if use_gate is None else bool(use_gate)
        if gate_enabled:
            h = (1.0 + self.compute_gate(c_gate)) * h
        h = self.dropout_layer(self.relu(h))
        h = self.dropout_layer(self.relu(self.fc2(h)))
        return self.out(h)

    def forward(self, data, return_aux=False):
        z = self.encode_pair(data)
        c = self.normalized_concepts(data)
        logits = self.head_from_pair(z, c)
        if return_aux:
            return logits, self.concept_head(z) if self.use_concept_head else None, c
        return logits
