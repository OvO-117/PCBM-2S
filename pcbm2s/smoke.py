"""Synthetic software check; this is not benchmark or biological evidence."""

import argparse
import json
import tempfile
from pathlib import Path
import torch

from .common import classification_metrics, load_checkpoint, resolve_device, seed_everything
from .data import CONCEPT_NAMES, DualGraphData, as_dual_graph, fit_descriptor_stats, make_loader, validate_dataset


def synthetic_pairs(count=4):
    """Create schema-valid toy tensors, never a replacement for real inputs."""
    samples = []
    for index in range(count):
        n, m = 4 + index % 2, 6 + index % 3
        edges = torch.stack([torch.arange(n), torch.arange(n).roll(1)])
        edges = torch.cat([edges, edges.flip(0)], dim=1)
        pedges = torch.stack([torch.arange(m), torch.arange(m).roll(1)])
        pedges = torch.cat([pedges, pedges.flip(0)], dim=1)
        x = torch.zeros(n, 21)
        x[:, 18:] = torch.tensor([1.7, 0.0, 12.0])
        sample = DualGraphData(
            x=x, edge_index=edges.long(), edge_attr=torch.zeros(edges.shape[1], 5, dtype=torch.long),
            prot_node_x=torch.rand(m, 28), prot_edge_index=pedges.long(), prot_edge_attr=torch.rand(pedges.shape[1], 5),
            molt5_emb=torch.randn(1, 768), target=torch.randn(1, 1024), y=torch.tensor([float(index % 2)]),
            **{name: torch.tensor([float(-7.0 + index) if j == 0 else float(index + j)]) for j, name in enumerate(CONCEPT_NAMES)},
        )
        samples.append(as_dual_graph(sample, add_protein_rwse=True))
    return samples


def diagnostic_config(encoder="gnnplus"):
    return dict(
        mol_encoder_type=encoder, mol_emb_dim=32, mol_num_layers=2, mol_dropout=0.0,
        mol_use_rwse=True, mol_num_rw_steps=5, mol_rwse_dim=8, mol_use_float=True,
        mol_molt5_output_dim=32, mol_mamba_d_state=8, mol_mamba_d_conv=4, mol_mamba_eval_samples=2,
        prot_encoder_type="gnnplus", prot_emb_dim=32, prot_num_layers=2, prot_dropout=0.0,
        prot_node_input_dim=28, prot_edge_input_dim=5, prot_prot5_output_dim=32, prot_use_dist_rbf=True,
        prot_use_rwse=True, prot_num_rw_steps=5, prot_rwse_encoder_type="linear",
        fusion_nhid=32, fusion_heads=2, n_output=1, final_dropout=0.0,
        use_gate=True, gate_scale=1.0, gate_hidden=0, use_concept_head=True, concept_head_hidden=16,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encoder", choices=["gnnplus", "graphmamba"], default="gnnplus",
                        help="Explicit diagnostic encoder; no silent fallback")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    seed_everything(42)
    device = resolve_device(args.device)
    from .model import PCBM2S
    from .train import train_epoch
    from .attribute import attribute_sample
    samples = synthetic_pairs()
    validate_dataset(samples)
    stats = fit_descriptor_stats(samples)
    config = diagnostic_config(args.encoder)
    model = PCBM2S(stats, **config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    loss = train_epoch(model, make_loader(samples, 4, False), optimizer, device)
    model.eval()
    batch = next(iter(make_loader(samples, 4, False))).to(device)
    with torch.no_grad():
        probability = model(batch).reshape(-1).sigmoid()
    if probability.shape != (4,) or not torch.isfinite(probability).all():
        raise AssertionError("Forward check failed")
    # Check checkpoint round-trip before stochastic GraphMamba sampling.
    with tempfile.TemporaryDirectory(prefix="pcbm2s_synthetic_") as directory:
        path = Path(directory) / "check.pt"
        torch.save({"schema_version": 1, "model_config": config, "descriptor_stats": stats, "state_dict": model.state_dict()}, path)
        restored, _ = load_checkpoint(path, device)
        seed_everything(123)
        with torch.no_grad():
            expected = model(batch)
        seed_everything(123)
        with torch.no_grad():
            actual = restored(batch)
        torch.testing.assert_close(actual, expected)
    attribution = attribute_sample(model, next(iter(make_loader(samples[:1], 1, False))).to(device))
    if abs(attribution["efficiency_error"]) > 1e-6:
        raise AssertionError("Shapley efficiency check failed")
    result = {
        "status": "passed", "synthetic_only": True, "encoder": args.encoder, "device_type": device.type,
        "protein_rwse": True, "train_loss": loss,
        "metrics": classification_metrics([int(s.y.item()) for s in samples], probability.cpu().tolist()),
        "attribution_efficiency_error": attribution["efficiency_error"],
        "checks": ["dual_graph_batching", "forward", "backward_optimizer_step", "protein_rwse", "checkpoint_roundtrip", "32_coalition_gate_shapley"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
