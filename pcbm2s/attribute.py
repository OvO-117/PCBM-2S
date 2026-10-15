"""Exact Shapley attribution of the five-input concept-gate pathway.

The pair representation is encoded once. Coalition evaluations vary only the
gate inputs; the measured-descriptor concatenation branch remains fixed.
An absent standardized concept is set to its training-set mean (zero).
Contributions decompose the probability change along this conditional pathway.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch


@contextmanager
def _evaluation(model: torch.nn.Module) -> Iterator[None]:
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


def exact_gate_shapley(
    model: torch.nn.Module,
    z: torch.Tensor,
    c_struct: torch.Tensor,
    concept_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Explain a single fixed pair using all 32 concept coalitions.

    ``z`` must have shape ``[1, pair_width]`` and ``c_struct`` must have shape
    ``[1, 5]``. Concepts must already be standardized by the checkpoint's
    training statistics. ``gate_effect`` is full minus mean-concept-baseline
    probability. This mean-concept baseline is distinct from disabling the gate.
    Signed contributions sum to the gate effect; a disabled gate yields zeros.
    """
    if not isinstance(z, torch.Tensor) or z.ndim != 2 or z.size(0) != 1:
        raise ValueError("z must be a tensor with shape [1, pair_width]")
    if not isinstance(c_struct, torch.Tensor) or c_struct.shape != (1, 5):
        raise ValueError("c_struct must be a tensor with shape [1, 5]")
    if not z.is_floating_point() or not c_struct.is_floating_point():
        raise ValueError("z and c_struct must be floating-point tensors")
    if z.device != c_struct.device:
        raise ValueError("z and c_struct must be on the same device")
    if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(c_struct).all()):
        raise ValueError("z and c_struct must contain finite values")
    n_concepts = int(getattr(model, "n_concept", 5))
    if n_concepts != 5:
        raise ValueError("This release attributes exactly five concepts")
    names = list(concept_names if concept_names is not None else model.struct_feature_names)
    if len(names) != n_concepts or len(set(names)) != n_concepts:
        raise ValueError("Exactly five unique concept names are required")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Concept names must be nonempty strings")

    coalition_count = 1 << n_concepts
    masks = torch.tensor(
        [[bool(mask & (1 << i)) for i in range(n_concepts)] for mask in range(coalition_count)],
        dtype=c_struct.dtype,
        device=c_struct.device,
    )
    fixed_z = z.detach().expand(coalition_count, -1)
    fixed_struct = c_struct.detach().expand(coalition_count, -1)
    coalition_gate = masks * fixed_struct
    with _evaluation(model):
        # Reuse the fixed pair representation across all coalitions.
        logits = model.head_from_pair(fixed_z, fixed_struct, c_gate=coalition_gate)
        if not isinstance(logits, torch.Tensor) or logits.numel() != coalition_count:
            raise ValueError("head_from_pair must return one logit per coalition")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("The model produced a nonfinite coalition logit")
        probabilities = logits.reshape(-1).sigmoid().double().cpu()

    contributions: dict[str, float] = {}
    denominator = math.factorial(n_concepts)
    for i, name in enumerate(names):
        value = 0.0
        for mask in range(coalition_count):
            if mask & (1 << i):
                continue
            size = bin(mask).count("1")  # Python 3.9-compatible population count.
            weight = math.factorial(size) * math.factorial(n_concepts - size - 1) / denominator
            value += weight * float(probabilities[mask | (1 << i)] - probabilities[mask])
        contributions[name] = value
    baseline = float(probabilities[0])
    full = float(probabilities[-1])
    effect = full - baseline
    return {
        "base_probability": baseline,
        "full_probability": full,
        "gate_effect": effect,
        "contributions": contributions,
        "efficiency_error": float(sum(contributions.values()) - effect),
        "n_coalitions": coalition_count,
        "baseline": "Each absent standardized gate concept is set to zero (training mean).",
        "fixed_inputs": "Pair representation and measured-descriptor concatenation are fixed.",
        "scope": "Concept-gate pathway in probability space; not whole-model or causal attribution.",
    }


def attribute_sample(model: torch.nn.Module, batch: Any) -> dict[str, Any]:
    """Encode one pair once, then explain its concept gate."""
    with _evaluation(model):
        z = model.encode_pair(batch)
        concepts = model.normalized_concepts(batch)
        result = exact_gate_shapley(model, z, concepts)
    result["standardized_concepts"] = {
        name: float(value)
        for name, value in zip(model.struct_feature_names, concepts.detach().cpu().reshape(-1))
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trusted trained checkpoint (.pt)")
    parser.add_argument("--data", required=True, help="Trusted prepared split (.pt)")
    parser.add_argument("--index", type=int, default=0, help="Zero-based index of the pair")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a PyTorch CUDA device")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the single graph encoding")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.index < 0:
        raise SystemExit("--index must be nonnegative")
    # Load model and data dependencies when running attribution.
    from .common import load_checkpoint
    from .data import load_split, make_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, _ = load_checkpoint(args.checkpoint, device)
    samples = load_split(args.data, add_protein_rwse=model.prot_use_rwse)
    if args.index >= len(samples):
        raise SystemExit(f"--index {args.index} is outside the split containing {len(samples)} pairs")
    batch = next(iter(make_loader([samples[args.index]], batch_size=1, shuffle=False))).to(device)
    result = attribute_sample(model, batch)
    result.update(sample_index=args.index, seed=args.seed, checkpoint=Path(args.checkpoint).name)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved gate attribution to {destination}; efficiency error={result['efficiency_error']:.3g}")


if __name__ == "__main__":
    main()
