"""Frozen T5 pooled features following create_data_scope_20260406.py.

Transformers is imported only when loading encoder weights. No model is
downloaded unless an explicit model identifier is passed and downloads
are enabled. Tests can use the pooling/preprocessing functions offline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any, Sequence

import torch


def prepare_encoder_texts(inputs: Sequence[str], kind: str) -> list[str]:
    """Preserve SMILES; apply the retained ProtT5 truncation and spacing."""
    if kind not in ("molt5", "prot5"):
        raise ValueError("kind must be molt5 or prot5")
    output = []
    for index, value in enumerate(inputs):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Input {index} is empty")
        if kind == "prot5":
            if not re.fullmatch(r"[A-Z]+", value):
                raise ValueError(f"Protein input {index} must be an uppercase sequence without spaces")
            if len(value) > 1000:
                value = value[:500] + value[-500:]
            value = " ".join(re.sub(r"[UZOB]", "X", value))
        output.append(value)
    return output


def masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean over all attended tokens, including an attended EOS token."""
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("Expected hidden [B,L,D] and attention_mask [B,L]")
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError("Encoder output contains nonfinite values")
    mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    counts = mask.sum(dim=1)
    if bool((counts <= 0).any()):
        raise ValueError("Every sequence must contain at least one attended token")
    return (hidden * mask).sum(dim=1) / counts


def encode_with_loaded_model(
    inputs: Sequence[str], *, kind: str, tokenizer: Any, model: Any,
    batch_size: int = 4, device: str | torch.device = "cpu", max_length: int | None = None,
) -> torch.Tensor:
    """Encode using caller-supplied T5 objects; useful for offline tests."""
    texts = prepare_encoder_texts(inputs, kind)
    if not texts:
        raise ValueError("At least one input is required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_length is None:
        max_length = 512 if kind == "molt5" else 1024
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    encoder = model.to(device).eval()
    features = []
    for start in range(0, len(texts), batch_size):
        tokenized = tokenizer.batch_encode_plus(texts[start:start + batch_size],
            add_special_tokens=True, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        with torch.inference_mode():
            hidden = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            features.append(masked_mean_pool(hidden, attention_mask).float().cpu())
    return torch.cat(features, dim=0)


def extract_embeddings(
    inputs: Sequence[str], *, kind: str, model_path: str | Path,
    batch_size: int = 4, device: str | torch.device = "cpu", max_length: int | None = None,
    local_files_only: bool = True,
) -> torch.Tensor:
    """Load an explicit T5 encoder and return CPU float32 pooled vectors."""
    if kind not in ("molt5", "prot5"):
        raise ValueError("kind must be molt5 or prot5")
    try:
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError as exc:
        raise ImportError("Embedding extraction requires pcbm2s[embeddings]") from exc
    length = max_length if max_length is not None else (512 if kind == "molt5" else 1024)
    kwargs = {"local_files_only": local_files_only, "model_max_length": length}
    if kind == "prot5":
        kwargs["do_lower_case"] = False
    tokenizer = T5Tokenizer.from_pretrained(str(model_path), **kwargs)
    tokenizer.padding_side = "right"
    model = T5EncoderModel.from_pretrained(str(model_path), local_files_only=local_files_only)
    return encode_with_loaded_model(inputs, kind=kind, tokenizer=tokenizer, model=model,
        batch_size=batch_size, device=device, max_length=length)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen MolT5 or ProtT5 pooled vectors")
    parser.add_argument("--model-path", required=True, help="Local model directory or explicit Hugging Face identifier")
    parser.add_argument("--kind", choices=("molt5", "prot5"), required=True)
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 text: one SMILES or sequence per line")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true", help="Allow fetching the explicitly specified model")
    args = parser.parse_args()
    inputs = args.input.read_text(encoding="utf-8-sig").splitlines()
    vectors = extract_embeddings(inputs, kind=args.kind, model_path=args.model_path,
        batch_size=args.batch_size, device=args.device, max_length=args.max_length,
        local_files_only=not args.allow_download)
    expected = 768 if args.kind == "molt5" else 1024
    if vectors.size(1) != expected:
        raise ValueError(f"Selected encoder returned {vectors.size(1)} features; default {args.kind} adapter expects {expected}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": args.kind, "inputs": inputs, "embeddings": vectors,
        "model": str(args.model_path), "dim": int(vectors.size(1)),
        "pooling": "attention_mask_mean_including_eos", "dtype": "float32",
        "max_length": args.max_length or (512 if args.kind == "molt5" else 1024)}, args.output)
    print(f"Saved {vectors.size(0)} {args.kind} embeddings to {args.output}")


if __name__ == "__main__":
    main()
