"""Train P-CBM-2S on prepared dual-graph pairs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import torch
from torch.nn import functional as F

from .common import classification_metrics, predict, read_config, resolve_device, seed_everything, validation_threshold, write_json
from .data import fit_descriptor_stats, load_split, make_loader


def train_epoch(model, loader, optimizer, device, *, aux_weight=0.1, label_smoothing=0.05, grad_clip=1.0):
    model.train()
    total, count = 0.0, 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.y.numel() < 2:
            raise ValueError("Training batches need at least two pairs for BatchNorm. Change batch_size or remove a singleton remainder.")
        optimizer.zero_grad(set_to_none=True)
        logits, predicted, concepts = model(batch, return_aux=True)
        labels = batch.y.reshape(-1)
        if logits.numel() != labels.numel():
            raise ValueError(f"Batch {step}: label/output shape mismatch")
        smoothed = labels * (1 - label_smoothing) + label_smoothing / 2
        loss = F.binary_cross_entropy_with_logits(logits.reshape(-1), smoothed)
        if predicted is not None and aux_weight:
            loss = loss + aux_weight * F.mse_loss(predicted, concepts)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Batch {step}: nonfinite loss; training stopped (no batches silently discarded)")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        count += labels.numel()
        total += loss.item() * labels.numel()
    if count == 0:
        raise ValueError("No training examples")
    return total / count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    config = read_config(args.config)
    settings, model_config = config["training"], config["model"]
    if not 0 <= settings["label_smoothing"] < 1 or settings["aux_weight"] < 0:
        parser.error("Invalid label smoothing or auxiliary weight")
    if settings["epochs"] < 1 or settings["patience"] < 1 or settings["batch_size"] < 2:
        parser.error("Need epochs/patience >=1 and batch_size >=2")
    destination = Path(args.output_dir)
    if destination.exists() and any(destination.iterdir()):
        parser.error("Output directory is not empty; use a separate directory per run")
    destination.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    root = Path(args.data_dir)
    prepare_rwse = model_config.get("prot_use_rwse", False)
    train = load_split(root / "train.pt", add_protein_rwse=prepare_rwse)
    val = load_split(root / "val.pt", add_protein_rwse=prepare_rwse)
    if len(train) % settings["batch_size"] == 1:
        parser.error("Training split leaves a singleton BatchNorm batch; choose another batch_size (no examples are dropped)")
    stats = fit_descriptor_stats(train)
    from .model import PCBM2S
    model = PCBM2S(stats, **model_config).to(device)
    train_loader = make_loader(train, settings["batch_size"], True, settings.get("num_workers", 0))
    val_loader = make_loader(val, settings["batch_size"], False, settings.get("num_workers", 0))
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    write_json(destination / "config.json", {**config, "seed": args.seed})
    write_json(destination / "descriptor_stats.json", stats)
    best, stale, history = float("-inf"), 0, []
    for epoch in range(1, settings["epochs"] + 1):
        started = time.monotonic()
        loss = train_epoch(model, train_loader, optimizer, device,
                           aux_weight=settings["aux_weight"], label_smoothing=settings["label_smoothing"],
                           grad_clip=settings.get("grad_clip", 1.0))
        # Same seed for validation permutation samples each epoch.
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            torch.manual_seed(args.seed + 100000)
            labels, probabilities = predict(model, val_loader, device)
        threshold = validation_threshold(labels, probabilities)
        metrics = classification_metrics(labels, probabilities, threshold)
        score = metrics["auroc"]
        scheduler.step(score)
        record = {"epoch": epoch, "train_loss": loss, "validation": metrics, "seconds": time.monotonic() - started}
        history.append(record)
        write_json(destination / "history.json", history)
        print(f"epoch={epoch} loss={loss:.6f} val_AUROC={score:.6f} val_AP={metrics['ap']:.6f}", flush=True)
        if score > best:
            best, stale = score, 0
            payload = {"schema_version": 1, "model_config": model_config, "descriptor_stats": stats,
                       "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       "epoch": epoch, "seed": args.seed, "threshold": threshold,
                       "validation_metrics": metrics, "training_config": settings,
                       "torch_version": torch.__version__}
            torch.save(payload, destination / "best.pt")
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    # Test labels never select checkpoints or thresholds. Evaluation is a separate command.
    write_json(destination / "summary.json", {"best_validation_auroc": best, "epochs_completed": len(history), "seed": args.seed})
    print(f"Saved validation-selected checkpoint: {destination / 'best.pt'}")


if __name__ == "__main__":
    main()
