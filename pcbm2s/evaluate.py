"""Evaluate a saved checkpoint without tuning on test labels."""

import argparse
import csv
from pathlib import Path

from .common import classification_metrics, load_checkpoint, predict, resolve_device, seed_everything, write_json
from .data import load_split, make_loader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True, help="Prediction CSV (a .metrics.json sidecar is also written)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    output = Path(args.output)
    metrics_path = output.with_suffix(".metrics.json")
    if output.exists() or metrics_path.exists():
        parser.error("Output already exists; choose a new path")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    samples = load_split(args.data, add_protein_rwse=model.prot_use_rwse)
    loader = make_loader(samples, batch_size=args.batch_size, shuffle=False)
    labels, probabilities = predict(model, loader, device)
    threshold = float(checkpoint.get("threshold", 0.5))
    metrics = classification_metrics(labels, probabilities, threshold)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "label", "probability", "prediction"])
        writer.writerows((i, int(y), p, int(p >= threshold)) for i, (y, p) in enumerate(zip(labels, probabilities)))
    write_json(metrics_path, {"seed": args.seed, "threshold_source": "checkpoint validation split", **metrics})
    print(metrics)


if __name__ == "__main__":
    main()
