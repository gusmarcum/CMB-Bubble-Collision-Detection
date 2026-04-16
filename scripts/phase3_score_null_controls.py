"""
Score real-SMICA null-control patches with a trained Phase 3 model.

This checks the frozen screener operating point against real cleaned-map
backgrounds. It is not a discovery run; it is a nuisance/systematics control
for false candidate emission on signal-free SMICA patches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import phase3_train_unet as p3
from phase_dataset_utils import build_patch_candidate, load_optional_metadata_array
from phase3_evaluate_run import resolve_checkpoint_path


MASK_SHAPE = (256, 256)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained screener on real-SMICA null-control patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--null-h5", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--split", type=str, default="val", choices=["all", "train", "val"])
    parser.add_argument("--ml-eval-dir", type=str, default="")
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--threshold-min", type=float, default=0.50)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-count", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def validate_args(args):
    if args.threshold_count <= 1:
        raise ValueError("--threshold-count must be greater than 1.")
    if not (0.0 <= args.target_fpr < 1.0):
        raise ValueError("--target-fpr must be in [0, 1).")
    if args.threshold >= 0.0 and not (0.0 < args.threshold < 1.0):
        raise ValueError("--threshold must be in (0, 1) when provided.")
    if not (0.0 < args.threshold_min < args.threshold_max < 1.0):
        raise ValueError("--threshold-min/max must satisfy 0 < min < max < 1.")
    if args.batch_size < 0:
        raise ValueError("--batch-size must be non-negative.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_frozen_threshold(args, run_config):
    if args.threshold >= 0.0:
        return float(args.threshold), "cli_threshold"
    if args.ml_eval_dir:
        summary = load_json(Path(args.ml_eval_dir) / "evaluation_summary.json")
        return float(summary["selected_threshold"]), "ml_eval_selected_threshold"
    return float(run_config["args"]["threshold"]), "training_config_threshold"


def load_null_split_indices(null_h5, split):
    with h5py.File(null_h5, "r") as h5:
        n = int(h5["patches"].shape[0])
        if split == "all":
            return np.arange(n, dtype=np.int64)
        if "splits" not in h5 or f"{split}_idx" not in h5["splits"]:
            raise KeyError(f"Null-control HDF5 missing splits/{split}_idx.")
        return np.asarray(h5["splits"][f"{split}_idx"][:], dtype=np.int64)


def make_loader(null_h5, indices, run_config, args, device):
    batch_size = args.batch_size or int(run_config["args"]["batch_size"])
    dataset = p3.H5BubbleDataset(
        h5_path=null_h5,
        indices=indices,
        **p3.dataset_kwargs_from_run_config(run_config),
        augment=False,
        seed=int(run_config["args"]["seed"]) + 997,
        max_translate_pixels=0,
    )
    return p3.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def score_thresholds(model, loader, thresholds, device):
    false_positive_counts = np.zeros(len(thresholds), dtype=np.int64)
    positive_pixel_sums = np.zeros(len(thresholds), dtype=np.float64)
    num_samples = 0
    progress = p3.ProgressPrinter(len(loader), "Null threshold sweep")
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            logits, _ = p3.unpack_model_output(model(images))
            probs = torch.sigmoid(logits)
            num_samples += int(images.shape[0])
            for idx, threshold in enumerate(thresholds):
                pred = probs >= float(threshold)
                false_positive_counts[idx] += int(pred.flatten(1).any(dim=1).sum().item())
                positive_pixel_sums[idx] += float(pred.float().mean(dim=(1, 2, 3)).sum().item())
            progress.update(batch_idx)

    rows = []
    for threshold, fp_count, pixel_sum in zip(thresholds, false_positive_counts, positive_pixel_sums):
        rows.append(
            {
                "threshold": float(threshold),
                "num_samples": int(num_samples),
                "false_positive_count": int(fp_count),
                "false_positive_rate": float(fp_count / max(num_samples, 1)),
                "mean_positive_fraction": float(pixel_sum / max(num_samples, 1)),
            }
        )
    return rows


def choose_null_threshold(rows, target_fpr):
    feasible = [row for row in rows if row["false_positive_rate"] <= target_fpr + 1e-12]
    if not feasible:
        return min(rows, key=lambda row: (row["false_positive_rate"], -row["threshold"])), "fallback_min_fpr"
    return min(feasible, key=lambda row: row["threshold"]), "min_threshold_under_null_fpr_cap"


def closest_row(rows, threshold):
    return min(rows, key=lambda row: abs(row["threshold"] - threshold))


def collect_null_candidates(model, loader, device, threshold, null_h5):
    glon = load_optional_metadata_array(null_h5, "glon_deg", dtype=np.float32, default_value=np.nan)
    glat = load_optional_metadata_array(null_h5, "glat_deg", dtype=np.float32, default_value=np.nan)
    coord_pool_idx = load_optional_metadata_array(null_h5, "coord_pool_idx", dtype=np.int32, default_value=-1)
    coord_mask_fraction = load_optional_metadata_array(
        null_h5,
        "coord_mask_fraction",
        dtype=np.float32,
        default_value=np.nan,
    )
    background_id = load_optional_metadata_array(null_h5, "background_id", dtype=np.uint64, default_value=0)

    records = []
    mask_bits = []
    sample_indices = []
    progress = p3.ProgressPrinter(len(loader), "Collect null candidates")
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            logits, _ = p3.unpack_model_output(model(images))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            indices = np.asarray(batch["index"][:], dtype=np.int64)
            for row, sample_idx in enumerate(indices):
                candidate, candidate_mask = build_patch_candidate(
                    prob_map=probs[row, 0],
                    threshold=float(threshold),
                    patch_center_glon_deg=float(glon[sample_idx]),
                    patch_center_glat_deg=float(glat[sample_idx]),
                    sample_index=int(sample_idx),
                )
                candidate.update(
                    {
                        "truth_label": 0,
                        "source_kind": "real_smica_null_control",
                        "coord_pool_idx": int(coord_pool_idx[sample_idx]),
                        "coord_mask_fraction": float(coord_mask_fraction[sample_idx]),
                        "background_id": int(background_id[sample_idx]),
                    }
                )
                records.append(candidate)
                sample_indices.append(int(sample_idx))
                mask_bits.append(np.packbits(candidate_mask.reshape(-1)).astype(np.uint8))
            progress.update(batch_idx)

    return {
        "records": records,
        "sample_indices": np.asarray(sample_indices, dtype=np.int64),
        "mask_bits": np.stack(mask_bits, axis=0) if mask_bits else np.zeros((0, 8192), dtype=np.uint8),
        "mask_shape": np.asarray(MASK_SHAPE, dtype=np.int32),
    }


def save_candidates(output_dir, bundle, threshold):
    records_path = output_dir / "null_candidate_records.jsonl"
    masks_path = output_dir / "null_candidate_masks.npz"
    with open(records_path, "w", encoding="utf-8") as handle:
        for row_id, record in enumerate(bundle["records"]):
            payload = dict(record)
            payload["threshold"] = float(threshold)
            payload["mask_row"] = row_id
            handle.write(json.dumps(payload) + "\n")
    np.savez_compressed(
        masks_path,
        sample_indices=bundle["sample_indices"],
        mask_bits=bundle["mask_bits"],
        mask_shape=bundle["mask_shape"],
    )
    return records_path, masks_path


def main():
    args = parse_args()
    validate_args(args)
    p3.require_ml_packages()
    p3.seed_everything(42)

    run_dir = Path(args.run_dir).resolve()
    null_h5 = Path(args.null_h5).resolve()
    if not null_h5.exists():
        raise FileNotFoundError(f"Null-control HDF5 not found: {null_h5}")
    checkpoint_path, checkpoint_label = resolve_checkpoint_path(run_dir, args.checkpoint)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / f"null_controls_{checkpoint_label}_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = load_json(run_dir / "run_config.json")
    frozen_threshold, threshold_source = resolve_frozen_threshold(args, run_config)
    indices = load_null_split_indices(null_h5, args.split)
    device = p3.resolve_device(args.device)
    loader = make_loader(null_h5, indices, run_config, args, device)

    model_args = p3.model_args_from_run_config(run_config)
    model = p3.build_model(model_args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    p3.load_model_state_dict(model, checkpoint["model_state_dict"])

    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_count, dtype=np.float64)
    rows = score_thresholds(model, loader, thresholds, device)
    frozen_row = closest_row(rows, frozen_threshold)
    null_threshold_row, null_threshold_rule = choose_null_threshold(rows, args.target_fpr)

    artifact_loader = make_loader(null_h5, indices, run_config, args, device)
    bundle = collect_null_candidates(model, artifact_loader, device, frozen_threshold, null_h5)
    records_path, masks_path = save_candidates(output_dir, bundle, frozen_threshold)

    with open(output_dir / "null_threshold_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    summary = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_label": checkpoint_label,
        "null_h5": str(null_h5),
        "split": args.split,
        "num_samples": int(len(indices)),
        "target_fpr": float(args.target_fpr),
        "frozen_threshold": float(frozen_threshold),
        "threshold_source": threshold_source,
        "frozen_threshold_nearest_grid_metrics": frozen_row,
        "null_calibrated_threshold": float(null_threshold_row["threshold"]),
        "null_calibrated_threshold_rule": null_threshold_rule,
        "null_calibrated_threshold_metrics": null_threshold_row,
        "artifacts": {
            "null_threshold_metrics_json": str((output_dir / "null_threshold_metrics.json").resolve()),
            "null_candidate_records_jsonl": str(records_path.resolve()),
            "null_candidate_masks_npz": str(masks_path.resolve()),
        },
    }
    with open(output_dir / "null_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Null summary: {output_dir / 'null_summary.json'}")


if __name__ == "__main__":
    main()
