"""
Evaluation script: runs the trained model on the held-out TEST split and
computes, for each of the 15 target depths: RMSE, bias (mean predicted -
mean actual), and Pearson correlation. Also reports overall (all-depth)
metrics. Metrics are computed in real-world degC units (denormalized).

Usage:
    python -m ml.evaluate
"""

from __future__ import annotations

import csv
import json

import numpy as np
import torch

from ml import config
from ml.dataset import get_dataloader
from ml.model import build_model
from ml.preprocessing import load_norm_stats, denormalize_targets
from ml.train import get_device


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, robust to zero-variance / NaN edge cases."""
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size < 2 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate() -> None:
    print("=" * 70)
    print("OceanEmbed - Evaluation (test split)")
    print("=" * 70)

    device = get_device()
    checkpoint_path = config.BEST_CHECKPOINT_PATH
    if not checkpoint_path.exists():
        checkpoint_path = config.FINAL_CHECKPOINT_PATH
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "No checkpoint found. Run training first (python -m ml.train)."
        )
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model, checkpoint = load_checkpoint(checkpoint_path, device)

    stats = load_norm_stats()
    test_loader = get_dataloader("test", shuffle=False)
    print(f"Test samples: {len(test_loader.dataset)}")

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())

    preds_norm = np.concatenate(all_preds, axis=0)      # [N, 15], normalized
    targets_norm = np.concatenate(all_targets, axis=0)  # [N, 15], normalized

    preds_c = denormalize_targets(preds_norm, stats)      # degC
    targets_c = denormalize_targets(targets_norm, stats)  # degC

    depths = config.TARGET_DEPTHS
    rows = []
    print(f"\n{'Depth (m)':>10} | {'RMSE (degC)':>12} | {'Bias (degC)':>12} | {'Correlation':>12}")
    for d_idx, depth in enumerate(depths):
        pred_d = preds_c[:, d_idx]
        true_d = targets_c[:, d_idx]
        mask = ~(np.isnan(pred_d) | np.isnan(true_d))
        pred_d_valid = pred_d[mask]
        true_d_valid = true_d[mask]

        if pred_d_valid.size == 0:
            rmse = bias = corr = float("nan")
        else:
            rmse = float(np.sqrt(np.mean((pred_d_valid - true_d_valid) ** 2)))
            bias = float(np.mean(pred_d_valid - true_d_valid))
            corr = _safe_corrcoef(pred_d_valid, true_d_valid)

        rows.append({"depth_m": depth, "rmse_degC": rmse, "bias_degC": bias,
                      "correlation": corr, "n_valid": int(pred_d_valid.size)})
        print(f"{depth:>10} | {rmse:>12.4f} | {bias:>12.4f} | {corr:>12.4f}")

    valid_mask_all = ~(np.isnan(preds_c) | np.isnan(targets_c))
    overall_rmse = float(np.sqrt(np.mean(
        (preds_c[valid_mask_all] - targets_c[valid_mask_all]) ** 2
    )))
    overall_bias = float(np.mean(preds_c[valid_mask_all] - targets_c[valid_mask_all]))
    overall_corr = _safe_corrcoef(preds_c[valid_mask_all], targets_c[valid_mask_all])

    print(f"\n{'OVERALL':>10} | {overall_rmse:>12.4f} | {overall_bias:>12.4f} | {overall_corr:>12.4f}")

    summary = {
        "checkpoint": str(checkpoint_path),
        "n_test_samples": int(preds_c.shape[0]),
        "per_depth": rows,
        "overall": {"rmse_degC": overall_rmse, "bias_degC": overall_bias,
                    "correlation": overall_corr},
    }

    json_path = config.METRICS_DIR / "test_metrics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics JSON -> {json_path}")

    csv_path = config.METRICS_DIR / "test_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["depth_m", "rmse_degC", "bias_degC",
                                                 "correlation", "n_valid"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved metrics CSV -> {csv_path}")

    print("\nEvaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
