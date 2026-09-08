"""
Evaluate reconstructions against independent Gridded ARGO observations
using standard skill metrics: correlation, RMSE, Bias.

Usage sketch:

    from app import validation
    metrics = validation.evaluate(predicted_da, argo_da)
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _paired_finite(pred: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(pred) & np.isfinite(obs)
    return pred[mask], obs[mask]


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    p, o = _paired_finite(pred, obs)
    if p.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - o) ** 2)))


def bias(pred: np.ndarray, obs: np.ndarray) -> float:
    p, o = _paired_finite(pred, obs)
    if p.size == 0:
        return float("nan")
    return float(np.mean(p - o))


def correlation(pred: np.ndarray, obs: np.ndarray) -> float:
    p, o = _paired_finite(pred, obs)
    if p.size < 2 or np.std(p) == 0 or np.std(o) == 0:
        return float("nan")
    return float(np.corrcoef(p, o)[0, 1])


def evaluate(
    predicted: xr.DataArray,
    observed: xr.DataArray,
    per_depth: bool = True,
) -> dict:
    """
    Compare a predicted temperature field (time, depth, lat, lon) against
    ARGO-derived gridded observations on the same grid/time/depth axes.

    Returns overall metrics, and optionally per-depth-level metrics
    (useful for spotting where the model underperforms, e.g. near the
    thermocline).
    """
    pred_aligned, obs_aligned = xr.align(predicted, observed, join="inner")

    overall = {
        "rmse": rmse(pred_aligned.values, obs_aligned.values),
        "bias": bias(pred_aligned.values, obs_aligned.values),
        "correlation": correlation(pred_aligned.values, obs_aligned.values),
        "n_points": int(np.isfinite(pred_aligned.values).sum()),
    }

    result = {"overall": overall}

    if per_depth and "depth" in pred_aligned.dims:
        per_depth_metrics = {}
        for d in pred_aligned["depth"].values:
            p = pred_aligned.sel(depth=d).values
            o = obs_aligned.sel(depth=d).values
            per_depth_metrics[float(d)] = {
                "rmse": rmse(p, o),
                "bias": bias(p, o),
                "correlation": correlation(p, o),
                "n_points": int(np.isfinite(p).sum()),
            }
        result["per_depth"] = per_depth_metrics

    return result
