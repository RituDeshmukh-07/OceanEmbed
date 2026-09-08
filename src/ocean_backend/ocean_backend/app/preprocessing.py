"""
Task 4: Harmonize data — spatial/temporal alignment, normalization, and
conversion to the tensors the model expects.

Pipeline:
  1. regrid_to_common_grid: put every variable on the same 0.25deg lat/lon
     grid (bilinear interpolation), regardless of its native resolution.
  2. align_in_time: reindex every variable onto the same daily time axis,
     with optional forward/back-fill for short gaps (e.g. SSH revisit gaps).
  3. fit_normalization / apply_normalization: per-variable z-score
     normalization, stats saved to disk so training and inference use the
     identical transform.
  4. stack_to_tensor: combine the available variables into a single
     (time, channel, lat, lon) numpy array in the model's expected
     channel order, skipping channels you don't have yet (currents/winds)
     by zero-filling — Ritu's model should be trained to accept a
     variable-length or masked channel set, or you retrain once those
     variables are available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from . import config


def build_common_grid() -> tuple[np.ndarray, np.ndarray]:
    """The canonical 0.25deg lat/lon grid for the North Indian Ocean box."""
    lats = np.arange(config.LAT_MIN, config.LAT_MAX + 1e-9,
                      config.GRID_RESOLUTION_DEG)
    lons = np.arange(config.LON_MIN, config.LON_MAX + 1e-9,
                      config.GRID_RESOLUTION_DEG)
    return lats, lons


def regrid_to_common_grid(da: xr.DataArray) -> xr.DataArray:
    """Bilinear interpolation onto the canonical 0.25deg grid."""
    lats, lons = build_common_grid()
    return da.interp(lat=lats, lon=lons, method="linear")


def align_in_time(
    variables: dict[str, xr.DataArray],
    max_gap_days: int = 3,
) -> dict[str, xr.DataArray]:
    """
    Reindex every variable onto the same daily time axis (the union of
    all available days, clipped to the overlapping range), filling short
    gaps by linear interpolation in time and leaving longer gaps as NaN
    (so QC masks stay honest).
    """
    starts = [da["time"].values.min() for da in variables.values()]
    ends = [da["time"].values.max() for da in variables.values()]
    common_start, common_end = max(starts), min(ends)

    daily_index = np.arange(common_start, common_end + np.timedelta64(1, "D"),
                             np.timedelta64(1, "D"))

    aligned = {}
    for key, da in variables.items():
        da = da.reindex(time=daily_index)
        da = da.interpolate_na(dim="time", method="linear",
                                limit=max_gap_days)
        aligned[key] = da
    return aligned


def fit_normalization(variables: dict[str, xr.DataArray]) -> dict:
    """Compute per-variable mean/std and persist to config.NORM_STATS_PATH."""
    stats = {}
    for key, da in variables.items():
        vals = da.values
        stats[key] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)) or 1.0,
        }
    config.NORM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.NORM_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def load_normalization() -> dict:
    if not config.NORM_STATS_PATH.exists():
        raise FileNotFoundError(
            f"No normalization stats at {config.NORM_STATS_PATH}. "
            f"Run fit_normalization() on the training set first."
        )
    with open(config.NORM_STATS_PATH) as f:
        return json.load(f)


def apply_normalization(
    variables: dict[str, xr.DataArray],
    stats: Optional[dict] = None,
) -> dict[str, xr.DataArray]:
    stats = stats or load_normalization()
    out = {}
    for key, da in variables.items():
        if key not in stats:
            out[key] = da  # leave un-normalized if no stats (shouldn't happen)
            continue
        out[key] = (da - stats[key]["mean"]) / stats[key]["std"]
    return out


def stack_to_tensor(
    variables: dict[str, xr.DataArray],
    channel_order: list[str] = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Stack variables into a (time, channel, lat, lon) float32 array in the
    model's expected channel order. Missing optional channels (e.g. you
    only have sst/sss/ssh right now) are filled with zeros so the tensor
    shape stays fixed; the model should be told which channels are
    real vs. zero-filled via the returned `present_channels` list if it
    needs that at inference time.
    """
    channel_order = channel_order or config.MODEL_INPUT_ORDER
    ref = next(iter(variables.values()))
    shape = (ref.sizes["time"], len(channel_order),
             ref.sizes["lat"], ref.sizes["lon"])
    tensor = np.zeros(shape, dtype="float32")

    present_channels = []
    for i, key in enumerate(channel_order):
        if key in variables:
            tensor[:, i, :, :] = np.nan_to_num(
                variables[key].values.astype("float32"), nan=0.0
            )
            present_channels.append(key)

    return tensor, present_channels


def harmonize(
    variables: dict[str, xr.DataArray],
    fit_norm: bool = False,
) -> tuple[np.ndarray, list[str], xr.DataArray]:
    """
    Convenience wrapper running the full harmonization pipeline:
    regrid -> align in time -> normalize -> stack to tensor.

    Returns (tensor, present_channels, reference_time_lat_lon_template)
    where the template DataArray is useful for re-wrapping model output
    with correct coordinates.
    """
    regridded = {k: regrid_to_common_grid(v) for k, v in variables.items()}
    aligned = align_in_time(regridded)

    if fit_norm:
        stats = fit_normalization(aligned)
    else:
        stats = load_normalization()

    normalized = apply_normalization(aligned, stats)
    tensor, present_channels = stack_to_tensor(normalized)

    template = next(iter(aligned.values()))
    return tensor, present_channels, template
