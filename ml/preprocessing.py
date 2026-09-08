"""
Preprocessing pipeline: turns the raw lazy GLORYS dataset into three
normalized NumPy sample arrays (train / val / test), each holding:

    inputs  : [N, C, PATCH_SIZE, PATCH_SIZE]   (C = len(config.INPUT_VARIABLES))
    targets : [N, N_DEPTHS]                     (temperature, degC)
    lat/lon/time metadata for traceability

How a "sample" is built
------------------------
Each timestep's surface fields (SST, SSS, ...) cover the whole region grid.
Rather than predicting a full 2D temperature-profile *field* (which the
architecture spec does not ask for - it asks for input [B,C,H,W] and
target [B,15]) we treat each sample as a small local PATCH of the surface
fields centered on one grid point, and the target is the subsurface
temperature profile AT THAT SAME GRID POINT. This lets a CNN look at the
local surface context (fronts, gradients, eddies) around a point to infer
what is happening underneath it - which is also the physically motivated
reading of "surface variables -> subsurface profile".

Everything below only materializes (.load()) small, already spatially and
temporally subset slices - never the full raw dataset.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from ml import config
from ml.data_loader import open_glorys_dataset, resolve_coords, resolve_variables, DataLoadError


# --------------------------------------------------------------------------
# Depth mapping
# --------------------------------------------------------------------------
def resolve_depth_mapping(depth_values: np.ndarray) -> List[Dict]:
    """
    For each configured TARGET_DEPTHS entry, find the nearest available
    depth level in depth_values. Never interpolates - always snaps to an
    existing model level and reports the mismatch.
    """
    depth_values = np.asarray(depth_values, dtype=float)
    mapping = []
    for target in config.TARGET_DEPTHS:
        idx = int(np.argmin(np.abs(depth_values - target)))
        chosen = float(depth_values[idx])
        mapping.append({
            "target_depth_m": target,
            "chosen_depth_m": chosen,
            "chosen_index": idx,
            "difference_m": round(chosen - target, 3),
        })
    return mapping


def print_depth_mapping(mapping: List[Dict]) -> None:
    print("\nTarget depth -> nearest available GLORYS level (no interpolation):")
    print(f"{'Target (m)':>12} | {'Chosen (m)':>12} | {'Index':>6} | {'Diff (m)':>10}")
    for row in mapping:
        print(f"{row['target_depth_m']:>12} | {row['chosen_depth_m']:>12.2f} | "
              f"{row['chosen_index']:>6} | {row['difference_m']:>10.2f}")


def save_depth_mapping(mapping: List[Dict]) -> None:
    with open(config.DEPTH_MAPPING_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nSaved depth mapping -> {config.DEPTH_MAPPING_PATH}")


def load_depth_mapping() -> List[Dict]:
    if not config.DEPTH_MAPPING_PATH.exists():
        raise DataLoadError(
            f"{config.DEPTH_MAPPING_PATH} not found. Run preprocessing first."
        )
    with open(config.DEPTH_MAPPING_PATH) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Region + variable subsetting (still lazy)
# --------------------------------------------------------------------------
def subset_region(ds: xr.Dataset, coords: Dict[str, str]) -> xr.Dataset:
    lat_name, lon_name = coords["lat"], coords["lon"]
    return ds.sel(
        {
            lat_name: slice(config.LAT_MIN, config.LAT_MAX),
            lon_name: slice(config.LON_MIN, config.LON_MAX),
        }
    )


def _mask_fill_values(arr: np.ndarray) -> np.ndarray:
    """Turn GLORYS fill values / absurd magnitudes into NaN."""
    arr = arr.astype(np.float64)
    arr[np.abs(arr) >= config.FILL_VALUE_THRESHOLD] = np.nan
    return arr


# --------------------------------------------------------------------------
# Sample extraction for one date range (train/val/test)
# --------------------------------------------------------------------------
def _select_timesteps(time_index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    mask = (time_index >= pd.Timestamp(start)) & (time_index <= pd.Timestamp(end))
    all_idx = np.where(mask)[0]
    if all_idx.size == 0:
        return all_idx
    # temporal stride subsampling
    stride = max(1, config.TEMPORAL_STRIDE_DAYS)
    return all_idx[::stride]


def _extract_samples_for_split(
    ds: xr.Dataset,
    coords: Dict[str, str],
    variables: Dict[str, str],
    depth_mapping: List[Dict],
    start: str,
    end: str,
    max_samples: int,
    split_name: str,
) -> Dict[str, np.ndarray]:
    lat_name, lon_name, time_name, depth_name = (
        coords["lat"], coords["lon"], coords["time"], coords["depth"]
    )
    temp_var = variables["sst"]  # 3D temperature field, e.g. "thetao"
    salt_var = variables["sss"]  # 3D salinity field, e.g. "so"

    time_index = pd.to_datetime(ds[time_name].values)
    ts_indices = _select_timesteps(time_index, start, end)

    if ts_indices.size == 0:
        print(f"[{split_name}] WARNING: no timesteps found in range {start}..{end}")
        return {
            "inputs": np.zeros((0, len(config.INPUT_VARIABLES),
                                 config.PATCH_SIZE, config.PATCH_SIZE), dtype=np.float32),
            "targets": np.zeros((0, config.N_DEPTHS), dtype=np.float32),
            "times": np.array([], dtype="datetime64[ns]"),
            "lats": np.array([], dtype=np.float32),
            "lons": np.array([], dtype=np.float32),
        }

    depth_indices = [row["chosen_index"] for row in depth_mapping]
    surface_depth_index = 0  # shallowest level = SST/SSS proxy

    half = config.PATCH_SIZE // 2
    n_lat = ds.sizes[lat_name]
    n_lon = ds.sizes[lon_name]

    candidate_lat_idx = np.arange(half, n_lat - half, config.SPATIAL_STRIDE)
    candidate_lon_idx = np.arange(half, n_lon - half, config.SPATIAL_STRIDE)

    if candidate_lat_idx.size == 0 or candidate_lon_idx.size == 0:
        raise DataLoadError(
            f"Region too small for PATCH_SIZE={config.PATCH_SIZE}. "
            f"Reduce config.PATCH_SIZE or widen the region."
        )

    lat_vals_full = ds[lat_name].values
    lon_vals_full = ds[lon_name].values

    inputs_list, targets_list = [], []
    times_list, lats_list, lons_list = [], [], []

    rng = np.random.default_rng(config.SAMPLING_SEED)

    for t_idx in ts_indices:
        # Materialize only this single timestep's already-region-subset
        # surface + profile slices (small: region x few depths).
        sst_2d = _mask_fill_values(
            ds[temp_var].isel({time_name: t_idx, depth_name: surface_depth_index}).values
        )
        sss_2d = _mask_fill_values(
            ds[salt_var].isel({time_name: t_idx, depth_name: surface_depth_index}).values
        )
        profile_3d = _mask_fill_values(
            ds[temp_var].isel({time_name: t_idx, depth_name: depth_indices}).values
        )  # shape [n_depths, lat, lon]

        surface_channels = {"sst": sst_2d, "sss": sss_2d}

        for i in candidate_lat_idx:
            for j in candidate_lon_idx:
                target_profile = profile_3d[:, i, j]
                if np.isnan(target_profile).any():
                    continue  # land / missing profile at this point

                patch_channels = []
                valid = True
                for var_name in config.INPUT_VARIABLES:
                    field = surface_channels[var_name]
                    patch = field[i - half:i + half + 1, j - half:j + half + 1]
                    if patch.shape != (config.PATCH_SIZE, config.PATCH_SIZE) or \
                            np.isnan(patch).any():
                        valid = False
                        break
                    patch_channels.append(patch)
                if not valid:
                    continue

                inputs_list.append(np.stack(patch_channels, axis=0))
                targets_list.append(target_profile)
                times_list.append(time_index[t_idx])
                lats_list.append(lat_vals_full[i])
                lons_list.append(lon_vals_full[j])

    n_found = len(inputs_list)
    print(f"[{split_name}] {n_found} candidate samples extracted "
          f"from {ts_indices.size} timesteps.")

    if n_found == 0:
        raise DataLoadError(
            f"No valid samples found for split '{split_name}' in range "
            f"{start}..{end}. Check region bounds / land mask / date range."
        )

    order = rng.permutation(n_found)
    keep = order[:max_samples]

    inputs = np.stack(inputs_list, axis=0)[keep].astype(np.float32)
    targets = np.stack(targets_list, axis=0)[keep].astype(np.float32)
    times = np.array(times_list, dtype="datetime64[ns]")[keep]
    lats = np.array(lats_list, dtype=np.float32)[keep]
    lons = np.array(lons_list, dtype=np.float32)[keep]

    print(f"[{split_name}] kept {inputs.shape[0]} samples "
          f"(cap={max_samples}). inputs={inputs.shape}, targets={targets.shape}")

    return {"inputs": inputs, "targets": targets, "times": times,
            "lats": lats, "lons": lons}


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
def compute_and_save_norm_stats(train_inputs: np.ndarray, train_targets: np.ndarray) -> Dict:
    """
    Compute per-channel input stats and per-depth target stats using ONLY
    the training split, and persist them so predict.py can reuse the exact
    same normalization at inference time.
    """
    stats = {"input_variables": config.INPUT_VARIABLES, "target_depths": config.TARGET_DEPTHS}

    input_mean = train_inputs.mean(axis=(0, 2, 3))  # per channel
    input_std = train_inputs.std(axis=(0, 2, 3))
    input_std[input_std < 1e-6] = 1e-6
    stats["input_mean"] = input_mean.tolist()
    stats["input_std"] = input_std.tolist()

    target_mean = train_targets.mean(axis=0)  # per depth
    target_std = train_targets.std(axis=0)
    target_std[target_std < 1e-6] = 1e-6
    stats["target_mean"] = target_mean.tolist()
    stats["target_std"] = target_std.tolist()

    with open(config.NORM_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved normalization statistics (train-only) -> {config.NORM_STATS_PATH}")
    return stats


def load_norm_stats() -> Dict:
    if not config.NORM_STATS_PATH.exists():
        raise DataLoadError(
            f"{config.NORM_STATS_PATH} not found. Run preprocessing first."
        )
    with open(config.NORM_STATS_PATH) as f:
        return json.load(f)


def apply_normalization(inputs: np.ndarray, targets: np.ndarray, stats: Dict) -> Tuple[np.ndarray, np.ndarray]:
    mean_in = np.array(stats["input_mean"], dtype=np.float32).reshape(1, -1, 1, 1)
    std_in = np.array(stats["input_std"], dtype=np.float32).reshape(1, -1, 1, 1)
    mean_t = np.array(stats["target_mean"], dtype=np.float32).reshape(1, -1)
    std_t = np.array(stats["target_std"], dtype=np.float32).reshape(1, -1)

    norm_inputs = (inputs - mean_in) / std_in
    norm_targets = (targets - mean_t) / std_t
    return norm_inputs.astype(np.float32), norm_targets.astype(np.float32)


def denormalize_targets(norm_targets: np.ndarray, stats: Dict) -> np.ndarray:
    mean_t = np.array(stats["target_mean"], dtype=np.float32)
    std_t = np.array(stats["target_std"], dtype=np.float32)
    return norm_targets * std_t + mean_t


def normalize_inputs_only(inputs: np.ndarray, stats: Dict) -> np.ndarray:
    mean_in = np.array(stats["input_mean"], dtype=np.float32).reshape(1, -1, 1, 1)
    std_in = np.array(stats["input_std"], dtype=np.float32).reshape(1, -1, 1, 1)
    return ((inputs - mean_in) / std_in).astype(np.float32)


# --------------------------------------------------------------------------
# Save / load sample arrays
# --------------------------------------------------------------------------
def _save_split(path, data: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        path,
        inputs=data["inputs"],
        targets=data["targets"],
        times=data["times"].astype("datetime64[ns]").astype(np.int64),
        lats=data["lats"],
        lons=data["lons"],
    )
    print(f"Saved {data['inputs'].shape[0]} samples -> {path}")


def load_split(path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise DataLoadError(f"{path} not found. Run preprocessing first.")
    npz = np.load(path)
    return {
        "inputs": npz["inputs"],
        "targets": npz["targets"],
        "times": npz["times"].astype("datetime64[ns]"),
        "lats": npz["lats"],
        "lons": npz["lons"],
    }


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def run_preprocessing() -> None:
    print("=" * 70)
    print("OceanEmbed - Preprocessing")
    print("=" * 70)

    ds = open_glorys_dataset()
    coords = resolve_coords(ds)
    variables = resolve_variables(ds)

    if "sst" not in variables or "sss" not in variables:
        raise DataLoadError(
            "Could not resolve both temperature and salinity 3D variables. "
            f"Resolved so far: {variables}. Run inspect first (python -m ml.data_loader)."
        )
    if "depth" not in coords:
        raise DataLoadError("No depth coordinate resolved - cannot build profile targets.")

    ds = subset_region(ds, coords)
    print(f"\nRegion subset -> lat[{config.LAT_MIN},{config.LAT_MAX}], "
          f"lon[{config.LON_MIN},{config.LON_MAX}]  "
          f"grid size: {ds.sizes[coords['lat']]} x {ds.sizes[coords['lon']]}")

    depth_values = ds[coords["depth"]].values
    depth_mapping = resolve_depth_mapping(depth_values)
    print_depth_mapping(depth_mapping)
    save_depth_mapping(depth_mapping)

    splits_cfg = [
        ("train", config.TRAIN_START, config.TRAIN_END, config.MAX_TRAIN_SAMPLES, config.TRAIN_SAMPLES_PATH),
        ("val", config.VAL_START, config.VAL_END, config.MAX_VAL_SAMPLES, config.VAL_SAMPLES_PATH),
        ("test", config.TEST_START, config.TEST_END, config.MAX_TEST_SAMPLES, config.TEST_SAMPLES_PATH),
    ]

    raw_splits = {}
    for name, start, end, max_samples, _path in splits_cfg:
        print(f"\n--- Extracting '{name}' split: {start} .. {end} ---")
        raw_splits[name] = _extract_samples_for_split(
            ds, coords, variables, depth_mapping, start, end, max_samples, name
        )

    # Normalization stats from TRAIN ONLY.
    stats = compute_and_save_norm_stats(raw_splits["train"]["inputs"], raw_splits["train"]["targets"])

    for (name, _start, _end, _max, path) in splits_cfg:
        raw = raw_splits[name]
        norm_inputs, norm_targets = apply_normalization(raw["inputs"], raw["targets"], stats)
        _save_split(path, {
            "inputs": norm_inputs, "targets": norm_targets,
            "times": raw["times"], "lats": raw["lats"], "lons": raw["lons"],
        })

    print("\nPreprocessing complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_preprocessing()
