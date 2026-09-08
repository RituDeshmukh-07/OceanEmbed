"""
NetCDF discovery + lazy loading for GLORYS12V1 data.

Design goals:
  * Never assume exact filenames.
  * Never assume exact variable/coordinate names - resolve them from a
    candidate list (see config.VARIABLE_CANDIDATES / COORD_CANDIDATES).
  * Never load the full dataset into RAM - everything here returns lazy
    (dask-backed) xarray objects. Materialization (.load()/.values) only
    happens in preprocessing.py, and only on an already spatially/temporally
    subset slice.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from ml import config


class DataLoadError(RuntimeError):
    """Raised when GLORYS NetCDF files cannot be found or understood."""


def discover_nc_files(raw_dir: Path = config.RAW_DATA_DIR) -> List[Path]:
    """Find all NetCDF files under raw_dir (non-recursive union of patterns)."""
    raw_dir = Path(raw_dir)
    files: List[Path] = []
    for pattern in config.NC_GLOB_PATTERNS:
        files.extend(sorted(raw_dir.glob(pattern)))
        files.extend(sorted(raw_dir.glob(f"**/{pattern}")))
    # de-duplicate while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            unique_files.append(f)

    if not unique_files:
        raise DataLoadError(
            f"No .nc/.nc4 files found under {raw_dir}. "
            f"Place your GLORYS12V1 NetCDF files there first."
        )
    return unique_files


def _resolve_name(ds: xr.Dataset, candidates: List[str]) -> Optional[str]:
    """Return the first candidate name present in ds.variables/coords, else None."""
    available = set(ds.variables.keys())
    for name in candidates:
        if name in available:
            return name
    # case-insensitive fallback
    lower_map = {v.lower(): v for v in available}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def resolve_coords(ds: xr.Dataset) -> Dict[str, str]:
    """Resolve lat/lon/time/depth coordinate names actually present in ds."""
    resolved = {}
    for logical, candidates in config.COORD_CANDIDATES.items():
        name = _resolve_name(ds, candidates)
        if name is not None:
            resolved[logical] = name
    missing = {"lat", "lon", "time"} - set(resolved.keys())
    if missing:
        raise DataLoadError(
            f"Could not resolve required coordinate(s) {missing} in dataset. "
            f"Available variables/coords: {sorted(ds.variables.keys())}"
        )
    return resolved


def resolve_variables(ds: xr.Dataset) -> Dict[str, str]:
    """Resolve the physical variable names (temperature, salinity, ...) present."""
    resolved = {}
    for logical, candidates in config.VARIABLE_CANDIDATES.items():
        name = _resolve_name(ds, candidates)
        if name is not None:
            resolved[logical] = name
    return resolved


def open_glorys_dataset(raw_dir: Path = config.RAW_DATA_DIR) -> xr.Dataset:
    """
    Lazily open every NetCDF file under raw_dir as a single combined,
    dask-backed xarray Dataset (chunked, not loaded into memory).
    """
    files = discover_nc_files(raw_dir)
    try:
        ds = xr.open_mfdataset(
            [str(f) for f in files],
            combine="by_coords",
            chunks={},  # let xarray/dask pick chunk sizes from the files
            decode_times=True,
            mask_and_scale=True,
        )
    except Exception as exc:  # noqa: BLE001 - broad on purpose, then fall back
        # Fallback: some GLORYS distributions are split with inconsistent
        # coordinate metadata that breaks "by_coords" combination. Try a
        # more permissive nested-by-time combination as a robust fallback.
        try:
            ds = xr.open_mfdataset(
                [str(f) for f in files],
                combine="nested",
                concat_dim="time",
                chunks={},
                decode_times=True,
                mask_and_scale=True,
            )
        except Exception as exc2:  # noqa: BLE001
            raise DataLoadError(
                f"Failed to open NetCDF files in {raw_dir}.\n"
                f"by_coords error: {exc}\nnested fallback error: {exc2}"
            ) from exc2

    coords = resolve_coords(ds)

    # Standardize latitude to ascending order (some GLORYS extracts are
    # descending) so slicing with lat_min:lat_max behaves predictably.
    lat_name = coords["lat"]
    if ds[lat_name].size > 1 and float(ds[lat_name][0]) > float(ds[lat_name][-1]):
        ds = ds.sortby(lat_name)

    # Standardize longitude convention to [-180, 180) if the file uses
    # [0, 360). Our region (60E-80E) is representable in both conventions,
    # but we normalize so config bounds always mean the same thing.
    lon_name = coords["lon"]
    lon_vals = ds[lon_name].values
    if np.nanmax(lon_vals) > 180.0:
        ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)})
        ds = ds.sortby(lon_name)

    return ds


def inspect_dataset(raw_dir: Path = config.RAW_DATA_DIR) -> None:
    """Print a human-readable summary of the discovered GLORYS data."""
    files = discover_nc_files(raw_dir)
    print("=" * 70)
    print("OceanEmbed - Dataset Inspection")
    print("=" * 70)
    print(f"\nFound {len(files)} NetCDF file(s) under {raw_dir}:")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name}  ({size_mb:.1f} MB)")

    ds = open_glorys_dataset(raw_dir)
    coords = resolve_coords(ds)
    variables = resolve_variables(ds)

    print(f"\nDimensions: {dict(ds.sizes)}")
    print(f"\nAll variables in files: {sorted(ds.data_vars.keys())}")
    print(f"\nResolved coordinates -> actual names: {coords}")
    print(f"Resolved physical variables -> actual names: {variables}")

    lat_name, lon_name, time_name = coords["lat"], coords["lon"], coords["time"]
    print(f"\nLatitude range:  {float(ds[lat_name].min()):.3f} to "
          f"{float(ds[lat_name].max()):.3f}")
    print(f"Longitude range: {float(ds[lon_name].min()):.3f} to "
          f"{float(ds[lon_name].max()):.3f}")

    time_vals = ds[time_name].values
    print(f"\nTime range: {np.datetime_as_string(time_vals.min(), unit='D')} "
          f"to {np.datetime_as_string(time_vals.max(), unit='D')}")
    print(f"Number of timesteps: {ds.sizes[time_name]}")

    if "depth" in coords:
        depth_name = coords["depth"]
        depth_vals = ds[depth_name].values
        print(f"\nDepth levels ({len(depth_vals)} total): "
              f"{np.array2string(depth_vals, precision=2)}")
    else:
        print("\nNo depth coordinate found (unexpected for GLORYS 3D fields).")

    print("\nUnits:")
    for logical, actual in variables.items():
        units = ds[actual].attrs.get("units", "unknown")
        print(f"  - {logical} ({actual}): {units}")

    missing_vars = set(config.TEMPERATURE_VAR_CANDIDATES) - {variables.get("sst")}
    if "sst" not in variables:
        print(
            "\nWARNING: no temperature-like 3D variable resolved from "
            f"candidates {config.TEMPERATURE_VAR_CANDIDATES}. Preprocessing "
            "will fail until this is fixed (check variable names above)."
        )
    if "sss" not in variables:
        print(
            "\nWARNING: no salinity-like variable resolved from candidates "
            f"{config.SALINITY_VAR_CANDIDATES}."
        )
    print("=" * 70)


if __name__ == "__main__":
    inspect_dataset()
