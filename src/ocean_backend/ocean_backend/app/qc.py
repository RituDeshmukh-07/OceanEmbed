"""
Task 3: Basic QC — missing/invalid values, masking, consistent coordinate
ordering.

Applied to every surface variable right after loading and before
harmonization/regridding, so downstream code can always assume:
  - lat ascending, lon ascending
  - time sorted, no duplicate timestamps
  - physically invalid values replaced with NaN
  - a boolean `valid_mask` attached for optional loss-masking during
    model training
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from . import config


def enforce_coordinate_order(da: xr.DataArray) -> xr.DataArray:
    """Sort dims to a canonical ascending order and drop duplicate times."""
    if "time" in da.dims:
        da = da.sortby("time")
        _, unique_idx = np.unique(da["time"].values, return_index=True)
        if len(unique_idx) != da.sizes["time"]:
            da = da.isel(time=sorted(unique_idx))
    if "lat" in da.dims:
        da = da.sortby("lat")
    if "lon" in da.dims:
        da = da.sortby("lon")
    return da


def mask_invalid_values(da: xr.DataArray, var_key: str) -> xr.DataArray:
    """
    Replace physically implausible values with NaN, using the valid_range
    configured for this variable. Also strips common sentinel fill values
    (e.g. -999, 1e20) that sometimes survive automatic decoding.
    """
    spec = config.SURFACE_VARIABLES.get(var_key) or config.TARGET_VARIABLE
    vmin, vmax = spec["valid_range"]

    cleaned = da.where((da >= vmin) & (da <= vmax))

    # common sentinel fill values
    for sentinel in (-999.0, -9999.0, 1e20, 1e37):
        cleaned = cleaned.where(np.abs(cleaned - sentinel) > 1e-3)

    return cleaned


def normalize_temperature_units(da: xr.DataArray) -> xr.DataArray:
    """
    Some SST products deliver Kelvin. Heuristic: if the median value is
    > 100, assume Kelvin and convert to Celsius.
    """
    vals = da.compute().values if hasattr(da.data, "compute") else da.values
    med = float(np.nanmedian(vals))
    if med > 100:
        da = da - 273.15
        da.attrs["units"] = "degC"
    return da


def build_valid_mask(da: xr.DataArray) -> xr.DataArray:
    """Boolean mask: True where data is finite/valid, for use as a loss mask."""
    mask = xr.apply_ufunc(np.isfinite, da, dask="parallelized",
                           output_dtypes=[bool])
    mask.name = f"{da.name}_valid_mask"
    return mask


def run_qc(da: xr.DataArray, var_key: str) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Full QC pipeline for one variable. Returns (cleaned_data, valid_mask).
    """
    da = enforce_coordinate_order(da)
    da = mask_invalid_values(da, var_key)
    if var_key == "sst":
        da = normalize_temperature_units(da)
    mask = build_valid_mask(da)
    return da, mask


def qc_report(da: xr.DataArray, var_key: str) -> dict:
    """Small summary dict useful for logging / the /metadata endpoint."""
    total = int(da.size)
    n_valid = int(np.isfinite(da.values).sum())
    return {
        "variable": var_key,
        "total_points": total,
        "valid_points": n_valid,
        "missing_pct": round(100.0 * (total - n_valid) / total, 3) if total else None,
    }
