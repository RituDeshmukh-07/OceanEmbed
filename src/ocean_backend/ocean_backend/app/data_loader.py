"""
Task 2: xarray loader that selects the required region, date and variables.

This module is the single place that knows how to open a raw SST / SSS / SSH
(or any configured surface variable) file and return a clean, subset
xr.DataArray for a given date range and the North Indian Ocean domain.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import xarray as xr

from . import config


DateLike = Union[str, np.datetime64]


def _find_files(var_key: str) -> list[str]:
    spec = config.SURFACE_VARIABLES[var_key]
    pattern = str(config.RAW_DIR / spec["file_glob"])
    files = sorted(glob.glob(pattern))
    return files


def _standardize_coord_names(ds: xr.Dataset) -> xr.Dataset:
    """Rename common coordinate aliases to lat/lon/time for consistency."""
    rename_map = {}
    for cand in ("latitude", "LAT", "nav_lat", "y"):
        if cand in ds.coords:
            rename_map[cand] = "lat"
    for cand in ("longitude", "LON", "nav_lon", "x"):
        if cand in ds.coords:
            rename_map[cand] = "lon"
    for cand in ("TIME", "Time", "time_counter"):
        if cand in ds.coords:
            rename_map[cand] = "time"
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def _ensure_ascending_lat(ds: xr.Dataset) -> xr.Dataset:
    """Standard depths/lat need a consistent, ascending coordinate order."""
    if "lat" in ds.coords and ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")
    if "lon" in ds.coords and ds["lon"].values[0] > ds["lon"].values[-1]:
        ds = ds.sortby("lon")
    return ds


def _wrap_longitudes_0_360_to_pm180_if_needed(ds: xr.Dataset) -> xr.Dataset:
    """
    Some products store longitude in 0-360. Our domain (45E-105E) is
    unambiguous either way, but this keeps things consistent if a global
    0-360 product is used.
    """
    lon = ds["lon"].values
    if lon.max() > 180.0 and config.LON_MAX <= 180.0:
        new_lon = ((lon + 180) % 360) - 180
        ds = ds.assign_coords(lon=new_lon).sortby("lon")
    return ds


def load_surface_variable(
    var_key: str,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    lat_bounds: tuple[float, float] = (config.LAT_MIN, config.LAT_MAX),
    lon_bounds: tuple[float, float] = (config.LON_MIN, config.LON_MAX),
    chunks: str | dict = "auto",
) -> xr.DataArray:
    """
    Load one configured surface variable (e.g. 'sst', 'sss', 'ssh'),
    subset to the requested date range and the North Indian Ocean box.

    Returns a DataArray named `var_key` with dims (time, lat, lon).
    Uses dask (via `chunks=`) so multi-file, multi-year loads stay lazy.
    """
    if var_key not in config.SURFACE_VARIABLES:
        raise KeyError(f"Unknown variable '{var_key}'. "
                        f"Configured keys: {list(config.SURFACE_VARIABLES)}")

    spec = config.SURFACE_VARIABLES[var_key]
    files = _find_files(var_key)
    if not files:
        if spec["required"]:
            raise FileNotFoundError(
                f"No raw files found for required variable '{var_key}' "
                f"matching pattern '{spec['file_glob']}' in {config.RAW_DIR}"
            )
        return None  # optional variable simply absent

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        chunks=chunks,
        decode_times=True,
    )
    ds = _standardize_coord_names(ds)
    ds = _ensure_ascending_lat(ds)
    ds = _wrap_longitudes_0_360_to_pm180_if_needed(ds)

    if spec["var_name"] not in ds.data_vars:
        raise KeyError(
            f"Configured var_name '{spec['var_name']}' for '{var_key}' not "
            f"found in file. Available variables: {list(ds.data_vars)}. "
            f"Update app/config.py SURFACE_VARIABLES['{var_key}']['var_name']."
        )

    da = ds[spec["var_name"]]

    # Spatial subset (North Indian Ocean box)
    da = da.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))

    # Temporal subset
    if start_date is not None or end_date is not None:
        da = da.sel(time=slice(start_date, end_date))

    da = da.rename(var_key)
    return da


def load_all_available_surface_variables(
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
) -> dict[str, xr.DataArray]:
    """
    Load every configured surface variable that has files available.
    Required variables that are missing raise an error; optional
    variables (currents, winds) are silently skipped if absent, matching
    your current situation of only having SST/SSS/SSH.
    """
    out: dict[str, xr.DataArray] = {}
    for var_key, spec in config.SURFACE_VARIABLES.items():
        da = load_surface_variable(var_key, start_date, end_date)
        if da is not None:
            out[var_key] = da
        elif spec["required"]:
            raise FileNotFoundError(f"Required variable '{var_key}' missing.")
    return out


def load_target_temperature(
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    depths: Sequence[float] = tuple(config.STANDARD_DEPTHS),
) -> xr.DataArray:
    """
    Load the GLORYS reanalysis subsurface temperature, used as the
    training target. Interpolates onto STANDARD_DEPTHS.
    """
    spec = config.TARGET_VARIABLE
    files = sorted(glob.glob(str(config.RAW_DIR / spec["file_glob"])))
    if not files:
        raise FileNotFoundError(
            f"No GLORYS target files found matching '{spec['file_glob']}' "
            f"in {config.RAW_DIR}"
        )

    ds = xr.open_mfdataset(files, combine="by_coords", chunks="auto")
    ds = _standardize_coord_names(ds)
    ds = _ensure_ascending_lat(ds)

    da = ds[spec["var_name"]]
    da = da.sel(lat=slice(config.LAT_MIN, config.LAT_MAX),
                lon=slice(config.LON_MIN, config.LON_MAX))

    if start_date is not None or end_date is not None:
        da = da.sel(time=slice(start_date, end_date))

    depth_dim = spec["depth_dim"]
    if depth_dim in da.dims:
        da = da.interp({depth_dim: list(depths)})

    return da.rename("temperature")
