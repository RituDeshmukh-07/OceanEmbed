"""
Task 1: Inspect the real SIH dataset.

Run this against every raw file you have (SST, SSS, SSH, GLORYS target,
ARGO, etc.) BEFORE writing any loader/harmonization code. It prints:
  - format
  - variables and their dimensions/units
  - time coverage
  - spatial coverage
  - missing value stats (count + %) per variable

Usage:
    python scripts/inspect_dataset.py /path/to/file.nc
    python scripts/inspect_dataset.py /path/to/folder/*.nc
"""

import sys
import glob
import xarray as xr
import numpy as np


def inspect_file(path: str) -> None:
    print("=" * 80)
    print(f"FILE: {path}")
    print("=" * 80)

    try:
        ds = xr.open_dataset(path, decode_times=True)
    except Exception as e:
        print(f"  Could not open as NetCDF/HDF5 with xarray: {e}")
        return

    print(f"\nFormat: {ds.encoding.get('source', 'unknown')} "
          f"(engine inferred by xarray)")

    print("\n--- Dimensions ---")
    for dim, size in ds.dims.items():
        print(f"  {dim}: {size}")

    print("\n--- Coordinates ---")
    for coord in ds.coords:
        c = ds.coords[coord]
        try:
            print(f"  {coord}: dtype={c.dtype}, "
                  f"range=({np.nanmin(c.values)} to {np.nanmax(c.values)})")
        except Exception:
            print(f"  {coord}: dtype={c.dtype} (range not printable)")

    print("\n--- Data variables ---")
    for var in ds.data_vars:
        da = ds[var]
        units = da.attrs.get("units", "N/A")
        long_name = da.attrs.get("long_name", "N/A")
        print(f"\n  Variable: {var}")
        print(f"    long_name : {long_name}")
        print(f"    units     : {units}")
        print(f"    dims      : {da.dims}")
        print(f"    shape     : {da.shape}")

        try:
            vals = da.values.astype("float64")
            n_total = vals.size
            n_missing = int(np.isnan(vals).sum())
            pct_missing = 100.0 * n_missing / n_total if n_total else 0.0
            finite = vals[np.isfinite(vals)]
            if finite.size:
                print(f"    min/max   : {finite.min():.4f} / {finite.max():.4f}")
            print(f"    missing   : {n_missing} / {n_total} ({pct_missing:.2f}%)")
        except Exception as e:
            print(f"    (could not compute stats: {e})")

    # Time coverage
    for time_name in ("time", "TIME", "Time"):
        if time_name in ds.coords:
            t = ds.coords[time_name].values
            print(f"\n--- Time coverage ({time_name}) ---")
            print(f"  start: {t.min()}")
            print(f"  end  : {t.max()}")
            print(f"  n_steps: {len(t)}")
            break

    # Spatial coverage
    for lat_name in ("lat", "latitude", "LAT", "nav_lat"):
        if lat_name in ds.coords:
            lat = ds.coords[lat_name].values
            print(f"\n--- Latitude coverage ({lat_name}) ---")
            print(f"  min: {np.nanmin(lat):.3f}, max: {np.nanmax(lat):.3f}")
            if lat.ndim == 1 and len(lat) > 1:
                print(f"  approx resolution: {abs(lat[1] - lat[0]):.4f} deg")
            break

    for lon_name in ("lon", "longitude", "LON", "nav_lon"):
        if lon_name in ds.coords:
            lon = ds.coords[lon_name].values
            print(f"\n--- Longitude coverage ({lon_name}) ---")
            print(f"  min: {np.nanmin(lon):.3f}, max: {np.nanmax(lon):.3f}")
            if lon.ndim == 1 and len(lon) > 1:
                print(f"  approx resolution: {abs(lon[1] - lon[0]):.4f} deg")
            break

    ds.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        paths.extend(glob.glob(arg))

    if not paths:
        print("No files matched the given path(s).")
        sys.exit(1)

    for p in sorted(paths):
        inspect_file(p)


if __name__ == "__main__":
    main()
