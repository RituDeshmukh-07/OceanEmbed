"""
Central configuration for the subsurface-temperature reconstruction backend.

Edit the paths / variable-name mappings here once you know the exact
filenames and variable names inside your real SST / SSS / SSH files.
Everything else in the pipeline reads from this file, so you should not
need to touch other modules just to point at a new dataset.
"""

from pathlib import Path

# ------------------------------------------------------------------
# Domain (North Indian Ocean, per the problem statement)
# ------------------------------------------------------------------
LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
GRID_RESOLUTION_DEG = 0.25

# Standard output depth levels (meters)
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# ------------------------------------------------------------------
# Raw data locations
# ------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"      # harmonized, QC'd NetCDF
PROCESSED_DIR = DATA_DIR / "processed"  # ML-ready Zarr stores

for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Input surface variables
#
# `file_glob`   : pattern used to find raw files for that variable
# `var_name`    : the variable name INSIDE the NetCDF (check with the
#                 inspect_dataset.py script and update this)
# `valid_range` : physically plausible range, used for QC masking
# `required`    : if False, the pipeline runs fine without this
#                 variable (useful since you don't have currents/winds yet)
# ------------------------------------------------------------------
SURFACE_VARIABLES = {
    "sst": {
        "file_glob":"SST/SST_*.nc",
        "var_name": "analysed_sst",   # e.g. OISST/GHRSST naming - adjust
        "units": "kelvin_or_celsius", # normalize_units() handles both
        "valid_range": (-2.0, 40.0),  # deg C
        "required": True,
    },
    "sss": {
        "file_glob": "SSS/SSS_*.nc",
        "var_name": "sss",            # e.g. SMAP/OISSS naming - adjust
        "units": "psu",
        "valid_range": (2.0, 42.0),
        "required": True,
    },
    "ssh": {
        "file_glob": "SSH/SSH_CURRENTS_*.nc",
        "var_name": "sla",            # SSH products often store SLA - adjust
        "units": "meters",
        "valid_range": (-2.0, 2.0),
        "required": True,
    },
    "u_current": {
        "file_glob": "cur_*.nc",
        "var_name": "ugos",
        "units": "m/s",
        "valid_range": (-3.0, 3.0),
        "required": False,
    },
    "v_current": {
        "file_glob": "cur_*.nc",
        "var_name": "vgos",
        "units": "m/s",
        "valid_range": (-3.0, 3.0),
        "required": False,
    },
    "u_wind": {
        "file_glob": "wind_*.nc",
        "var_name": "u10",
        "units": "m/s",
        "valid_range": (-40.0, 40.0),
        "required": False,
    },
    "v_wind": {
        "file_glob": "wind_*.nc",
        "var_name": "v10",
        "units": "m/s",
        "valid_range": (-40.0, 40.0),
        "required": False,
    },
}

# ------------------------------------------------------------------
# Target variable (GLORYS reanalysis temperature, used only for training)
# ------------------------------------------------------------------
TARGET_VARIABLE = {
    "file_glob": "glorys_thetao_*.nc",
    "var_name": "thetao",
    "depth_dim": "depth",
    "valid_range": (-2.0, 40.0),
}

# ------------------------------------------------------------------
# Model checkpoint (Ritu's trained model)
# ------------------------------------------------------------------
MODEL_CHECKPOINT_PATH = DATA_DIR / "model" / "subsurface_temp_model.pt"
MODEL_INPUT_ORDER = ["sst", "sss", "ssh", "u_current", "v_current", "u_wind", "v_wind"]

# ------------------------------------------------------------------
# Normalization stats file (produced by preprocessing.fit_normalization)
# ------------------------------------------------------------------
NORM_STATS_PATH = PROCESSED_DIR / "normalization_stats.json"
