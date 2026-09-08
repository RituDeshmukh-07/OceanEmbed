"""
Central configuration for OceanEmbed.

Everything that a user might reasonably want to tweak (region, depths,
sampling, model size, training hyperparameters, split dates) lives here.
Nothing outside this file should hard-code these values.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "glorys"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
PLOTS_DIR = OUTPUTS_DIR / "plots"

for _d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, METRICS_DIR,
           PREDICTIONS_DIR, PLOTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

NORM_STATS_PATH = PROCESSED_DATA_DIR / "norm_stats.json"
DEPTH_MAPPING_PATH = PROCESSED_DATA_DIR / "depth_mapping.json"

TRAIN_SAMPLES_PATH = PROCESSED_DATA_DIR / "train_samples.npz"
VAL_SAMPLES_PATH = PROCESSED_DATA_DIR / "val_samples.npz"
TEST_SAMPLES_PATH = PROCESSED_DATA_DIR / "test_samples.npz"

BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"
FINAL_CHECKPOINT_PATH = CHECKPOINT_DIR / "final_model.pth"

# --------------------------------------------------------------------------
# REGION OF INTEREST (Arabian Sea / Bay of Bengal approach box)
# --------------------------------------------------------------------------
LAT_MIN = 0.0
LAT_MAX = 25.0
LON_MIN = 60.0
LON_MAX = 80.0

# --------------------------------------------------------------------------
# TEMPORAL SPLIT (chronological, to avoid leakage). Edit freely.
# Format: "YYYY-MM-DD"
# --------------------------------------------------------------------------
TRAIN_START = "2022-01-01"
TRAIN_END = "2022-09-30"

VAL_START = "2022-10-01"
VAL_END = "2022-11-30"

TEST_START = "2022-12-01"
TEST_END = "2022-12-31"

# --------------------------------------------------------------------------
# TARGET DEPTHS (m). Nearest available GLORYS depth levels are chosen for
# each of these; the actual chosen levels are printed (never silently
# interpolated) and saved to DEPTH_MAPPING_PATH.
# --------------------------------------------------------------------------
TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
N_DEPTHS = len(TARGET_DEPTHS)

# --------------------------------------------------------------------------
# INPUT VARIABLES
#
# The FIRST prototype uses only SST + SSS. The list below is deliberately
# an ordered, named list (not a hard-coded "2") so extra surface channels
# (SSH/SLA, currents, wind) can be added later purely through config -
# model.py reads len(INPUT_VARIABLES) to size its first conv layer.
#
# Each entry maps a logical channel name -> list of candidate GLORYS/CMEMS
# variable names to look for (first match wins). Only variables enabled
# here are actually required to be present in the NetCDF files.
# --------------------------------------------------------------------------
INPUT_VARIABLES = ["sst", "sss"]

VARIABLE_CANDIDATES = {
    # surface temperature: taken from the shallowest level of the 3D
    # temperature field (thetao). GLORYS calls the field "thetao"; some
    # distributions use "temperature" or "votemper" (NEMO/ORCA legacy).
    "sst": ["thetao", "temperature", "votemper", "sst", "analysed_sst"],
    # surface salinity: shallowest level of the 3D salinity field.
    "sss": ["so", "salinity", "vosaline", "sss"],
    # kept here for future 7-variable support (not required in prototype)
    "ssh": ["zos", "sla", "ssh", "adt"],
    "u_current": ["uo", "u_current", "eastward_current"],
    "v_current": ["vo", "v_current", "northward_current"],
    "u_wind": ["uas", "u10", "eastward_wind"],
    "v_wind": ["vas", "v10", "northward_wind"],
}

# The 3D field used both to derive SST/SSS-like surface channels via the
# shallowest level AND to build the temperature profile target.
TEMPERATURE_VAR_CANDIDATES = VARIABLE_CANDIDATES["sst"] if False else \
    ["thetao", "temperature", "votemper"]
SALINITY_VAR_CANDIDATES = ["so", "salinity", "vosaline"]

COORD_CANDIDATES = {
    "lat": ["latitude", "lat", "nav_lat", "y"],
    "lon": ["longitude", "lon", "nav_lon", "x"],
    "time": ["time", "time_counter"],
    "depth": ["depth", "deptht", "lev", "z"],
}

# --------------------------------------------------------------------------
# SPATIAL SAMPLING (turns a full lat/lon field per timestep into many
# small "patch -> profile" training samples, keeping the prototype
# trainable on a laptop).
# --------------------------------------------------------------------------
# Odd patch size (pixels) of the surface CNN input, e.g. 15x15 grid cells
# centered on the point whose subsurface profile is the prediction target.
PATCH_SIZE = 15

# Spatial stride (in grid cells) between sampled center points. Larger =
# fewer, more spread-out samples per timestep.
SPATIAL_STRIDE = 12

# Temporal stride (in days) between timesteps used for sampling, so we
# don't need every single day of the year.
TEMPORAL_STRIDE_DAYS = 7

# Hard cap on number of samples per split, so the prototype always fits
# comfortably in memory/time on a laptop, regardless of region size.
MAX_TRAIN_SAMPLES = 6000
MAX_VAL_SAMPLES = 1500
MAX_TEST_SAMPLES = 1500

# Random seed used only for shuffling *which* candidate samples are kept
# when subsampling down to the MAX_*_SAMPLES cap (NOT for the chronological
# train/val/test split itself, which is date-based and deterministic).
SAMPLING_SEED = 42

# --------------------------------------------------------------------------
# MODEL
# --------------------------------------------------------------------------
EMBEDDING_DIM = 256  # 128 / 256 / 512 supported
CONV_CHANNELS = [32, 64, 128]  # feature maps per conv block
DECODER_HIDDEN_DIMS = [128, 64]

# --------------------------------------------------------------------------
# TRAINING
# --------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 8
RANDOM_SEED = 42
NUM_WORKERS = 0  # safest default on Windows/laptops; raise on Linux if desired

# --------------------------------------------------------------------------
# MISC
# --------------------------------------------------------------------------
NC_GLOB_PATTERNS = ["*.nc", "*.nc4"]
FILL_VALUE_THRESHOLD = 1e30  # values >= this magnitude are treated as missing
