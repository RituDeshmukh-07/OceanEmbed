"""
OceanEmbed inference.

Loads the trained checkpoint and converts an SST/SSS surface patch
into a 15-depth temperature profile.
"""

from __future__ import annotations

import numpy as np
import torch

from ml import config
from ml.model import build_model
from ml.preprocessing import load_norm_stats, normalize_inputs_only, denormalize_targets


_MODEL = None
_DEVICE = None
_STATS = None


def _load_model():
    global _MODEL, _DEVICE, _STATS

    if _MODEL is not None:
        return

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = config.BEST_CHECKPOINT_PATH

    if not checkpoint_path.exists():
        checkpoint_path = config.FINAL_CHECKPOINT_PATH

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "No trained checkpoint found. Run training first."
        )

    _MODEL = build_model().to(_DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=_DEVICE
    )

    _MODEL.load_state_dict(checkpoint["model_state_dict"])
    _MODEL.eval()

    _STATS = load_norm_stats()


def predict_temperature(sst, sss):
    """
    Predict 15-depth temperature profile.

    Parameters
    ----------
    sst : 2D numpy array
        SST surface patch.
    sss : 2D numpy array
        SSS surface patch.

    Returns
    -------
    numpy array
        Predicted temperatures in deg C at the 15 target depths.
    """

    _load_model()

    sst = np.asarray(sst, dtype=np.float32)
    sss = np.asarray(sss, dtype=np.float32)

    expected = config.PATCH_SIZE

    if sst.shape != (expected, expected):
        raise ValueError(
            f"SST must have shape {(expected, expected)}, "
            f"got {sst.shape}"
        )

    if sss.shape != (expected, expected):
        raise ValueError(
            f"SSS must have shape {(expected, expected)}, "
            f"got {sss.shape}"
        )

    inputs = np.stack([sst, sss], axis=0)
    inputs = inputs[None, ...]

    inputs_norm = normalize_inputs_only(
        inputs,
        _STATS
    )

    x = torch.from_numpy(inputs_norm).to(_DEVICE)

    with torch.no_grad():
        prediction_norm = _MODEL(x).cpu().numpy()

    prediction_c = denormalize_targets(
        prediction_norm,
        _STATS
    )[0]

    return prediction_c


def predict_from_patch(patch):
    """
    Convenience function.

    patch shape:
        [2, PATCH_SIZE, PATCH_SIZE]
    """

    patch = np.asarray(patch, dtype=np.float32)

    if patch.shape != (
        2,
        config.PATCH_SIZE,
        config.PATCH_SIZE
    ):
        raise ValueError(
            "Expected patch shape "
            f"(2, {config.PATCH_SIZE}, {config.PATCH_SIZE}), "
            f"got {patch.shape}"
        )

    return predict_temperature(
        patch[0],
        patch[1]
    )


if __name__ == "__main__":
    print("OceanEmbed prediction module")
    print("Target depths:")
    print(config.TARGET_DEPTHS)