"""
Task 6 & 7: Load Ritu's checkpoint and expose a single predict() function
the API can call.

This module intentionally isolates all model-specific code. When Ritu
hands you her actual model class + checkpoint, you only need to edit:
  - `_build_model_architecture()`  -> instantiate her nn.Module
  - `predict()`                    -> if her forward() signature differs

Until then, `predict()` transparently falls back to a lightweight mock
predictor (a smooth climatological-shape profile scaled by SST) so the
rest of the backend (FastAPI, harmonization, tests) can be built and
demoed end-to-end without blocking on the model.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import config

logger = logging.getLogger("ocean_backend.model")

_MODEL = None          # lazy-loaded singleton
_MODEL_BACKEND = None  # "torch" | "mock"


def _build_model_architecture():
    """
    Instantiate Ritu's model class here, e.g.:

        from .architectures.reconstruction_net import SubsurfaceReconstructor
        model = SubsurfaceReconstructor(
            in_channels=len(config.MODEL_INPUT_ORDER),
            out_depths=len(config.STANDARD_DEPTHS),
        )
        return model

    Left unimplemented until her architecture module is available.
    """
    raise NotImplementedError(
        "Plug in Ritu's model class in _build_model_architecture()."
    )


def load_model():
    """
    Load the trained checkpoint once per process. Falls back to a mock
    predictor with a warning if the checkpoint or torch is unavailable,
    so the API stays usable during development.
    """
    global _MODEL, _MODEL_BACKEND
    if _MODEL is not None:
        return _MODEL

    try:
        import torch  # noqa: F401
    except ImportError:
        logger.warning("PyTorch not installed — using mock predictor.")
        _MODEL_BACKEND = "mock"
        _MODEL = "mock"
        return _MODEL

    if not config.MODEL_CHECKPOINT_PATH.exists():
        logger.warning(
            "No checkpoint found at %s — using mock predictor.",
            config.MODEL_CHECKPOINT_PATH,
        )
        _MODEL_BACKEND = "mock"
        _MODEL = "mock"
        return _MODEL

    import torch

    try:
        model = _build_model_architecture()
        state_dict = torch.load(config.MODEL_CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        _MODEL = model
        _MODEL_BACKEND = "torch"
        logger.info("Loaded model checkpoint from %s", config.MODEL_CHECKPOINT_PATH)
    except NotImplementedError:
        logger.warning(
            "_build_model_architecture() not implemented yet — using mock predictor."
        )
        _MODEL_BACKEND = "mock"
        _MODEL = "mock"

    return _MODEL


def _mock_predict(tensor: np.ndarray) -> np.ndarray:
    """
    Deterministic placeholder: builds a plausible-looking thermocline
    profile per grid cell, anchored to the (normalized) SST channel, so
    the API returns sane-shaped output during development. This has NO
    scientific validity — replace with the real model as soon as
    possible.
    """
    n_time, _, n_lat, n_lon = tensor.shape
    n_depths = len(config.STANDARD_DEPTHS)
    depths = np.array(config.STANDARD_DEPTHS, dtype="float32")

    sst_channel = tensor[:, 0, :, :]  # sst is channel 0 in MODEL_INPUT_ORDER
    out = np.zeros((n_time, n_depths, n_lat, n_lon), dtype="float32")

    # simple exponential decay with depth, offset/scaled by (normalized) SST
    decay = np.exp(-depths / 250.0)  # (n_depths,)
    for d_idx in range(n_depths):
        out[:, d_idx, :, :] = sst_channel * decay[d_idx]

    return out


def predict(tensor: np.ndarray) -> np.ndarray:
    """
    Run inference.

    Parameters
    ----------
    tensor : np.ndarray, shape (time, channel, lat, lon)
        Output of preprocessing.harmonize().

    Returns
    -------
    np.ndarray, shape (time, depth, lat, lon)
        Reconstructed temperature at config.STANDARD_DEPTHS.
    """
    model = load_model()

    if _MODEL_BACKEND == "mock":
        return _mock_predict(tensor)

    import torch
    with torch.no_grad():
        x = torch.from_numpy(tensor).float()
        y = model(x)
        return y.cpu().numpy()


def model_status() -> dict:
    load_model()
    return {
        "backend": _MODEL_BACKEND,
        "checkpoint_path": str(config.MODEL_CHECKPOINT_PATH),
        "checkpoint_found": config.MODEL_CHECKPOINT_PATH.exists(),
        "is_mock": _MODEL_BACKEND == "mock",
    }
