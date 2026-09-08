"""
Task 5-8: FastAPI backend exposing /health, /metadata and /predict.

Run locally with:
    uvicorn app.main:app --reload --port 8000

Then:
    curl http://localhost:8000/health
    curl http://localhost:8000/metadata
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"date": "2023-06-15"}'
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import config, data_loader, model_interface, preprocessing, qc
from .schemas import HealthResponse, MetadataResponse, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocean_backend.api")

app = FastAPI(
    title="North Indian Ocean Subsurface Temperature Reconstruction API",
    description=(
        "Reconstructs depth-wise subsurface ocean temperature from daily "
        "surface satellite observations (SST/SSS/SSH[/currents/winds]) "
        "using a satellite-embedding deep learning model."
    ),
    version="0.1.0",
)

# Allow Sakshi's frontend to call this API from a different origin during dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to the frontend's actual origin in prod
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + model-availability check."""
    status = model_interface.model_status()
    return HealthResponse(
        status="ok",
        model_backend=status["backend"],
        checkpoint_found=status["checkpoint_found"],
    )


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    """Domain, grid, variable and model metadata for the frontend to render."""
    status = model_interface.model_status()
    return MetadataResponse(
        domain={
            "lat_min": config.LAT_MIN,
            "lat_max": config.LAT_MAX,
            "lon_min": config.LON_MIN,
            "lon_max": config.LON_MAX,
            "region_name": "North Indian Ocean",
        },
        grid_resolution_deg=config.GRID_RESOLUTION_DEG,
        standard_depths_m=[float(d) for d in config.STANDARD_DEPTHS],
        surface_variables={
            k: {"required": v["required"], "units": v["units"]}
            for k, v in config.SURFACE_VARIABLES.items()
        },
        model={
            "backend": status["backend"],
            "is_mock": status["is_mock"],
            "input_channel_order": config.MODEL_INPUT_ORDER,
        },
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    End-to-end: load surface obs for the requested date/region -> QC ->
    harmonize -> run model -> return depths + predicted temperatures + JSON.
    """
    date_str = req.date.isoformat()

    try:
        raw_vars = data_loader.load_all_available_surface_variables(
            start_date=date_str, end_date=date_str,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not raw_vars:
        raise HTTPException(
            status_code=404,
            detail=f"No surface observations found for {date_str}.",
        )

    cleaned_vars = {}
    for key, da in raw_vars.items():
        cleaned, _mask = qc.run_qc(da, key)
        cleaned_vars[key] = cleaned

    try:
        tensor, present_channels, template = preprocessing.harmonize(cleaned_vars)
    except FileNotFoundError as e:
        # normalization stats missing -> fit on the fly as a dev fallback
        logger.warning("Normalization stats missing, fitting on this batch: %s", e)
        regridded = {k: preprocessing.regrid_to_common_grid(v)
                     for k, v in cleaned_vars.items()}
        aligned = preprocessing.align_in_time(regridded)
        stats = preprocessing.fit_normalization(aligned)
        normalized = preprocessing.apply_normalization(aligned, stats)
        tensor, present_channels = preprocessing.stack_to_tensor(normalized)
        template = next(iter(aligned.values()))

    prediction = model_interface.predict(tensor)  # (time, depth, lat, lon)

    # Requested date is the only time step we return.
    temp_slice = prediction[0]  # (depth, lat, lon)

    lats = template["lat"].values.tolist()
    lons = template["lon"].values.tolist()

    return PredictResponse(
        date=req.date,
        depths_m=[float(d) for d in config.STANDARD_DEPTHS],
        lat=[float(v) for v in lats],
        lon=[float(v) for v in lons],
        temperature=np.round(temp_slice, 3).tolist(),
        channels_used=present_channels,
        model_backend=model_interface.model_status()["backend"],
        metrics=None,  # populate once ARGO validation (task: evaluate) is wired in
    )
