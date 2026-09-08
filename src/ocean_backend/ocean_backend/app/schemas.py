"""
Pydantic request/response models for the FastAPI service.
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from . import config


class PredictRequest(BaseModel):
    date: _dt.date = Field(..., description="Date to reconstruct (YYYY-MM-DD)")
    lat_min: float = Field(config.LAT_MIN, ge=-90, le=90)
    lat_max: float = Field(config.LAT_MAX, ge=-90, le=90)
    lon_min: float = Field(config.LON_MIN, ge=-180, le=360)
    lon_max: float = Field(config.LON_MAX, ge=-180, le=360)

    @field_validator("lat_max")
    @classmethod
    def lat_order(cls, v, info):
        if "lat_min" in info.data and v <= info.data["lat_min"]:
            raise ValueError("lat_max must be greater than lat_min")
        return v

    @field_validator("lon_max")
    @classmethod
    def lon_order(cls, v, info):
        if "lon_min" in info.data and v <= info.data["lon_min"]:
            raise ValueError("lon_max must be greater than lon_min")
        return v


class PredictResponse(BaseModel):
    date: _dt.date
    depths_m: list[float]
    lat: list[float]
    lon: list[float]
    temperature: list[list[list[float]]] = Field(
        ..., description="temperature[depth_idx][lat_idx][lon_idx], degC"
    )
    channels_used: list[str]
    model_backend: str
    metrics: Optional[dict] = Field(
        None, description="Skill metrics vs. independent ARGO obs, if available"
    )


class MetadataResponse(BaseModel):
    domain: dict
    grid_resolution_deg: float
    standard_depths_m: list[float]
    surface_variables: dict
    model: dict


class HealthResponse(BaseModel):
    status: str
    model_backend: str
    checkpoint_found: bool
