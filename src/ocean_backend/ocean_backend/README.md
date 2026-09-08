# North Indian Ocean Subsurface Temperature Backend

Backend for reconstructing depth-wise subsurface ocean temperature from
daily surface satellite observations (SST / SSS / SSH, optionally +
currents/winds later), for the North Indian Ocean (5-30N, 45-105E) at
0.25 deg / daily resolution. Built to plug into Ritu's DL model and
Sakshi's frontend.

## Project layout

```
ocean_backend/
  app/
    config.py          # domain, grid, variable name mappings, paths
    data_loader.py      # Task 2: xarray loader (region/date/variable select)
    qc.py                # Task 3: missing/invalid masking, coord ordering
    preprocessing.py    # Task 4: regrid, time-align, normalize, tensor stack
    model_interface.py  # Task 6/7: load Ritu's checkpoint, run inference
    validation.py        # Evaluate: RMSE / Bias / correlation vs ARGO
    schemas.py           # Pydantic request/response models
    main.py               # Task 5/6/8: FastAPI app (/health /metadata /predict)
  scripts/
    inspect_dataset.py  # Task 1: inspect real SIH dataset files
  data/
    raw/        # put your raw SST/SSS/SSH/GLORYS/ARGO NetCDF files here
    interim/    # harmonized/QC'd intermediate NetCDF
    processed/  # ML-ready Zarr + normalization_stats.json
    model/      # Ritu's checkpoint goes here
  requirements.txt
```

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# only if/when Ritu's model needs torch:
pip install torch
```

## 2. Inspect your real datasets first

Before touching the loader, run this against every file you have:

```bash
python scripts/inspect_dataset.py "data/raw/sst_*.nc"
python scripts/inspect_dataset.py "data/raw/sss_*.nc"
python scripts/inspect_dataset.py "data/raw/ssh_*.nc"
```

It prints dimensions, variable names/units, time coverage, spatial
coverage and missing-value stats. Use the printed variable names to
update `app/config.py -> SURFACE_VARIABLES[...]["var_name"]` if they
don't match the current guesses (`analysed_sst`, `sss`, `sla`).

## 3. Drop your data in

Put raw files matching the globs in `config.py` into `data/raw/`:
- `sst_*.nc`, `sss_*.nc`, `ssh_*.nc` (you have these three now)
- `cur_*.nc`, `wind_*.nc` (optional, add later — pipeline works without them)
- `glorys_thetao_*.nc` (training target only)

## 4. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

- `GET /health` — liveness + whether a real model checkpoint is loaded
  (falls back to a clearly-labeled mock predictor otherwise, so the API
  is testable before Ritu's checkpoint exists).
- `GET /metadata` — domain box, grid resolution, standard depths,
  configured surface variables, model info. This is what Sakshi's
  frontend should call first to know what params/labels to render.
- `POST /predict` — body `{"date": "2023-06-15"}` (optionally
  `lat_min/lat_max/lon_min/lon_max` to subset). Returns depths, lat/lon
  grid, and `temperature[depth][lat][lon]` as JSON.

## 5. Plug in Ritu's model

Edit `app/model_interface.py`:
- `_build_model_architecture()` — instantiate her `nn.Module`
- Put her checkpoint at `data/model/subsurface_temp_model.pt`
  (or update `config.MODEL_CHECKPOINT_PATH`)
- If her `forward()` signature differs from
  `(time, channel, lat, lon) -> (time, depth, lat, lon)`, adjust
  `predict()` accordingly.

Until then, `/predict` transparently uses a mock predictor so the rest
of the stack (loader, QC, harmonization, API, frontend integration) can
be built and demoed without blocking on the model.

## 6. Training pipeline (once you get to that)

```python
from app import data_loader, qc, preprocessing

surface_vars = data_loader.load_all_available_surface_variables("2020-01-01", "2023-12-31")
cleaned = {k: qc.run_qc(v, k)[0] for k, v in surface_vars.items()}
tensor, channels, template = preprocessing.harmonize(cleaned, fit_norm=True)

target = data_loader.load_target_temperature("2020-01-01", "2023-12-31")
# -> feed (tensor, target) into Ritu's training loop
```

## 7. Validate against ARGO

```python
from app import validation
metrics = validation.evaluate(predicted_da, argo_gridded_da)
# metrics["overall"] = {"rmse":..., "bias":..., "correlation":..., "n_points":...}
# metrics["per_depth"][depth_m] = {...}   # per standard-depth breakdown
```

## Notes / gotchas

- **Do not overbuild**: no PostgreSQL/PostGIS. Raw/intermediate = NetCDF,
  ML-ready = Zarr, as specified.
- SST unit auto-detection: if the median value in a file is > 100 it's
  assumed to be Kelvin and converted to Celsius — verify this against
  the real file's `units` attribute printed by `inspect_dataset.py`.
- Optional channels (currents/winds) are zero-filled in the tensor when
  absent so the model's input shape stays fixed; `channels_used` in the
  `/predict` response tells the frontend which channels were real for
  that request.
