import os
from pathlib import Path

import boto3
import numpy as np
import s3fs
import xarray as xr
from dotenv import load_dotenv

load_dotenv()
BUCKET_NAME = os.environ["S3_BUCKET"]
NDVI_ZARR_PATH = f"{BUCKET_NAME}/ndvi_processed.zarr"
PRED_ZARR_PATH = f"{BUCKET_NAME}/ndvi_predictions.zarr"
NUM_FUTURE_STEPS = 3
NUM_PAST_STEPS = 3
RASTER_LOCAL_DIR = "/tmp"
RASTER_PREFIX = "COG"

fs = s3fs.S3FileSystem()

if fs.exists(f"{NDVI_ZARR_PATH}/zarr.json"):
    ndvi_store = s3fs.S3Map(root=NDVI_ZARR_PATH, s3=fs, check=False)
    ndvi = xr.open_zarr(ndvi_store).ndvi_8d_processed
else:
    raise FileNotFoundError(f"{NDVI_ZARR_PATH} was not found.")

if fs.exists(f"{PRED_ZARR_PATH}/zarr.json"):
    pred_store = s3fs.S3Map(root=PRED_ZARR_PATH, s3=fs, check=False)
    preds = xr.open_zarr(pred_store).ndvi_8d_forecast
else:
    raise FileNotFoundError(f"{PRED_ZARR_PATH} was not found.")

s3 = boto3.client("s3")

### Trend plot
ndvi_recent = ndvi.isel(time=slice(-4, -1))
ndvi_last = ndvi_recent.isel(time=-1)

fit = ndvi_recent.polyfit(dim="time", deg=1)
slope = fit.polyfit_coefficients.sel(degree=1)
slope = (
    slope * 1e9 * 60 * 60 * 24 * 8
)  # convert from slope based on nanoseconds to 8 day intervals

# Thresholds
ndvi_thresh_sparse = 0.1
ndvi_thresh_peak = 0.6
slope_thresh = 0.01  # tolerance for "stable"

# Start with all zeros
classes = xr.full_like(ndvi_last, fill_value=0, dtype=np.int8)

# Bare/sparse vegetation
classes = classes.where(~(ndvi_last <= ndvi_thresh_sparse), 0)

# Stable (slope near zero, regardless of NDVI)
stable_mask = np.abs(slope) < slope_thresh
classes = classes.where(~stable_mask, 1)

# Greening (NDVI > 0.1 and slope > 0)
greening_mask = (ndvi_last > ndvi_thresh_sparse) & (slope > slope_thresh)
classes = classes.where(~greening_mask, 2)

# Browning/senescence (NDVI > 0.1 and slope < 0)
browning_mask = (ndvi_last > ndvi_thresh_sparse) & (slope < -slope_thresh)
classes = classes.where(~browning_mask, 3)

# Peak growth (NDVI > 0.6 and slope near zero but preceded by positive slope)
# Here we check if NDVI is high, slope near zero, and previous slope was positive
ndvi_prev = ndvi.isel(time=slice(-6, -3))
fit_prev = ndvi_prev.polyfit(dim="time", deg=1)
slope_prev = fit_prev.polyfit_coefficients.sel(degree=1)
slope_prev = slope_prev * 1e9 * 60 * 60 * 24 * 8
peak_mask = (
    (ndvi_last > ndvi_thresh_peak)
    & (np.abs(slope) < slope_thresh)
    & (slope_prev > slope_thresh)
)
classes = classes.where(~peak_mask, 4)

raster_name = "ndvi_recent_trend.tif"
raster_path = Path(RASTER_LOCAL_DIR) / raster_name
classes.rio.to_raster(raster_path=raster_path, driver="COG")
s3.upload_file(raster_path, BUCKET_NAME, f"{RASTER_PREFIX}/{raster_name}")

pred_recent = preds.isel(time=slice(-3, None))
pred_last = pred_recent.isel(time=-1)

fit_pred = pred_recent.polyfit(dim="time", deg=1)
slope_pred = fit_pred.polyfit_coefficients.sel(degree=1)
slope_pred = slope_pred * 1e9 * 60 * 60 * 24 * 8

# Start with all zeros
classes = xr.full_like(pred_last, fill_value=0, dtype=np.int8)

# Bare/sparse vegetation
classes = classes.where(~(pred_last <= ndvi_thresh_sparse), 0)

# Stable (slope near zero, regardless of NDVI)
stable_mask = np.abs(slope_pred) < slope_thresh
classes = classes.where(~stable_mask, 1)

# Greening (NDVI > 0.1 and slope > 0)
greening_mask = (pred_last > ndvi_thresh_sparse) & (slope_pred > slope_thresh)
classes = classes.where(~greening_mask, 2)

# Browning/senescence (NDVI > 0.1 and slope < 0)
browning_mask = (pred_last > ndvi_thresh_sparse) & (slope_pred < -slope_thresh)
classes = classes.where(~browning_mask, 3)

# Peak growth (NDVI > 0.6 and slope near zero but preceded by positive slope)
# Here we check if NDVI is high, slope near zero, and previous slope was positive
peak_mask = (
    (pred_last > ndvi_thresh_peak)
    & (np.abs(slope_pred) < slope_thresh)
    & (slope > slope_thresh)
)
classes = classes.where(~peak_mask, 4)

raster_name = "ndvi_forecast_trend.tif"
raster_path = Path(RASTER_LOCAL_DIR) / raster_name
classes.rio.to_raster(raster_path=raster_path, driver="COG")
s3.upload_file(raster_path, BUCKET_NAME, f"{RASTER_PREFIX}/{raster_name}")
