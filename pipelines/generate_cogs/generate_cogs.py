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


def calculate_slope(da):
    fit = da.polyfit(dim="time", deg=1)
    slope = fit.polyfit_coefficients.sel(degree=1)
    slope = (
        slope * 1e9 * 60 * 60 * 24 * 8
    )  # convert from slope based on nanoseconds to 8 day intervals
    return slope


def percent_missing(da):
    return da.isnull().sum(dim="time") / da.sizes["time"] * 100


def classify_trend(last_timestep, slope, percent_missing):
    # Thresholds
    ndvi_thresh_bare = 0.08  # based on typical NDVI for barren in this region
    ndvi_thresh_peak = 0.26  # 95th percentile NDVI cropland/grassland in this region
    slope_thresh = 0.001  # tolerance for "no change"
    percent_missing_thresh = 0.5

    # Start with all zeros. Represents bare/sparse vegetation
    classes = xr.full_like(last_timestep, fill_value=0, dtype=np.int8)

    # No change (NDVI > bare threshold slope near zero)
    no_change_mask = (last_timestep > ndvi_thresh_bare) & (np.abs(slope) < slope_thresh)
    classes = classes.where(~no_change_mask, 1)

    # Greening (NDVI > bare threshold and slope > 0)
    greening_mask = (last_timestep > ndvi_thresh_bare) & (slope > slope_thresh)
    classes = classes.where(~greening_mask, 2)

    # Browning/senescence (NDVI > bare threshold and slope < 0)
    browning_mask = (last_timestep > ndvi_thresh_bare) & (slope < -slope_thresh)
    classes = classes.where(~browning_mask, 3)

    # Peak growth (NDVI > peak growth threshold)
    peak_mask = last_timestep > ndvi_thresh_peak
    classes = classes.where(~peak_mask, 4)

    # Don't classify trend if too much missing data
    missing_mask = percent_missing > percent_missing_thresh
    classes = classes.where(~missing_mask, 5)

    return classes


fs = s3fs.S3FileSystem()

if fs.exists(f"{NDVI_ZARR_PATH}/zarr.json"):
    ndvi_store = s3fs.S3Map(root=NDVI_ZARR_PATH, s3=fs, check=False)
    ndvi = xr.open_zarr(ndvi_store)
    ndvi_processed = ndvi.ndvi_8d_processed
    ndvi_raw = ndvi.ndvi_8d_raw
else:
    raise FileNotFoundError(f"{NDVI_ZARR_PATH} was not found.")

if fs.exists(f"{PRED_ZARR_PATH}/zarr.json"):
    pred_store = s3fs.S3Map(root=PRED_ZARR_PATH, s3=fs, check=False)
    preds = xr.open_zarr(pred_store).ndvi_8d_forecast
else:
    raise FileNotFoundError(f"{PRED_ZARR_PATH} was not found.")

s3 = boto3.client("s3")

# Recent Trend
ndvi_recent = ndvi_processed.isel(time=slice(-4, -1))
ndvi_last = ndvi_recent.isel(time=-1)
ndvi_recent_raw = ndvi_raw.isel(time=slice(-4, -1))
recent_percent_missing = percent_missing(ndvi_recent_raw)

slope = calculate_slope(ndvi_recent)
recent_trend = classify_trend(ndvi_last, slope, recent_percent_missing)

raster_name = "ndvi_recent_trend.tif"
raster_path = Path(RASTER_LOCAL_DIR) / raster_name
recent_trend.rio.to_raster(raster_path=raster_path, driver="COG")
s3.upload_file(raster_path, BUCKET_NAME, f"{RASTER_PREFIX}/{raster_name}")
print("Created recent trend raster.")

# Forecasted Trend
pred_recent = preds.isel(time=slice(-3, None))
pred_last = pred_recent.isel(time=-1)

slope = calculate_slope(pred_recent)
forecast_trend = classify_trend(pred_last, slope, recent_percent_missing)

raster_name = "ndvi_forecast_trend.tif"
raster_path = Path(RASTER_LOCAL_DIR) / raster_name
forecast_trend.rio.to_raster(raster_path=raster_path, driver="COG")
s3.upload_file(raster_path, BUCKET_NAME, f"{RASTER_PREFIX}/{raster_name}")
print("Created forecasted trend raster.")
