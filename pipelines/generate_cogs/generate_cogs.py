import os
from pathlib import Path

import boto3
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
    ndvi = xr.open_zarr(ndvi_store)
else:
    raise FileNotFoundError(f"{NDVI_ZARR_PATH} was not found.")

if fs.exists(f"{PRED_ZARR_PATH}/zarr.json"):
    pred_store = s3fs.S3Map(root=PRED_ZARR_PATH, s3=fs, check=False)
    preds = xr.open_zarr(pred_store)
else:
    raise FileNotFoundError(f"{PRED_ZARR_PATH} was not found.")

for i in range(NUM_PAST_STEPS):
    img = ndvi.ndvi_8d_processed.isel(time=-(i + 1))
    date = str(img["time"].values.astype("datetime64[D]"))
    raster_name = f"ndvi_8d_processed_{date}.tif"
    raster_path = Path(RASTER_LOCAL_DIR) / raster_name
    img.rio.to_raster(raster_path=raster_path, driver="COG")
    s3 = boto3.client("s3")
    s3.upload_file(raster_path, BUCKET_NAME, f"{RASTER_PREFIX}/{raster_name}")

for i in range(NUM_FUTURE_STEPS):
    img = preds.ndvi_8d_forecast.isel(time=-(i + 1))
    date = str(img["time"].values.astype("datetime64[D]"))
    raster_name = f"ndvi_8d_forecast_{date}.tif"
    raster_path = Path(RASTER_LOCAL_DIR) / raster_name
    img.rio.to_raster(raster_path=raster_path, driver="COG")
    s3 = boto3.client("s3")
    s3.upload_file(raster_path, BUCKET_NAME, f"{RASTER_PREFIX}/{raster_name}")
