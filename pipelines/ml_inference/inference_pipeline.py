import os

import boto3
import numpy as np
import pandas as pd
import s3fs
import torch
import xarray as xr
from botocore.exceptions import ClientError
from dask.diagnostics import ProgressBar
from dask.distributed import Client, get_worker
from dotenv import load_dotenv

load_dotenv()
BUCKET_NAME = os.environ["S3_BUCKET"]
NDVI_ZARR_PATH = f"{BUCKET_NAME}/ndvi_processed.zarr"
PRED_ZARR_PATH = f"{BUCKET_NAME}/ndvi_predictions.zarr"
MODEL_KEY = os.environ["MODEL_KEY"]
MODEL_LOCAL_PATH = "/tmp/model.pt"
NUM_FUTURE_STEPS = 3
NUM_PAST_STEPS = 10


def load_model_on_worker(model_path, dask_worker, device="cpu"):
    if not hasattr(dask_worker, "model"):
        dask_worker.model = torch.jit.load(model_path, map_location=device)
        dask_worker.model.eval()


def forecast(chunk):
    worker = get_worker()
    y_chunk, x_chunk = chunk.shape[0], chunk.shape[1]  # chunk shape: [y, x, time]
    stacked = chunk.reshape(-1, chunk.shape[2])
    past_steps = torch.tensor(stacked[..., np.newaxis], dtype=torch.float32)
    mean = past_steps.mean(dim=1, keepdim=True)
    std = past_steps.std(dim=1, keepdim=True)
    past_steps_normalized = (past_steps - mean) / (std + 1e-12)
    output = worker.model(past_steps_normalized)
    output_denormalized = (output * std + mean).squeeze(-1)
    forecast_steps = output_denormalized.shape[1]
    return (
        output_denormalized.reshape(y_chunk, x_chunk, forecast_steps).detach().numpy()
    )


def index_slice_for_time(ds, start, end):
    idx = ds.get_index("time")
    i0 = idx.get_loc(start)
    i1 = idx.get_loc(end)
    return slice(i0, i1 + 1)


def main():
    client = Client()
    print(client)

    fs = s3fs.S3FileSystem()

    if fs.exists(f"{NDVI_ZARR_PATH}/zarr.json"):
        ndvi_store = s3fs.S3Map(root=NDVI_ZARR_PATH, s3=fs, check=False)
        ndvi = xr.open_zarr(ndvi_store)
        print(f"Last date in {NDVI_ZARR_PATH}: {ndvi.time.values[-1]}")
    else:
        raise FileNotFoundError(f"{NDVI_ZARR_PATH} was not found.")

    pred_zarr_exists = False
    if fs.exists(f"{PRED_ZARR_PATH}/zarr.json"):
        pred_store = s3fs.S3Map(root=PRED_ZARR_PATH, s3=fs, check=False)
        preds = xr.open_zarr(pred_store)
        pred_zarr_exists = True
    else:
        print(f"{PRED_ZARR_PATH} was not found, creating new store.")
        pred_store = s3fs.S3Map(root=PRED_ZARR_PATH, s3=fs, check=False)

    s3 = boto3.client("s3")
    try:
        s3.download_file(BUCKET_NAME, MODEL_KEY, MODEL_LOCAL_PATH)
        print("Model file download succeeded.")
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404" or error_code == "NoSuchKey":
            print(f"File not found: s3://{BUCKET_NAME}/{MODEL_KEY}")
        else:
            print(f"Failed to download file: {e}")
            raise

    client.run(load_model_on_worker, model_path=MODEL_LOCAL_PATH)

    lookback = ndvi["ndvi_8d_processed"].isel(time=slice(-NUM_PAST_STEPS, None))
    lookback = lookback.chunk({"x": 10, "y": 10, "time": NUM_PAST_STEPS})
    print("Loaded lookback data.")

    forecast_da = xr.apply_ufunc(
        forecast,
        lookback,
        input_core_dims=[["time"]],
        output_core_dims=[["forecast_time"]],
        output_sizes={"forecast_time": NUM_FUTURE_STEPS},
        output_dtypes=[np.float32],
        dask="parallelized",
    )
    forecast_da = forecast_da.rename("ndvi_8d_forecast")
    forecast_da = forecast_da.rename({"forecast_time": "time"}).transpose(
        "time", "y", "x"
    )
    last_date = ndvi.time.values[-1]
    forecast_dates = pd.date_range(
        start=last_date, periods=NUM_FUTURE_STEPS + 1, freq="8D"
    )[1:]
    forecast_da = forecast_da.assign_coords(time=forecast_dates)

    print("Computing forecast...")
    with ProgressBar():
        forecast_da = forecast_da.compute()
    print(f"Predicted NDVI for {NUM_FUTURE_STEPS} steps ahead.")

    if pred_zarr_exists:
        overlap_dates = np.intersect1d(preds["time"].values, forecast_da["time"].values)
        updated_preds = forecast_da.sel(time=overlap_dates)
        updated_slice = index_slice_for_time(
            preds, overlap_dates.min(), overlap_dates.max()
        )
        updated_preds.drop_vars(["y", "x", "spatial_ref"]).to_zarr(
            pred_store,
            mode="a",
            region={"time": updated_slice},
        )
        print(f"Dates updated: {updated_preds['time'].values}")
        new_preds = forecast_da.sel(time=~forecast_da["time"].isin(overlap_dates))
        new_preds.to_zarr(pred_store, mode="a", append_dim="time")
        print(f"Dates added: {new_preds['time'].values}")
    else:
        forecast_da.to_zarr(pred_store, mode="w", consolidated=True)
        print(f"Dates added: {forecast_da['time'].values}")
    print("Finished adding predictions to zarr store.")


if __name__ == "__main__":
    main()
