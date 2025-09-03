import json
import os
from datetime import date, datetime, time

import numpy as np
import pystac_client
import s3fs
import xarray as xr
from dotenv import load_dotenv
from odc.stac import load
from planetary_computer import sign_url
from shapely.geometry import shape

load_dotenv()
BUCKET_NAME = os.environ["S3_BUCKET"]
AOI_PATH = f"{BUCKET_NAME}/aoi.geojson"
ZARR_PATH = f"{BUCKET_NAME}/ndvi_processed.zarr"

fs = s3fs.S3FileSystem()

if fs.exists(AOI_PATH):
    with fs.open(AOI_PATH, "r") as file:
        area_of_interest = json.load(file)["features"][0]["geometry"]
        geom = shape(area_of_interest)
        bbox = tuple(geom.bounds)
else:
    raise FileNotFoundError(f"{AOI_PATH} was not found.")

if fs.exists(f"{ZARR_PATH}/zarr.json"):
    store = s3fs.S3Map(root=ZARR_PATH, s3=fs, check=False)
    ds = xr.open_zarr(store)
    start_date = ds.time[-1].values
    print(f"Last 5 dates in {ZARR_PATH} before update: {ds.time.values[-5:]}")
else:
    raise FileNotFoundError(f"{ZARR_PATH} was not found.")

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1/"
)
collection = "landsat-c2-l2"
bbox = bbox
end_date = datetime.combine(date.today(), time()).strftime("%Y-%m-%dT%H:%M:%SZ")

search = catalog.search(
    collections=[collection],
    bbox=bbox,
    datetime=f"{start_date}/{end_date}",
)
items = search.item_collection()

if len(items) == 0:
    print("STAC search returned no items.")
    exit()
else:
    unique_dates = sorted(
        set(
            datetime.fromisoformat(item.properties["datetime"])
            .date()
            .strftime("%Y-%m-%d")
            for item in items
        )
    )
    print(f"STAC search found {len(items)} items for dates {unique_dates}.")

new_data = load(
    items,
    bands=["red", "nir08", "qa_pixel"],
    bbox=bbox,
    chunks={"x": 2048, "y": 2048},
    resolution=300,
    groupby="solar_day",
    patch_url=sign_url,
)

# Mask out nodata and cloud pixels
# Bit 3 is cloud shadow, bit 4 is cloud, and bit 0 is nodata
mask_bits = 0b00011001

mask = (new_data.qa_pixel & mask_bits) != 0

new_data = new_data.where(~mask, other=np.nan).drop_vars("qa_pixel")

ndvi = (new_data.nir08 - new_data.red) / (new_data.nir08 + new_data.red)
new_data["ndvi"] = ndvi.clip(-1, 1)
new_data = new_data.drop_vars(["red", "nir08"])
new_data = new_data.compute()

# origin=start_date so that the previous last time window is recomputed with any new data
eight_day = new_data.ndvi.resample(time="8D", origin=start_date).max()

new_data = xr.Dataset(
    {
        "ndvi_8d_raw": eight_day,
        "ndvi_8d_processed": eight_day,
    }
)

overlap = new_data.sel(time=slice(start_date, start_date))

overlap.drop_vars(["y", "x", "spatial_ref"]).to_zarr(
    store, mode="a", region={"time": slice(-1, None)}
)
print(f"Dates updated: {overlap.time.values}")
new = new_data.isel(time=slice(1, None))
new.to_zarr(store, mode="a", append_dim="time")
print(f"Dates added: {new.time.values}")

# Fill the newly added data using the last filled date
ds = xr.open_zarr(store)
last_filled_date = start_date - np.timedelta64(8, "D")
filled = (
    ds["ndvi_8d_processed"]
    .sel(time=slice(last_filled_date, None))
    .bfill("time")
    .ffill("time")
)
filled.drop_vars(["y", "x", "spatial_ref"]).to_zarr(
    store, mode="a", region={"time": slice(len(ds.time) - len(filled.time), None)}
)
print(f"Dates filled: {filled.time.values}")
print("Finished adding data to zarr store.")
print(f"Last 5 dates in {ZARR_PATH} after update: {ds.time.values[-5:]}")
