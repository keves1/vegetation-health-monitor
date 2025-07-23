import json
import os
from datetime import date, datetime, time

import pystac_client
import s3fs
from dotenv import load_dotenv
from odc.stac import load
from planetary_computer import sign_url
from shapely.geometry import shape

load_dotenv()
BUCKET_NAME = os.environ["S3_BUCKET"]
AOI_PATH = f"{BUCKET_NAME}/aoi.geojson"
END_DATE_PATH = f"{BUCKET_NAME}/last_end_date.txt"
ZARR_PATH = f"{BUCKET_NAME}/ndvi_processed.zarr"

# dask_client = DaskClient()

fs = s3fs.S3FileSystem()

if fs.exists(AOI_PATH):
    with fs.open(AOI_PATH, "r") as file:
        area_of_interest = json.load(file)["features"][0]["geometry"]
        geom = shape(area_of_interest)
        bbox = tuple(geom.bounds)
else:
    raise FileNotFoundError(f"{AOI_PATH} was not found.")

if fs.exists(END_DATE_PATH):
    with fs.open(END_DATE_PATH, "r") as file:
        start_date = file.readline().strip()
else:
    raise FileNotFoundError(f"{END_DATE_PATH} was not found.")

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1/"
)
collection = "modis-09Q1-061"
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
            datetime.fromisoformat(item.properties["end_datetime"])
            .date()
            .strftime("%Y-%m-%d")
            for item in items
        )
    )
    print(f"STAC search found {len(items)} items for dates {unique_dates}.")
    with fs.open(END_DATE_PATH, "w") as file:
        file.write(end_date)

data = load(
    items,
    bands=["red", "nir08", "sur_refl_qc_250m"],
    bbox=bbox,
    chunks={"x": 2048, "y": 2048},
    groupby="solar_day",
    patch_url=sign_url,
)

# Mask out low-quality pixels
mask = data.sur_refl_qc_250m.where((data.sur_refl_qc_250m & 0b11110000) == 0)
data = data.where(mask)

ndvi = (data.nir08 - data.red) / (data.nir08 + data.red)
data["ndvi"] = ndvi.clip(-1, 1)
data = data.drop_vars(["red", "nir08", "sur_refl_qc_250m"])
data = data.compute()

if fs.exists(f"{ZARR_PATH}/zarr.json"):
    store = s3fs.S3Map(root=ZARR_PATH, s3=fs, check=False)
    data.to_zarr(store, mode="a", append_dim="time")
    print("New data added to zarr store.")
else:
    raise FileNotFoundError(f"{ZARR_PATH} was not found.")

fs.close()
