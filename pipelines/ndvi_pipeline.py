import json
import os
from datetime import date, datetime, time

import pystac_client
from odc.stac import load
from planetary_computer import sign_url
from shapely.geometry import shape

# dask_client = DaskClient()

if os.path.exists("aoi.geojson"):
    with open("aoi.geojson", "r") as file:
        area_of_interest = json.load(file)["features"][0]["geometry"]
        geom = shape(area_of_interest)
        bbox = tuple(geom.bounds)
else:
    raise FileNotFoundError("aoi.geojson was not found.")

if os.path.exists("last_end_date.txt"):
    with open("last_end_date.txt", "r") as file:
        start_date = file.readline().strip()
else:
    raise FileNotFoundError("last_end_date.txt was not found.")

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1/"
)
collection = "modis-09Q1-061"
bbox = bbox
end_date = datetime.combine(date.today(), time()).strftime("%Y-%m-%dT%H:%M:%SZ")

with open("last_end_date.txt", "w") as file:
    file.write(end_date)

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

zarr_store_path = "data/ndvi_processed.zarr"
if os.path.exists(zarr_store_path):
    data.to_zarr(zarr_store_path, mode="a", append_dim="time")
else:
    raise FileNotFoundError(f"{zarr_store_path} was not found.")
