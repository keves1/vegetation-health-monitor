import numpy as np
import pandas as pd
import pytest
import xarray as xr

from inference_pipeline import update_zarr_store


@pytest.fixture
def zarr_store(tmp_path):
    return tmp_path / "test.zarr"


@pytest.fixture
def base_dataset():
    nx, ny, nt = 4, 3, 3

    x = np.arange(nx)
    y = np.arange(ny)
    t = pd.to_datetime(["2020-01-01", "2020-01-09", "2020-01-17"])

    data = np.random.rand(ny, nx, nt)

    # Build dataset
    ds = xr.Dataset(
        {"data_var": (("y", "x", "time"), data)}, coords={"x": x, "y": y, "time": t}
    )
    return ds


@pytest.fixture
def initialized_store(zarr_store, base_dataset):
    base_dataset.to_zarr(zarr_store, mode="w")
    return zarr_store


def test_overlap_dates(initialized_store):
    store = initialized_store
    nx, ny, nt = 4, 3, 3
    x = np.arange(nx)
    y = np.arange(ny)
    t = pd.to_datetime(["2020-01-09", "2020-01-17", "2020-01-25"])
    data = np.random.rand(ny, nx, nt)
    new_data = xr.DataArray(data, coords={"y": y, "x": x, "time": t})

    update_zarr_store(store, new_data)


def test_non_overlap_dates(initialized_store):
    store = initialized_store
    nx, ny, nt = 4, 3, 2
    x = np.arange(nx)
    y = np.arange(ny)
    t = pd.to_datetime(["2020-01-25", "2020-02-02"])
    data = np.random.rand(ny, nx, nt)
    new_data = xr.DataArray(data, coords={"y": y, "x": x, "time": t})

    update_zarr_store(store, new_data)
