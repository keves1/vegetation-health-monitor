# Vegetation Health Monitor

The Vegetation Health Monitor forecasts NDVI using 250 m resolution MODIS imagery to show predicted trends in vegetation health for an area in the Sahel region of Africa. NDVI is widely used for monitoring the health and growth stages of ecosystems and crops, which is important for understanding the effects of drought, desertification, and other natural or human-caused events. Forecasting NDVI allows for advanced operational planning and intervention.

This system contains the following components:

- NDVI pipeline: Every 8 days, this pipeline runs in AWS Batch. It searches a STAC catalog for any new MODIS data in the specified region since the last run, downloads the data, masks clouds and other low-quality pixels, computes NDVI, fills missing data, and appends the result to a zarr store.
- Forecasting model: After new data is processed, the model will use prior NDVI values to predict NDVI for the next 3 timesteps, or 24 days into the future.  