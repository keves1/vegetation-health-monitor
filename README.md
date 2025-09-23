# Vegetation Health Monitor (work in progress)

The Vegetation Health Monitor forecasts NDVI using Landsat imagery to show predicted trends in vegetation health for an area in the Sahel region of Africa. NDVI is widely used for monitoring the health and growth stages of ecosystems and crops, which is important for understanding the effects of drought, desertification, and other natural or human-caused events. Forecasting NDVI allows for advanced operational planning and intervention.

This system contains the following components, each a separate containerized application orchestrated using AWS Step Functions.

- Data ingest: This pipeline runs every 8 days. It searches a STAC catalog for any new Landsat data in the specified region since the last run, downloads the data, masks clouds, computes NDVI, creates an 8 day maximum-value composite (MVC), fills missing data, and adds the result to a zarr store.
- Forecasting model: After new data is processed, the model uses prior NDVI values to predict NDVI for the next 3 timesteps, or 24 days into the future. 