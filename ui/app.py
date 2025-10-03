import fnmatch
import os
from datetime import datetime, timedelta

import boto3
import leafmap.foliumap as leafmap
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BUCKET_NAME = os.environ["S3_BUCKET"]


class VegetationHealthMonitor:
    def __init__(self):
        if "map_center" not in st.session_state:
            st.session_state.map_center = {
                "lat": 13.541243890565807,
                "lon": -2.430867732749294,
                "zoom": 10,
            }

        self.cogs = self.get_cogs()

    def list_s3_keys(self, bucket_name, prefix, pattern="*"):
        s3 = boto3.client("s3")
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        keys = []
        if "Contents" in response:
            for obj in response["Contents"]:
                key = obj["Key"]
                if fnmatch.fnmatch(key, pattern):
                    keys.append(key)

        return keys

    def extract_date_from_url(self, url):
        date_str = url.split("_")[-1].split(".")[0]
        date = datetime.strptime(date_str, "%Y-%m-%d").date()

        return date

    def get_cogs(self):
        ndvi_cog_keys = self.list_s3_keys(
            bucket_name=BUCKET_NAME, prefix="COG/", pattern="COG/ndvi_8d*.tif"
        )
        cogs = {}
        if ndvi_cog_keys:
            for key in ndvi_cog_keys:
                date = self.extract_date_from_url(key)
                cogs[date] = key

        return cogs

    def display(self):
        dates = self.cogs.keys()

        date_min = min(dates)
        date_max = max(dates)

        selected_date = st.slider(
            label="NDVI Dates",
            min_value=date_min,
            max_value=date_max,
            value=date_min,
            step=timedelta(days=8),
            format="YYYY-MM-DD",
        )
        cog_url = self.cogs[selected_date]
        s3_client = boto3.client("s3")
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": BUCKET_NAME,
                "Key": cog_url,
            },
            ExpiresIn=3600,
        )
        url2 = s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": BUCKET_NAME,
                "Key": "COG/ndvi_8d_forecast_2025-08-23.tif",
            },
            ExpiresIn=3600,
        )
        m = leafmap.Map()
        m.add_basemap("HYBRID")
        m.add_cog_layer(url=url, colormap_name="viridis", name="cog1", rescale=["-1,1"])
        m.add_cog_layer(
            url=url2, colormap_name="viridis", name="cog2", rescale=["-1,1"]
        )
        m.add_colormap(cmap="viridis", vmin=-1, vmax=1, position=(5, 5))
        m.to_streamlit(height=700)


if __name__ == "__main__":
    app = VegetationHealthMonitor()
    app.display()
