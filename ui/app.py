import os

import boto3
import leafmap.foliumap as leafmap
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BUCKET_NAME = os.environ["S3_BUCKET"]


class VegetationHealthMonitor:
    def __init__(self):
        st.set_page_config(layout="wide", page_title="Vegetation Health Monitor")

    def generate_presigned_url(self, key):
        s3_client = boto3.client("s3")
        return s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": BUCKET_NAME,
                "Key": key,
            },
            ExpiresIn=3600,
        )

    def display(self):
        st.title("Vegetation Health Monitor")
        st.write(
            "The Vegetation Health Monitor forecasts NDVI using Landsat imagery to show predicted trends in vegetation health. "
            "NDVI is widely used for monitoring the health and growth stages of ecosystems and crops, which is important for "
            "understanding the effects of drought, desertification, and other natural or human-caused events. Forecasting NDVI allows for advanced operational planning and intervention."
        )

        recent_trend_url = self.generate_presigned_url("COG/ndvi_recent_trend.tif")
        forecast_trend_url = self.generate_presigned_url("COG/ndvi_forecast_trend.tif")

        m = leafmap.Map(
            center=[13.541243890565807, -2.430867732749294],
            zoom=10,
            draw_control=False,
            search_control=False,
        )
        m.add_basemap("HYBRID")
        custom_cmap = {
            "0": "#8b6f47",
            "1": "#a9a9a9",
            "2": "#45e640",
            "3": "#dc4133",
            "4": "#ffd700",
        }
        m.add_cog_layer(
            url=recent_trend_url,
            colormap=custom_cmap,
            name="Recent Trend",
            fit_bounds=True,
        )
        m.add_cog_layer(
            url=forecast_trend_url,
            colormap=custom_cmap,
            name="Forecast Trend",
            fit_bounds=True,
        )
        m.add_vector(
            filename="Burkina_Faso_ADM2_simplified.simplified.geojson",
            zoom_to_layer=False,
        )
        m.add_vector(
            filename="Mali_ADM2_simplified.simplified.geojson",
            zoom_to_layer=False,
        )
        m.add_legend(
            title="Vegetation Growth Trend",
            labels=[
                "Bare/Sparse Vegetation",
                "No Change",
                "Green-up",
                "Senescence/Browning",
                "Peak Growth",
            ],
            colors=[
                "#8b6f47",
                "#a9a9a9",
                "#45e640",
                "#dc4133",
                "#ffd700",
            ],
        )
        m.to_streamlit(height=600)


if __name__ == "__main__":
    app = VegetationHealthMonitor()
    app.display()
