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

    def generate_presigned_url(self, bucket, key):
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=st.secrets["aws_access_key_id"],
            aws_secret_access_key=st.secrets["aws_secret_access_key"],
            region_name=st.secrets["aws_region"],
        )
        return s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": bucket,
                "Key": key,
            },
            ExpiresIn=1200,
        )

    def display(self):
        st.title("Vegetation Health Monitor")
        st.write(
            """
            The Vegetation Health Monitor forecasts NDVI using Landsat imagery to show predicted trends in vegetation growth in the Sahel region of Africa,
            an area affected by drought and desertification.
            NDVI is widely used for monitoring the health and growth stages of ecosystems and crops, which is important for 
            understanding the effects of drought, desertification, and other natural or human-caused events. Forecasting NDVI allows for advanced operational planning and intervention.
            """
        )

        recent_trend_url = self.generate_presigned_url(
            BUCKET_NAME, "COG/ndvi_recent_trend.tif"
        )
        forecast_trend_url = self.generate_presigned_url(
            BUCKET_NAME, "COG/ndvi_forecast_trend.tif"
        )

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
            "5": "#ffffff",
        }
        m.add_cog_layer(
            url=recent_trend_url,
            colormap=custom_cmap,
            name="Recent Trend",
            fit_bounds=True,
            titiler_endpoint="https://titiler.xyz",
        )
        m.add_cog_layer(
            url=forecast_trend_url,
            colormap=custom_cmap,
            name="Forecast Trend",
            fit_bounds=True,
            titiler_endpoint="https://titiler.xyz",
        )
        m.add_legend(
            title="Vegetation Growth Trend",
            labels=[
                "Bare/Sparse Vegetation",
                "No Change",
                "Growing",
                "Senescence/Browning",
                "Peak Growth",
                "Insufficient Data",
            ],
            colors=[c for c in custom_cmap.values()],
        )
        m.to_streamlit(height=600)

        with st.sidebar:
            st.title("About the data")
            st.write(
                """
                This system is based on Landsat imagery and is updated every 8 days. 
                The map shows two NDVI trends: recent and forecasted. You can toggle the layers using the
                layer control in the top right corner of the map to see both of these trends. Recent Trend is
                based on data from the past 24 days and Forecast Trend uses 3 forecasted timesteps at 8 day intervals (24 days ahead).
                Thresholds based on historical data are also used in classifying each pixel. Forecasts are made using
                past NDVI values. Note that some areas shown as bare/sparse vegatation are actually water bodies.
                """
            )


if __name__ == "__main__":
    app = VegetationHealthMonitor()
    app.display()
