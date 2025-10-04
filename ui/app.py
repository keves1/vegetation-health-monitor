import os

import boto3
import leafmap.foliumap as leafmap
from dotenv import load_dotenv

load_dotenv()
BUCKET_NAME = os.environ["S3_BUCKET"]


class VegetationHealthMonitor:
    def display(self):
        s3_client = boto3.client("s3")
        recent_trend_url = s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": BUCKET_NAME,
                "Key": "COG/ndvi_recent_trend.tif",
            },
            ExpiresIn=3600,
        )
        forecast_trend_url = s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": BUCKET_NAME,
                "Key": "COG/ndvi_forecast_trend.tif",
            },
            ExpiresIn=3600,
        )
        m = leafmap.Map(
            center=[13.541243890565807, -2.430867732749294],
            zoom=10,
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
        m.to_streamlit(height=600)


if __name__ == "__main__":
    app = VegetationHealthMonitor()
    app.display()
