FROM python:3.11-slim

# Install git
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory.
WORKDIR /app

# Install the application dependencies.
COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-cache

# Copy the application into the container.
COPY inference_pipeline.py ./

CMD ["uv", "run", "python", "inference_pipeline.py"]