FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libasound-dev \
    libportaudio2 \
    portaudio19-dev \
    curl ca-certificates \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download and install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Copy project files needed for dependency resolution
COPY ./.python-version ./.python-version
COPY ./pyproject.toml ./pyproject.toml
COPY ./uv.lock ./uv.lock

# Copy the entrypoint script
COPY ./docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Copy the rest of the application
COPY . .

# Set environment variables for cache directories (can be overridden)
ENV CACHE_DIR=/cache
ENV UV_CACHE_DIR=/cache/uv
ENV MODELS_CACHE_DIR=/cache/models
ENV VENV_CACHE_DIR=/cache/venv

# Place venv executables at the front of the path (will be symlinked at runtime)
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7865

ENTRYPOINT ["/docker-entrypoint.sh"]
