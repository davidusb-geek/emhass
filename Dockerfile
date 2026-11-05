## EMHASS Docker
## Docker run app testing example:
## docker build -t emhass .
## OR docker build --build-arg TARGETARCH=amd64 -t emhass .
## docker run --rm -it -p 5000:5000 --name emhass-container -v ./config.json:/share/config.json -v ./secrets_emhass.yaml:/app/secrets_emhass.yaml emhass

# armhf,amd64,armv7,aarch64
ARG TARGETARCH
# armhf=raspbian, amd64,armv7,aarch64=debian
ARG os_version=debian

FROM ghcr.io/home-assistant/$TARGETARCH-base-$os_version:bookworm AS base

# Check if TARGETARCH was passed by build-arg
ARG TARGETARCH
ENV TARGETARCH=${TARGETARCH:?}

# Set Python and uv environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    WORKER_CLASS=uvicorn.workers.UvicornWorker \
    PORT=5000 \
    IP=0.0.0.0

WORKDIR /app

# Copy uv binaries directly from official image (faster and cleaner than curl installer)
COPY --from=ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 /uv /uvx /usr/local/bin/

# 1. Install ONLY persistent runtime shared libraries in a cached layer
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
        # Numpy / Scipy runtime dependencies
        libgfortran5 \
        libopenblas0-pthread \
        libatlas3-base \
        # HDF5 / PyTables runtime dependencies
        libsz2 \
        libaec0 \
        libhdf5-hl-100 \
        libhdf5-103-1 \
    && mkdir -p /data/ /share/emhass/ \
    && rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# 2. Install Python and create virtualenv (cached unless .python-version changes)
COPY .python-version /app/
RUN uv python install && uv venv

# 3. Copy project files (allowed via !src/emhass/ and !src/emhass/** in .dockerignore)
COPY pyproject.toml README.md gunicorn.conf.py /app/
COPY src/emhass/ /app/src/emhass/
COPY data/opt_res_latest.csv /app/data/opt_res_latest.csv
COPY data/long_train_data.pkl /app/data/long_train_data.pkl

# 4. Install build-only packages, generate uv.lock, sync environment from lock, and purge compilers in ONE layer
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        patchelf \
        cmake \
        ninja-build \
        libopenblas-dev \
        libatlas-base-dev \
        libhdf5-dev \
        libhdf5-serial-dev \
    && ln -sf /usr/include/hdf5/serial /usr/include/hdf5/include \
    && uv lock \
    && uv sync --frozen --verbose \
    && apt-get remove --purge -y --auto-remove \
        gcc \
        g++ \
        patchelf \
        cmake \
        ninja-build \
        libopenblas-dev \
        libatlas-base-dev \
        libhdf5-dev \
        libhdf5-serial-dev \
    && rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# 5. Declare BUILD_VERSION and labels last so version bumps don't bust the build cache
ARG build_version
ARG BUILD_VERSION=${build_version}
LABEL \
    io.hass.name="emhass" \
    io.hass.description="EMHASS: Energy Management for Home Assistant" \
    io.hass.version="${BUILD_VERSION}" \
    io.hass.type="addon" \
    io.hass.arch="aarch64|amd64" \
    org.opencontainers.image.source="https://github.com/davidusb-geek/emhass" \
    org.opencontainers.image.description="EMHASS python package and requirements, in Home Assistant Debian container."

# Entrypoint script inline
ENTRYPOINT ["/usr/bin/tini", "--", "/bin/bash", "-c", "set -e && \
if [ ! -f /data/long_train_data.pkl ]; then \
    echo 'Initializing data: Copying default PKL file...'; \
    cp /app/data/long_train_data.pkl /data/; \
fi && \
if [ ! -f /data/opt_res_latest.csv ]; then \
    echo 'Initializing data: Copying default CSV file...'; \
    cp /app/data/opt_res_latest.csv /data/; \
fi && \
WORKER_CLASS=${WORKER_CLASS:-uvicorn.workers.UvicornWorker} && \
PORT=${PORT:-5000} && \
IP=${IP:-0.0.0.0} && \
exec uv run --frozen gunicorn emhass.web_server:app -c gunicorn.conf.py -k \"$WORKER_CLASS\""]