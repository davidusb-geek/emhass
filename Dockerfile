## EMHASS Docker
## Docker run addon testing example:
## docker build -t emhass .
## OR docker build --build-arg TARGETARCH=amd64 -t emhass .
## docker run --rm -it -p 5000:5000 --name emhass-container -v ./config.json:/share/config.json -v ./secrets_emhass.yaml:/app/secrets_emhass.yaml emhass

# armhf,amd64,armv7,aarch64
ARG TARGETARCH
# armhf=raspbian, amd64,armv7,aarch64=debian
ARG os_version=debian

FROM ghcr.io/home-assistant/$TARGETARCH-base-$os_version:bookworm AS base

# check if TARGETARCH was passed by build-arg
ARG TARGETARCH
ENV TARGETARCH=${TARGETARCH:?}

WORKDIR /app
COPY pyproject.toml /app/
COPY .python-version /app/
COPY gunicorn.conf.py /app/

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    # Basic utilities
    gnupg \
    curl \
    ca-certificates \
    # Numpy dependencies
    libgfortran5 \
    libopenblas0-pthread \
    libopenblas-dev \
    libatlas3-base \
    libatlas-base-dev \
    # h5py / tables dependencies
    libsz2 \
    libaec0 \
    libhdf5-hl-100 \
    libhdf5-103-1 \
    libhdf5-dev \
    libhdf5-serial-dev \
    # build packages
    gcc \
    g++ \
    patchelf \
    cmake \
    ninja-build \
    # Cleanup apt caches to reduce image size
    && rm -rf /var/cache/apt/* \
    && rm -rf /var/lib/apt/lists/* \
    # Install uv
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh \
    # Install python
    && uv python install \
    # Setup HDF5 symlinks
    && ln -s /usr/include/hdf5/serial /usr/include/hdf5/include \
    # Create directories
    && mkdir -p /data/ \
    && mkdir -p /share/emhass/

# copy required EMHASS files
COPY src/emhass/ /app/src/emhass/

# webserver files
COPY src/emhass/templates/ /app/src/emhass/templates/
COPY src/emhass/static/ /app/src/emhass/static/
COPY src/emhass/static/data/ /app/src/emhass/static/data/
COPY src/emhass/static/img/ /app/src/emhass/static/img/

# emhass extra packadge data
COPY src/emhass/data/ /app/src/emhass/data/

# pre generated optimization results
COPY data/opt_res_latest.csv /app/data/opt_res_latest.csv
COPY data/long_train_data.pkl /app/data/long_train_data.pkl
COPY README.md /app/
COPY pyproject.toml /app/

# secrets file (secrets_emhass.yaml) can be copied into the container with volume mounts with docker run
# options.json file will be automatically generated and passed from Home Assistant using the addon

#set python env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Docker Labels for hass
LABEL \
    io.hass.name="emhass" \
    io.hass.description="EMHASS: Energy Management for Home Assistant" \
    io.hass.version=${BUILD_VERSION} \
    io.hass.type="addon" \
    io.hass.arch="aarch64|amd64" \
    org.opencontainers.image.source="https://github.com/davidusb-geek/emhass" \
    org.opencontainers.image.description="EMHASS python package and requirements, in Home Assistant Debian container."

# Set up venv
RUN uv venv && . .venv/bin/activate

RUN [[ "${TARGETARCH}" == "aarch64" ]] && uv pip install --verbose ndindex || echo "libatomic1 cant be installed"

# install packages and build EMHASS
RUN uv pip install --verbose .
RUN uv lock

# remove build only packages
RUN apt-get remove --purge -y --auto-remove \
    gcc \
    g++ \
    patchelf \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Environment variables for flexibility
ENV WORKER_CLASS=uvicorn.workers.UvicornWorker
ENV PORT=5000
ENV IP=0.0.0.0

# Entrypoint script inline
ENTRYPOINT ["/bin/bash", "-c", "set -e && \
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