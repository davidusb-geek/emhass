## EMHASS Docker (Async Version)
## Docker run addon testing example:
## docker build -f Dockerfile.async -t emhass-async .
## OR docker build --build-arg TARGETARCH=amd64 -f Dockerfile.async -t emhass-async .
## docker run --rm -it -p 5000:5000 --name emhass-async-container -v ./config.json:/share/config.json -v ./secrets_emhass.yaml:/app/secrets_emhass.yaml emhass-async

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

RUN apt update \
    && apt install -y --no-install-recommends \
    # Numpy
    libgfortran5 \
    libopenblas0-pthread \
    libopenblas-dev \
    libatlas3-base \
    libatlas-base-dev \
    # h5py / tables
    libsz2 \
    libaec0 \
    libhdf5-hl-100 \
    libhdf5-103-1 \
    libhdf5-dev \
    libhdf5-serial-dev \
    # cbc
    coinor-cbc \
    coinor-libcbc-dev \
    # glpk
    glpk-utils \
    # build packages (just in case wheel does not exist)
    gcc \
    g++ \
    patchelf \
    cmake \
    ninja-build \
    && rm -rf /var/cache/apt/* \
    && rm -rf /var/lib/apt/lists/*

# Install uv (pip alternative)
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh
# Install python (version based on .python-version)
RUN uv python install

# specify hdf5
RUN ln -s /usr/include/hdf5/serial /usr/include/hdf5/include && export HDF5_DIR=/usr/include/hdf5

# make sure data directory exists
RUN mkdir -p /data/

# make sure emhass share directory exists
RUN mkdir -p /share/emhass/

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
COPY data/opt_res_latest.csv /data/
COPY data/long_train_data.pkl /data/
COPY README.md /app/
COPY pyproject.toml /app/

# secrets file (secrets_emhass.yaml) can be copied into the container with volume mounts with docker run
# options.json file will be automatically generated and passed from Home Assistant using the addon

#set python env variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Docker Labels for hass
LABEL \
    io.hass.name="emhass" \
    io.hass.description="EMHASS: Energy Management for Home Assistant (Async Version)" \
    io.hass.version=${BUILD_VERSION} \
    io.hass.type="addon" \
    io.hass.arch="aarch64|amd64" \
    org.opencontainers.image.source="https://github.com/davidusb-geek/emhass" \
    org.opencontainers.image.description="EMHASS python package and requirements, in Home Assistant Debian container (Async Version)."

# Set up venv
RUN uv venv && . .venv/bin/activate

RUN [[ "${TARGETARCH}" == "aarch64" ]] && uv pip install --verbose ndindex || echo "libatomic1 cant be installed"

# install packadges and build EMHASS
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

# Use async web server with Hypercorn instead of gunicorn
ENTRYPOINT [ "uv", "run", "--frozen", "emhass.web_server_async" ]

# for running Unittest
#COPY tests/ /app/tests
#RUN apt-get update &&  apt-get install python3-requests-mock -y
#COPY data/ /app/data/
#ENTRYPOINT ["uv","run","unittest","discover","-s","./tests","-p","test_*.py"]

# Example of 32 bit specific
# try, symlink apt cbc, to pulp cbc, in python directory (for 32bit)
#RUN [[ "${TARGETARCH}" == "armhf" || "${TARGETARCH}" == "armv7"  ]] &&  ln -sf /usr/bin/cbc /usr/local/lib/python3.11/dist-packages/pulp/solverdir/cbc/linux/32/cbc || echo "cbc symlink didnt work/not required"
