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
    # # cbc
    coinor-cbc \
    coinor-libcbc-dev

    # libffi-dev \
    # gfortran \
    # netcdf-bin \
    # libnetcdf-dev \
    # libglpk-dev \
    # glpk-utils \
    # libatlas3-base \
    # libatlas-base-dev \
    # libopenblas-dev \
    # libopenblas0-pthread \
    # libgfortran5 \

# add build packadges (just in case wheel does not exist)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    gcc \
    patchelf \
    cmake \
    ninja-build

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
COPY data/data_train_load_forecast.pkl /data/
COPY data/data_train_load_clustering.pkl /data/
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


# install packadges and build EMHASS
RUN uv pip install --verbose .
RUN uv lock

# remove build only packages
RUN apt-get remove --purge -y --auto-remove \
    gcc \
    patchelf \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT [ "uv", "run", "gunicorn", "emhass.web_server:'create_app()'" ]

# for running Unittest
#COPY tests/ /app/tests
#RUN apt-get update &&  apt-get install python3-requests-mock -y
#COPY data/ /app/data/
#ENTRYPOINT ["uv","run","unittest","discover","-s","./tests","-p","test_*.py"]

# Example of 32 bit specific 
# try, symlink apt cbc, to pulp cbc, in python directory (for 32bit)
#RUN [[ "${TARGETARCH}" == "armhf" || "${TARGETARCH}" == "armv7"  ]] &&  ln -sf /usr/bin/cbc /usr/local/lib/python3.11/dist-packages/pulp/solverdir/cbc/linux/32/cbc || echo "cbc symlink didnt work/not required"
