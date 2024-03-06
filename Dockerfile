## EMHASS Docker
## Docker run ADD-ON testing example: 
    ## docker build -t emhass/docker --build-arg build_version=addon-local .
    ## docker run -it -p 5000:5000 --name emhass-container -e LAT="45.83" -e LON="6.86" -e ALT="4807.8" -e TIME_ZONE="Europe/Paris" emhass/docker --url YOURHAURLHERE --key YOURHAKEYHERE
## Docker run Standalone example:
    ## docker build -t emhass/docker --build-arg build_version=standalone .
    ## docker run -it -p 5000:5000 --name emhass-container -v $(pwd)/config_emhass.yaml:/app/config_emhass.yaml -v $(pwd)/secrets_emhass.yaml:/app/secrets_emhass.yaml emhass/docker 

#build_version options are: addon, addon-pip, addon-git, addon-local, standalone (default)
ARG build_version=standalone 
#os_version 
#armhf = raspbian
#amd64, armv7, aarch64 = debian
ARG os_version=debian 

FROM ghcr.io/home-assistant/$TARGETARCH-base-$os_version:bookworm AS base

#check if TARGETARCH passed by build-arg
ARG TARGETARCH
ENV TARGETARCH=${TARGETARCH:?}

WORKDIR /app
COPY requirements.txt /app/

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libffi-dev \
    python3 \
    python3-pip \
    python3-dev \
    git \
    gcc \
    patchelf \
    cmake \
    meson \
    ninja-build \
    build-essential \
    libhdf5-dev \
    libhdf5-serial-dev \
    pkg-config \
    gfortran \
    netcdf-bin \
    libnetcdf-dev \
    coinor-cbc \
    coinor-libcbc-dev \
    libglpk-dev \
    glpk-utils \
    libatlas-base-dev \
    libopenblas-dev 
RUN ln -s /usr/include/hdf5/serial /usr/include/hdf5/include 
RUN export HDF5_DIR=/usr/include/hdf5 

#install packages from pip, use piwheels if arm
RUN [[ "${TARGETARCH}" == "armhf" || "${TARGETARCH}" == "armv7" ]] &&  pip3 install --index-url=https://www.piwheels.org/simple --no-cache-dir --break-system-packages -r requirements.txt ||  pip3 install --no-cache-dir --break-system-packages -r requirements.txt

#remove build only packages
RUN apt-get purge -y --auto-remove \
    git \
    gcc \
    patchelf \
    cmake \
    meson \
    ninja-build \
    build-essential \
    pkg-config \
    gfortran \
    netcdf-bin \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*


#try symlink cbc to python directory
RUN ln -sf /usr/bin/cbc /usr/local/lib/python3.11/dist-packages/pulp/solverdir/cbc/linux/32/cbc || echo "cbc link didnt work"

#copy config file (on all builds)
COPY config_emhass.yaml /app/

#make sure data directory exists
RUN mkdir -p /app/data/

#-------------------------
#EMHASS-ADDON Default (this has no emhass packadge)
FROM base as addon

LABEL \
    io.hass.name="emhass" \
    io.hass.description="EMHASS: Energy Management for Home Assistant" \
    io.hass.version=${BUILD_VERSION} \
    io.hass.type="addon" \
    io.hass.arch="aarch64|amd64|armhf|armv7"

#-----------
#EMHASS-ADD-ON Testing with pip emhass (closest testing reference) 
FROM addon as addon-pip
#set build arg for pip version
ARG build_pip_version=""
RUN pip3 install --no-cache-dir --break-system-packages --upgrade --force-reinstall --no-deps --upgrade-strategy=only-if-needed -U emhass${build_pip_version}

COPY options.json /app/

ENTRYPOINT [ "python3", "-m", "emhass.web_server","--addon", "True", "--no_response", "True"]

#-----------
#EMHASS-ADD-ON Testing from local files
FROM addon as addon-local
COPY src/emhass/ /app/src/emhass/ 
COPY src/emhass/templates/ /app/src/emhass/templates/
COPY src/emhass/static/ /app/src/emhass/static/
COPY src/emhass/static/img/ /app/src/emhass/static/img/
COPY data/opt_res_latest.csv /app/data/
#add options.json, this otherwise would be generated via HA
COPY options.json /app/
COPY README.md /app/
COPY setup.py /app/
#compile EMHASS locally
RUN pip3 install --no-cache-dir --break-system-packages --no-deps --force-reinstall  .
ENTRYPOINT [ "python3", "-m", "emhass.web_server","--addon", "True" , "--no_response", "True"]


#-----------
#EMHASS-ADD-ON testing with git
FROM addon as addon-git
ARG build_repo=https://github.com/davidusb-geek/emhass.git
ARG build_branch=master
WORKDIR /tmp/
#Repo
RUN git clone $build_repo
WORKDIR /tmp/emhass
#Branch
RUN git checkout $build_branch
RUN mkdir -p /app/src/emhass/
RUN cp -r /tmp/emhass/src/emhass/. /app/src/emhass/
RUN cp /tmp/emhass/data/opt_res_latest.csv  /app/data/
RUN cp /tmp/emhass/setup.py /app/
RUN cp /tmp/emhass/README.md /app/
#add options.json, this otherwise would be generated via HA
RUN cp /tmp/emhass/options.json /app/
WORKDIR /app
RUN pip3 install --no-cache-dir --break-system-packages --no-deps --force-reinstall  .
ENTRYPOINT [ "python3", "-m", "emhass.web_server","--addon", "True" , "--no_response", "True"]

#-------------------------
#EMHASS STANDALONE
FROM base as standalone

COPY src/emhass/ /app/src/emhass/ 
COPY src/emhass/templates/ /app/src/emhass/templates/
COPY src/emhass/static/ /app/src/emhass/static/
COPY src/emhass/static/img/ /app/src/emhass/static/img/
COPY data/opt_res_latest.csv /app/data/
COPY README.md /app/
COPY setup.py /app/
#secrets file will need to be copied manually at docker run

#set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

#build EMHASS
RUN pip3 install --no-cache-dir --break-system-packages --no-deps --force-reinstall  .
ENTRYPOINT [ "python3", "-m", "emhass.web_server"]
#-------------------------


#check build arguments and build
FROM ${build_version} AS final