## EMHASS Docker 
## Docker run ADD-ON testing example: 
## docker build -t emhass/docker --build-arg build_version=addon-local .
## docker run -it -p 5000:5000 --name emhass-container -e LAT="45.83" -e LON="6.86" -e ALT="4807.8" -e TIME_ZONE="Europe/Paris" emhass/docker --url YOURHAURLHERE --key YOURHAKEYHERE
##
## Docker run Standalone example:
## docker build -t emhass/docker --build-arg build_version=standalone .
## docker run -it -p 5000:5000 --name emhass-container -v $(pwd)/config_emhass.yaml:/app/config_emhass.yaml -v $(pwd)/secrets_emhass.yaml:/app/secrets_emhass.yaml emhass/docker 

#build_version options are: addon, addon-pip, addon-git, addon-local, standalone (default)
ARG build_version=standalone

FROM ghcr.io/home-assistant/$TARGETARCH-base-debian:bookworm AS base

WORKDIR /app
COPY requirements.txt /app/

# Setup
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libffi-dev \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    gcc \
    coinor-cbc \
    coinor-libcbc-dev \
    libglpk-dev \
    glpk-utils \
    libhdf5-dev \
    libhdf5-serial-dev \
    netcdf-bin \
    libnetcdf-dev \
    pkg-config \
    gfortran \
    libatlas-base-dev \
    && ln -s /usr/include/hdf5/serial /usr/include/hdf5/include \
    && export HDF5_DIR=/usr/include/hdf5 \
    && pip3 install --extra-index-url=https://www.piwheels.org/simple --no-cache-dir --break-system-packages -U setuptools wheel \
    && pip3 install --extra-index-url=https://www.piwheels.org/simple --no-cache-dir --break-system-packages -r requirements.txt \
    && apt-get purge -y --auto-remove \
    gcc \
    build-essential \
    libhdf5-dev \
    libhdf5-serial-dev \
    pkg-config \
    gfortran \
    && rm -rf /var/lib/apt/lists/*


#copy config file (on all builds)
COPY config_emhass.yaml /app/

# Make sure data directory exists
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

ENTRYPOINT [ "python3", "-m", "emhass.web_server","--addon", "True", "--url", "http://supervisor/core/api"]

#-----------
#EMHASS-ADD-ON Testing with pip emhass (closest testing reference) 
FROM addon as addon-pip
#set build arg for pip version
ARG build_pip_version=""
RUN pip3 install --no-cache-dir --break-system-packages --upgrade --upgrade-strategy=only-if-needed -U emhass${build_pip_version}

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
RUN python3 -m pip install --no-cache-dir --break-system-packages -U  .
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
RUN python3 -m pip install --no-cache-dir --break-system-packages -U  .
ENTRYPOINT [ "python3", "-m", "emhass.web_server","--addon", "True" , "--no_response", "True"]

#-------------------------
#EMHASS STANDALONE
FROM base as standalone

RUN pip3 install --extra-index-url=https://www.piwheels.org/simple --no-cache-dir --break-system-packages -U  flask waitress plotly

COPY src/emhass/ /app/src/emhass/ 
COPY src/emhass/templates/ /app/src/emhass/templates/
COPY src/emhass/static/ /app/src/emhass/static/
COPY src/emhass/static/img/ /app/src/emhass/static/img/
COPY data/opt_res_latest.csv /app/data/
COPY README.md /app/
COPY setup.py /app/
#secrets file will need to be copied manually at docker run

# # set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

#build EMHASS
# RUN python3 setup.py install
RUN python3 -m pip install --no-cache-dir --break-system-packages -U  .
ENTRYPOINT [ "python3", "-m", "emhass.web_server"]
#-------------------------


#check build arguments and build
FROM ${build_version} AS final