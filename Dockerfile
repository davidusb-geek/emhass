## EMHASS Docker 
## Docker run ADD-ON testing example: 
## docker build -t emhass/docker --build-arg build_version=addon-local .
## docker run -it -p 5000:5000 --name emhass-container -e LAT="45.83" -e LON="6.86" -e ALT="4807.8" emhass/docker --url YOURHAURLHERE --key YOURHAKEYHERE
##
## Docker run Standalone example 
## docker build -t emhass/docker --build-arg build_version=standalone .
## docker run -it -p 5000:5000 --name emhass-container -v $(pwd)/config_emhass.yaml:/data/config_emhass.yaml -v $(pwd)/secrets_emhass.yaml:/data/secrets_emhass.yaml emhass/docker 

#build_version options are: addon, addon-pip, addon-git, addon-local standalone
ARG build_version

FROM ghcr.io/home-assistant/$TARGETARCH-base-debian:bookworm AS base

WORKDIR /data
COPY requirements.txt /data/

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
COPY config_emhass.yaml /data/

#-------------------------
#EMHASS-ADDON Default 
FROM base as addon
COPY ./requirements_addon.txt /data/requirements.txt

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
RUN  pip3 install --no-cache-dir --break-system-packages --upgrade --upgrade-strategy=only-if-needed -U emhass

COPY options.json /data/

ENTRYPOINT [ "python3", "-m", "emhass.web_server","--addon", "True", "--no_response", "True",]

#-----------
#EMHASS-ADD-ON Testing from local files
FROM addon as addon-local
COPY src/emhass/ /data/src/emhass/ 
COPY src/emhass/templates/ /data/src/emhass/templates/
COPY src/emhass/static/ /data/src/emhass/static/
COPY src/emhass/static/img/ /data/src/emhass/static/img/
COPY data/opt_res_latest.csv /data/data/
COPY options.json /data/
COPY README.md /data/
COPY setup.py /data/
#compile EMHASS locally
RUN python3 setup.py install
ENTRYPOINT [ "python3", "-m", "emhass.web_server","--addon", "True" , "--no_response", "True"]


#-----------
#EMHASS-ADD-ON  testing via git 
FROM addon as addon-git 
WORKDIR /tmp/
RUN git clone https://github.com/davidusb-geek/emhass.git
RUN mkdir -p  /data/src/emhass/
RUN mkdir -p  /data/data
RUN cp -r /tmp/emhass/src/emhass/ /data/src/emhass/
RUN cp /tmp/emhass/data/opt_res_latest.csv  /data/data/
RUN cp /tmp/emhass/options.json /data/
RUN cp /tmp/emhass/setup.py /data/
RUN cp /tmp/emhass/README.md /data/
#compile EMHASS locally
WORKDIR /data
RUN python3 setup.py install
ENTRYPOINT [ "python3", "-m", "emhass.web_server","--addon", "True" , "--no_response", "True"]

#-------------------------
#EMHASS STANDALONE
FROM base as standalone

RUN pip3 install --extra-index-url=https://www.piwheels.org/simple --no-cache-dir --break-system-packages -U  flask waitress plotly

COPY src/emhass/ /data/src/emhass/ 
COPY src/emhass/templates/ /data/src/emhass/templates/
COPY src/emhass/static/ /data/src/emhass/static/
COPY src/emhass/static/img/ /data/src/emhass/static/img/
COPY data/opt_res_latest.csv /data/data/
COPY README.md /data/
COPY setup.py /data/
#secrets file will need to be copied manually at docker run

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

#build EMHASS
RUN python3 setup.py install
ENTRYPOINT [ "python3", "-m", "emhass.web_server"]
#-------------------------


#check build arguments and build
FROM ${build_version} AS final