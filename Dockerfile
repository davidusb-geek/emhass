FROM python:3.11-slim-buster
# FROM ghcr.io/home-assistant/amd64-base-debian:bookworm # Uncomment to test add-on
# FROM ghcr.io/home-assistant/armhf-base-debian:bookworm

# switch working directory
WORKDIR /app

# copy the requirements file into the image
COPY requirements.txt requirements.txt
COPY requirements_webserver.txt requirements_webserver.txt
COPY setup.py setup.py
COPY README.md README.md

# Setup
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        # libffi-dev \
        # python3 \
        # python3-pip \
        # python3-dev \
        # git \
        # build-essential \
        gcc \
        coinor-cbc \
        coinor-libcbc-dev \
        libhdf5-dev \
        libhdf5-serial-dev \
        netcdf-bin \
        libnetcdf-dev \
        # pkg-config \
        # gfortran \
        # libatlas-base-dev \
    && ln -s /usr/include/hdf5/serial /usr/include/hdf5/include \
    && export HDF5_DIR=/usr/include/hdf5 \
    # && pip3 install --extra-index-url=https://www.piwheels.org/simple --no-cache-dir --break-system-packages -U setuptools wheel \
    && pip3 install --no-cache-dir --break-system-packages -r requirements_webserver.txt \
    && apt-get purge -y --auto-remove \
        gcc \
        libhdf5-dev \
        libhdf5-serial-dev \
        netcdf-bin \
        libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# copy contents
COPY src/emhass/__init__.py /app/src/emhass/__init__.py
COPY src/emhass/command_line.py /app/src/emhass/command_line.py
COPY src/emhass/forecast.py /app/src/emhass/forecast.py
COPY src/emhass/machine_learning_forecaster.py /app/src/emhass/machine_learning_forecaster.py
COPY src/emhass/optimization.py /app/src/emhass/optimization.py
COPY src/emhass/retrieve_hass.py /app/src/emhass/retrieve_hass.py
COPY src/emhass/utils.py /app/src/emhass/utils.py
COPY src/emhass/web_server.py /app/src/emhass/web_server.py
COPY src/emhass/templates/index.html /app/src/emhass/templates/index.html
COPY src/emhass/static/style.css /app/src/emhass/static/style.css
COPY src/emhass/static/img/emhass_logo_short.svg /app/src/emhass/static/img/emhass_logo_short.svg
COPY src/emhass/static/img/emhass_icon.png /app/src/emhass/static/img/emhass_icon.png
COPY data/opt_res_latest.csv /app/data/opt_res_latest.csv

RUN python3 setup.py install

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# configure the container to run in an executed manner
CMD [ "python3", "src/emhass/web_server.py" ]
