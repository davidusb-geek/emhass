FROM mcr.microsoft.com/devcontainers/python:1-3.8-buster

# copy the requirements file into the image
COPY requirements.txt requirements.txt
COPY requirements_webserver.txt requirements_webserver.txt

# Setup
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        coinor-cbc \
        coinor-libcbc-dev \
        gcc \
        gfortran \
        libhdf5-dev \
        libhdf5-serial-dev \
        libnetcdf-dev \
        netcdf-bin \
    && ln -s /usr/include/hdf5/serial /usr/include/hdf5/include \
    && export HDF5_DIR=/usr/include/hdf5 \
    && pip install netCDF4 \
    && apt-get purge -y --auto-remove \
        gcc \
        libhdf5-dev \
        libhdf5-serial-dev \
        netcdf-bin \
        libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements_webserver.txt
