#!/usr/bin/env bash
set -euo pipefail

main() {
    set -x
    apt-get update
    apt-get install -y --no-install-recommends \
        coinor-cbc \
        coinor-libcbc-dev \
        gcc \
        gfortran \
        libhdf5-dev \
        libhdf5-serial-dev \
        libnetcdf-dev \
        netcdf-bin

    ln -s /usr/include/hdf5/serial /usr/include/hdf5/include
    export HDF5_DIR=/usr/include/hdf5
    pip install netCDF4

    pip install -r requirements_webserver.txt

    rm -rf "$0"
}

main
