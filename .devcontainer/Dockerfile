FROM mcr.microsoft.com/devcontainers/python:0-3.11
# EXPOSE 5000:5000
COPY .devcontainer/setup.sh ./
COPY requirements.txt ./
RUN ./setup.sh
