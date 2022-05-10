FROM python:3.8-slim-buster

# switch working directory
WORKDIR /app

# copy the requirements file into the image
COPY requirements_webserver.txt requirements_webserver.txt

# Setup slim-buster
RUN pip3 install --no-cache-dir -r requirements_webserver.txt

# copy contents
COPY src/emhass/__init__.py /app/src/emhass/__init__.py
COPY src/emhass/command_line.py /app/src/emhass/command_line.py
COPY src/emhass/forecast.py /app/src/emhass/forecast.py
COPY src/emhass/optimization.py /app/src/emhass/optimization.py
COPY src/emhass/retrieve_hass.py /app/src/emhass/retrieve_hass.py
COPY src/emhass/utils.py /app/src/emhass/utils.py
COPY src/emhass/web_server.py /app/src/emhass/web_server.py
COPY config_emhass.json /app/config_emhass.json
COPY data/opt_res_dayahead_latest.csv /app/data/opt_res_dayahead_latest.csv

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# configure the container to run in an executed manner
CMD [ "python3", "src/emhass/web_server.py" ]