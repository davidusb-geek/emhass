# Installation Methods

## Method 1) The EMHASS add-on for Home Assistant OS and supervised users

For Home Assistant OS and HA Supervised users, A [EMHASS an add-on repository](https://github.com/davidusb-geek/emhass-add-on) has been developed to allow the EMHASS Docker container to run as a [Home Assistant Addon](https://www.home-assistant.io/addons/). The add-on is more user-friendly as the Home Assistant secrets (URL and API key) are automatically placed inside of the EMHASS container, and web server port *(default 5000)* is already opened.

You can find the add-on with the installation instructions here: [https://github.com/davidusb-geek/emhass-add-on](https://github.com/davidusb-geek/emhass-add-on)

These architectures are supported: `amd64` and `aarch64` (currently `armv7` and `armhf` are not supported).

_Note: Both EMHASS via Docker and EMHASS-Add-on contain the same Docker image. The EMHASS-Add-on repository however, stores Home Assistant addon specific configuration information and maintains EMHASS image version control._

## Method 2) Running EMHASS in Docker

You can also install EMHASS using Docker as a container. This can be in the same machine as Home Assistant (if your running Home Assistant as a Docker container) or in a different distant machine. The "share" folder is where EMHASS stores the config.json file. In the examples below adjust the "-v" volume mappings to reflect where your path to the local host directory needs to be mapped to.
To install first pull the latest image:
```bash
# pull Docker image
docker pull ghcr.io/davidusb-geek/emhass:latest
# run Docker image, mounting the dir storing config.json and secrets_emhass.yaml from host
docker run --rm -it --restart always  -p 5000:5000 --name emhass-container -v /emhass/share:/share/ -v /emhass/secrets_emhass.yaml:/app/secrets_emhass.yaml ghcr.io/davidusb-geek/emhass:latest
```
*Note it is not recommended to install the latest EMHASS image with `:latest` *(as you would likely want to control when you update EMHASS version)*. Instead, find the [latest version tag](https://github.com/davidusb-geek/emhass/pkgs/container/emhass) (E.g: `v0.2.1`) and replace `latest`*

You can also build your image locally. For this clone this repository, and build the image from the Dockerfile:
```bash
# git clone EMHASS repo
git clone https://github.com/davidusb-geek/emhass.git
# move to EMHASS directory 
cd emhass
# build Docker image 
# may need to set architecture tag (docker build --build-arg TARGETARCH=amd64 -t emhass-local .)
docker build -t emhass-local . 
# run built Docker image, mounting config.json and secrets_emhass.yaml from host
docker run --rm -it -p 5000:5000 --name emhass-container -v /emhass/share:/share -v /emhass/secrets_emhass.yaml:/app/secrets_emhass.yaml emhass-local
```

Before running the docker container, make sure you have a designated folder for emhass on your host device and a `secrets_emhass.yaml` file. You can get a example of the secrets file from [`secrets_emhass(example).yaml`](https://github.com/davidusb-geek/emhass/blob/master/secrets_emhass(example).yaml) file on this repository.
```bash
# cli example of creating an emhass directory and appending a secrets_emhass.yaml file inside
mkdir ~/emhass
cd ~/emhass 
cat <<EOT >> ~/emhass/secrets_emhass.yaml
hass_url: https://myhass.duckdns.org/
long_lived_token: thatverylongtokenhere
time_zone: Europe/Paris
lat: 45.83
lon: 6.86
alt: 4807.8
EOT
docker run --rm -it --restart always  -p 5000:5000 --name emhass-container -v /emhass/share:/share -v /emhass/secrets_emhass.yaml:/app/secrets_emhass.yaml ghcr.io/davidusb-geek/emhass:latest
```

### Docker, things to note 

- You can create a `config.json` file prior to running emhass. *(obtain a example from: [config_defaults.json](https://github.com/davidusb-geek/emhass/blob/enhass-standalone-addon-merge/src/emhass/data/config_defaults.json)* Alteratively, you can insert your parameters into the configuration page on the EMHASS web server. (for EMHASS to auto create a config.json) With either option, the volume mount `-v /emhass/share:/share` should be applied to make sure your config is stored on the host device. (to be not deleted when the EMHASS container gets removed/image updated)*

- If you wish to keep a local, semi-persistent copy of the EMHASS-generated data, create a local folder on your device, then mount said folder inside the container.  
  ```bash
  #create data folder 
  mkdir -p ~/emhass/data 
  docker run -it --restart always -p 5000:5000 -e LOCAL_COSTFUN="profit" -v /emhass/share:/share -v /emhass/data:/data  -v /emhass/secrets_emhass.yaml:/app/secrets_emhass.yaml --name DockerEMHASS <REPOSITORY:TAG>
  ```
    
- If you wish to set the web_server's homepage optimization diagrams to a timezone other than UTC, set `TZ` environment variable on docker run:
  ```bash
  docker run -it --restart always -p 5000:5000  -e TZ="Europe/Paris" -v /emhass/share:/share -v /emhass/secrets_emhass.yaml:/app/secrets_emhass.yaml --name DockerEMHASS <REPOSITORY:TAG>
  ```  

## Method 3) Legacy method using a Python virtual environment *(Legacy CLI)*

If you wish to run EMHASS optimizations with cli commands. *(no persistent web server session)* you can run EMHASS via the python package alone *(not wrapped in a Docker container)*.

With this method it is recommended to install on a virtual environment.
- Create and activate a virtual environment:
  ```bash
  python3 -m venv ~/emhassenv
  cd ~/emhassenv
  source bin/activate
  ```
- Install using the distribution files:
  ```bash
  python3 -m pip install emhass
  ```
- Create and store configuration (config.json), secret (secrets_emhass.yaml) and data (/data) files in the emhass dir (`~/emhassenv`)  
Note: You may wish to copy the `config.json` (config_defaults.json), `secrets_emhass.yaml` (secrets_emhass(example).yaml) and/or `/scripts/` files from this repository to the `~/emhassenv` folder for a starting point and/or to run the bash scripts described below. 

- To upgrade the installation in the future just use:
  ```bash
  python3 -m pip install --upgrade emhass
  ```
