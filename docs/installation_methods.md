# Installation Methods

## Method 1) The EMHASS app for Home Assistant OS and supervised users

For Home Assistant OS and HA Supervised users, A [EMHASS app repository](https://github.com/davidusb-geek/emhass-add-on) has been developed to allow the EMHASS Docker container to run as a [Home Assistant app](https://www.home-assistant.io/addons/). The app is more user-friendly as the Home Assistant secrets (URL and API key) are automatically placed inside of the EMHASS container, and web server port *(default 5000)* is already opened.

You can find the app with the installation instructions here: [https://github.com/davidusb-geek/emhass-app](https://github.com/davidusb-geek/emhass-add-on)

```{important}
The EMHASS API requires a token. Open EMHASS from the Home Assistant sidebar and choose the key icon (**API token**) to read the token EMHASS generated, or set your own in the app **Configuration** (`emhass_api_token`). Home Assistant signs you in to the web UI through ingress; automations that call the API on port 5000 must send `Authorization: Bearer <token>`. See [API authentication](api_authentication.md).
```

These architectures are supported: `amd64` and `aarch64` (currently `armv7` and `armhf` are not supported).

```{note} 

_Both EMHASS via Docker and EMHASS-app contain the same Docker image. The EMHASS-app repository however, stores Home Assistant addon specific configuration information and maintains EMHASS image version control._
```

## Method 2) Running EMHASS in Docker

You can also install EMHASS using Docker as a container. This can be in the same machine as Home Assistant (if your running Home Assistant as a Docker container) or in a different distant machine. The "share" folder is where EMHASS stores the config.json file. In the examples below adjust the "-v" volume mappings to reflect where your path to the local host directory needs to be mapped to.
To install first pull the latest image:
```bash
# pull Docker image
docker pull ghcr.io/davidusb-geek/emhass:latest
# run Docker image, mounting the dir storing config.json and secrets_emhass.yaml from host
docker run -it --restart always  -p 5000:5000 --name emhass-container -v /emhass/share:/share/ -v /emhass/data:/data -v /emhass/secrets_emhass.yaml:/app/secrets_emhass.yaml ghcr.io/davidusb-geek/emhass:latest
```

```{important}
The web API requires a token. If you do not set `EMHASS_API_TOKEN` (or `emhass_api_token` in `secrets_emhass.yaml`), EMHASS generates one at first start and keeps it in `/data/emhass_api_token`, so keep `-v /emhass/data:/data`. Read it with `docker exec emhass-container cat /data/emhass_api_token` and send it as `Authorization: Bearer <token>`. See [API authentication](api_authentication.md).
```

```{note} 

*It is not recommended to install the latest EMHASS image with `:latest` *(as you would likely want to control when you update EMHASS version)*. Instead, find the [latest version tag](https://github.com/davidusb-geek/emhass/pkgs/container/emhass) (E.g: `v0.2.1`) and replace `latest`*
```

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
docker run --rm -it -p 5000:5000 --name emhass-container -v /emhass/share:/share -v /emhass/data:/data -v /emhass/secrets_emhass.yaml:/app/secrets_emhass.yaml emhass-local
```

Before running the docker container, make sure you have a designated folder for emhass on your host device and a `secrets_emhass.yaml` file. You can get a example of the secrets file from [`secrets_emhass(example).yaml`](https://github.com/davidusb-geek/emhass/blob/master/secrets_emhass(example).yaml) file on this repository.
```bash
# cli example of creating an emhass directory and appending a secrets_emhass.yaml file inside
mkdir ~/emhass
cd ~/emhass 
cat <<EOT >> ~/emhass/secrets_emhass.yaml
hass_url: https://homeassistant.invalid:8123/
long_lived_token: thatverylongtokenhere
time_zone: Europe/Paris
lat: 45.83
lon: 6.86
alt: 4807.8
EOT
docker run -it --restart always  -p 5000:5000 --name emhass-container -v /emhass/share:/share -v /emhass/data:/data -v /emhass/secrets_emhass.yaml:/app/secrets_emhass.yaml ghcr.io/davidusb-geek/emhass:latest
```

### Docker, things to note 

- You can create a `config.json` file prior to running emhass. *(obtain an example from: [config_defaults.json](https://github.com/davidusb-geek/emhass/blob/master/src/emhass/data/config_defaults.json)* Alternatively, you can insert your parameters into the configuration page on the EMHASS web server. (for EMHASS to auto create a config.json) With either option, the volume mount `-v /emhass/share:/share` should be applied to make sure your config is stored on the host device. (to be not deleted when the EMHASS container gets removed/image updated)*

- Mount a local folder on `/data` (below). Besides the EMHASS-generated data it holds the API token EMHASS generates when you set none (`/data/emhass_api_token`); without it every new container gets a new token and your automations stop working after each update. Setting `EMHASS_API_TOKEN` instead avoids the question.  
  ```bash
  #create data folder 
  mkdir -p ~/emhass/data 
  docker run -it --restart always -p 5000:5000 -e LOCAL_COSTFUN="profit" -v /emhass/share:/share -v /emhass/data:/data  -v /emhass/secrets_emhass.yaml:/app/secrets_emhass.yaml --name DockerEMHASS <REPOSITORY:TAG>
  ```
    
- If you wish to set the web_server's homepage optimization diagrams to a timezone other than UTC, set `TZ` environment variable on docker run:
  ```bash
  docker run -it --restart always -p 5000:5000  -e TZ="Europe/Paris" -v /emhass/share:/share -v /emhass/data:/data -v /emhass/secrets_emhass.yaml:/app/secrets_emhass.yaml --name DockerEMHASS <REPOSITORY:TAG>
  ```  

- Inside the container the web server listens on the address in the `IP` environment variable (default `0.0.0.0`, all interfaces) and the port in `PORT` (default `5000`). `server_ip` in `secrets_emhass.yaml` does not change it: it is used only when `web_server.py` is run directly. To choose which host address the port is reachable on, use Docker's port mapping, for example `-p 192.0.2.10:5000:5000`.

### Running as a non-root user

The image runs as root unless you pick another user, and the Home Assistant app keeps running it that way. To run it as a non-root user, give Docker the UID and group to use. The image's own `/data` and `/share` folders belong to group 0 and that group can write to them, so without mounts any UID in group 0 works. With the folders mounted from the host as in Method 2:

```bash
docker run -it --restart always -p 5000:5000 --user 1000:0 -v /emhass/share:/share -v /emhass/data:/data -v /emhass/secrets_emhass.yaml:/app/secrets_emhass.yaml --name emhass-container ghcr.io/davidusb-geek/emhass:latest
```

- Folders you mount from the host must be writable by that UID, and `secrets_emhass.yaml` readable by it. For example `sudo chown -R 1000 /emhass/share /emhass/data`.
- If the container ran as root before, the files EMHASS created in those folders belong to root. Change their owner as above before you start it as another user, or it starts but can't use the files it made before and its API requests fail. For a named volume, run the chown in a container, for example `docker run --rm -v emhass-data:/data -v emhass-share:/share --entrypoint chown ghcr.io/davidusb-geek/emhass:latest -R 1000 /data /share`.
- Another group (for example `--user 1000:1000`) works too, as long as both `/data` and `/share` are mounted from folders that user can write to. EMHASS writes to both, and saves `config.json` in `/share` by default.
- With new named volumes, keep group 0. Docker copies the image folder's owner and mode into an empty named volume each time it is mounted, so the volume stays writable for group 0 and a `chown` of the empty volume doesn't last.

With Docker Compose, set `user:` on the service:

```yaml
services:
  emhass:
    image: ghcr.io/davidusb-geek/emhass:latest
    user: "1000:0"
    restart: always
    ports:
      - "5000:5000"
    volumes:
      - /emhass/share:/share
      - /emhass/data:/data
      - /emhass/secrets_emhass.yaml:/app/secrets_emhass.yaml
```

On Kubernetes, set the user in the pod's `securityContext` and mount writable volumes on `/data` and `/share`. `fsGroup` makes the volumes writable for that group, on volume types that support it (`hostPath` doesn't). This is the pod spec; in a Deployment it goes under `spec.template.spec`. Together with the pod-level settings, the container `securityContext` below meets the restricted Pod Security profile. EMHASS needs no capabilities and no privilege escalation:

```yaml
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: emhass
      image: ghcr.io/davidusb-geek/emhass:latest
      securityContext:
        allowPrivilegeEscalation: false
        capabilities:
          drop: ["ALL"]
      ports:
        - containerPort: 5000
      volumeMounts:
        - name: data
          mountPath: /data
        - name: share
          mountPath: /share
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: emhass-data
    - name: share
      persistentVolumeClaim:
        claimName: emhass-share
```

Provide `secrets_emhass.yaml` the same way as with Docker, for example from a Secret mounted at `/app/secrets_emhass.yaml` with `subPath`.

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
