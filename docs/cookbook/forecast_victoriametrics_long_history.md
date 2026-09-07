# Long training history with VictoriaMetrics (InfluxDB 1.x replacement)

## Goal

Keep months of high-resolution sensor history next to the (short) Home Assistant recorder so the ML load forecaster can train on 45-365 days, using VictoriaMetrics as the store now that the InfluxDB 1.x add-on is gone, and migrate the existing InfluxDB history into it so no training data is lost.

## Prerequisites

- EMHASS with the `use_victoriametrics` data source (davidusb-geek/emhass#1110) and, for the Add-on, the matching add-on version exposing `victoriametrics_username` / `victoriametrics_password` (davidusb-geek/emhass-add-on#132).
- Home Assistant OS / Supervised with the [VictoriaMetrics community add-on](https://github.com/hassio-addons/addon-victoriametrics) (hassio-addons, Apache-2.0).
- The standard Home Assistant `influxdb` integration (it keeps writing; VictoriaMetrics accepts the InfluxDB line protocol).
- For the migration step only: the Advanced SSH & Web Terminal add-on with *Protection mode* off (Docker access), and either the old InfluxDB add-on still running or a Home Assistant backup that contains it.
- Transport: direct EMHASS config / runtime payloads, verified on Home Assistant 2026.9 with EMHASS 0.18.2 + #1110. Adapter-specific transports (Node-RED, AppDaemon) untested.

Why this exists: the InfluxDB 1.x community add-on was **archived on 2026-08-28** (InfluxDB 1.x is end-of-life). Since the Home Assistant 2026.9 era the Supervisor raises a repair *"App InfluxDB has been removed from the repository"*: confirming that repair **uninstalls the add-on and deletes its database**, so migrate first. InfluxDB 3 Core is not a replacement for this use case (a single query is capped at about 72 hours of data by default, so a 45-day training query fails) and the InfluxDB 3 Enterprise "at-home" licence that lifts the cap is non-commercial. VictoriaMetrics is licence-clean, small (one binary, low RAM), keeps years of data, and needs no change on the writing side.

## Step 1: Install VictoriaMetrics and expose its port

<!-- transport: Home Assistant add-on store (tested) -->

Install the add-on from the hassio-addons repository and set its options:

```yaml
# VictoriaMetrics add-on options
retention_period: 3y
ssl: false                     # the add-on default is true; keep false unless you provide certificates
leave_front_door_open: false   # false = the direct port requires HTTP Basic auth, validated against Home Assistant users
home_assistant: false          # we push through the influxdb integration, not the Prometheus scrape
```

In the add-on's **Network** section set the host port for `8428/tcp` to `8428`. Without a published host port the add-on only serves its ingress endpoint, which refuses Home Assistant Core and any other client with 403, even with `leave_front_door_open`.

With `leave_front_door_open: false`, create a dedicated Home Assistant user (for example `emhass_vm`, local user, no administrator rights needed) whose credentials both the integration and EMHASS will use. Only the Basic-auth *header* is accepted, not `?u=&p=` query parameters.

Expected: `curl -u emhass_vm:<pw> http://<ha-host>:8428/api/v1/status/tsdb` returns JSON, and `401` without credentials.

## Step 2: Point the Home Assistant influxdb integration at VictoriaMetrics

<!-- transport: Home Assistant YAML / config entry (tested) -->

VictoriaMetrics ingests the InfluxDB v1 line protocol, so the integration only needs a new host, port and `api_version: 1`:

```yaml
# configuration.yaml (or a package)
influxdb:
  api_version: 1
  host: a0d7b954-victoriametrics     # the add-on hostname, as seen from Home Assistant Core
  port: 8428
  ssl: false
  verify_ssl: false
  database: homeassistant            # becomes the "db" label of every metric
  username: !secret victoriametrics_username
  password: !secret victoriametrics_password
  max_retries: 3
  include:
    domains: [sensor]                # or list the entities EMHASS and your reports need
```

Keep the default measurement format (no `measurement_attr`, no extra `tags_attributes`): every numeric state is then stored as `<unit>_value{entity_id="<object_id>", domain="sensor", db="homeassistant"}`, which is exactly what the migration in Step 3 produces and what EMHASS looks up.

A caveat that costs people hours: the `influxdb` integration is **config-entry based**, the YAML is imported once. If an InfluxDB entry already exists (Settings, Devices & services, InfluxDB) it keeps pointing at the old host, silently fails to write and logs nothing at the default log level. Delete that entry, then restart Home Assistant so the YAML is imported again. Restart once more if nothing arrives after the first restart.

Expected: after the restart, `curl -u emhass_vm:<pw> "http://<ha-host>:8428/api/v1/label/entity_id/values"` lists your sensors, and `.../api/v1/query?query=W_value{entity_id="<your_load_sensor>"}` returns a fresh value.

## Step 3: Migrate the InfluxDB 1.x history (optional but recommended)

<!-- transport: Docker from the Advanced SSH add-on, protection mode off (tested) -->

[vmctl](https://docs.victoriametrics.com/vmctl/) copies an InfluxDB 1.x database into VictoriaMetrics with the same metric names as the live integration. Run it from the SSH add-on while the old InfluxDB add-on is still running (before the repair is confirmed):

```bash
docker run -d --name vmctl_mig --network hassio victoriametrics/vmctl:latest influx \
  --influx-addr http://a0d7b954-influxdb:8086 --influx-database homeassistant \
  --influx-user <influx-user> --influx-password <influx-pw> \
  --vm-addr http://a0d7b954-victoriametrics:8428 --vm-user emhass_vm --vm-password <pw> \
  --influx-measurement-field-separator _ --influx-concurrency 1 -s --disable-progress-bar
docker logs -f vmctl_mig      # exit code 0 = done; about 2.5 h for 400M points at concurrency 1
```

`-s` (skip the confirmation prompt) is an `influx` sub-command flag, not a global one, and `--disable-progress-bar` is required because there is no TTY. The `_` separator turns measurement `W` + field `value` into `W_value`, identical to the live writes.

If the InfluxDB add-on is already gone but you have a Home Assistant backup containing it: extract `a0d7b954_influxdb.tar.gz` from the backup with GNU tar (the HAOS busybox tar cannot read the PAX archive; the `influxdb:1.8` image has both GNU tar and `influxd`), start a throw-away `influxdb:1.8` container on the extracted data directory, and run the same vmctl command against it.

Expected: `count_over_time(W_value{entity_id="<load>"}[1d])` over the migrated range shows no empty days, and the labels of a migrated point equal those of a live point.

## Step 4: Configure EMHASS

<!-- source: src/emhass/data/config_defaults.json:42 -->
<!-- source: src/emhass/retrieve_hass.py:875 -->
<!-- transport: EMHASS config.json + Add-on options (tested) -->

```json
{
  "use_victoriametrics": true,
  "victoriametrics_host": "a0d7b954-victoriametrics",
  "victoriametrics_port": 8428,
  "victoriametrics_database": "homeassistant",
  "victoriametrics_metric_regex": ".+_value",
  "victoriametrics_use_ssl": false,
  "victoriametrics_verify_ssl": false,
  "historic_days_to_retrieve": 90
}
```

<!-- source: src/emhass/utils.py:3264 -->
Put `victoriametrics_username` / `victoriametrics_password` in the **Add-on Configuration pane** (Docker: `secrets_emhass.yaml`). They are secrets: EMHASS does not read them from `config.json`. Leave them empty if the add-on runs with `leave_front_door_open: true`.

<!-- source: src/emhass/retrieve_hass.py:1008 -->
EMHASS finds a sensor by its `entity_id` label plus the metric-name regex, so the unit of each sensor is never configured. `use_websocket` can stay as it was; `use_influxdb: true` would take precedence, so set it to `false` once migrated.

Expected: the EMHASS log shows `VictoriaMetrics integration enabled: a0d7b954-victoriametrics:8428` and, on the next retrieval, `Retrieving N sensors over D days from VictoriaMetrics` followed by `VictoriaMetrics data retrieval completed: (rows, N)`.

## Step 5: Train on the long history

<!-- source: src/emhass/forecast.py:2161 -->
<!-- transport: Home Assistant rest_command / shell_command payload (tested) -->

```json
{
  "historic_days_to_retrieve": 90,
  "model_type": "load_forecast",
  "var_model": "sensor.power_load_no_var_loads",
  "sklearn_model": "KNeighborsRegressor",
  "num_lags": 96,
  "split_date_delta": "48h",
  "perform_backtest": false
}
```

Runtime parameters override `config.json`: if your fit automation still passes `"historic_days_to_retrieve": 9` the long history is never used. `num_lags` must be at least the MPC `prediction_horizon` (96 for 24 h at 15 min), otherwise `naive-mpc-optim` fails with *Unable to obtain: 96 lags_opt values*.

Expected: `Retrieved 8640 data points for sensor.power_load_no_var_loads (W_value)` (90 days at 15 min) and a normal model fit.

## Caveats

- Long windows are fetched in chunks of 10 000 points, below the server's `-search.maxPointsPerTimeseries` (30 000). A year at 15 min needs 4 requests; nothing to configure.
- The add-on runs VictoriaMetrics with `-search.latencyOffset=1m`: the last minute is not returned by queries. Irrelevant for training, do not "fix" it.
- A sensor that changed unit (W to kW) matches two metrics; EMHASS keeps the one with the most samples and warns. Pin it with `victoriametrics_metric_regex: "W_value"` if needed, or use a `{{ 'sensor.x' * 1000 }}` expression to rescale.
- Empty buckets are forward-filled like InfluxDB `FILL(previous)`; a sensor that reports rarely simply repeats its last value inside each step.
- The recorder is untouched: keep its short `purge_keep_days`; VictoriaMetrics is the long-term store.
- Publishing port 8428 exposes VictoriaMetrics on your LAN, hence the Basic auth. Do not use `leave_front_door_open: true` on a network you do not trust.
- Restoring a Home Assistant backup taken after the InfluxDB add-on was archived removes "detached" add-ons together with their data. Migrate before restoring anything.

## Credits

- Discussion in davidusb-geek/emhass#1100 (InfluxDB 1.x add-on archived, InfluxDB 3 licensing).
- Migration and integration recipe validated on two production installs by @scruysberghs, 2026-09.
- Field names verified against `src/emhass/data/config_defaults.json` and `src/emhass/retrieve_hass.py` on 2026-09-05.
