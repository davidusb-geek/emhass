# API authentication

EMHASS's web server requires credentials. Without them nobody can change the configuration or secrets,
run optimizations, or publish states to Home Assistant. This fixes GHSA-c39q-j4r5-9xq5, where anyone who could
reach the HTTP port could do all of that.

```{important}
**Upgrading from 0.18.9 or earlier:** every request to the API now needs `Authorization: Bearer <token>`.
Your existing `rest_command`, `shell_command`, `curl` and Node-RED calls get `401` until you add the header.
Read [Migrating](#migrating-from-an-unauthenticated-install) first.
```

## Quick start

| You run EMHASS as | Get your token | Use it |
|---|---|---|
| Home Assistant add-on | Open EMHASS in the sidebar, choose the key icon (**API token**; only Home Assistant administrators see it), or set your own in the add-on **Configuration** (`emhass_api_token`) | Automations send it as a bearer header; the web UI needs nothing (Home Assistant signs you in) |
| Docker | `docker exec emhass-container cat /data/emhass_api_token`, or set `-e EMHASS_API_TOKEN=...` | Bearer header; browser: open `/login` |
| Python / source | Set `EMHASS_API_TOKEN` or `emhass_api_token` in `secrets_emhass.yaml` | Same |

Send it with every call:

```bash
curl -H "Authorization: Bearer $EMHASS_API_TOKEN" -H "Content-Type: application/json" \
  -X POST -d '{}' http://localhost:5000/action/publish-data
```

```yaml
# configuration.yaml
rest_command:
  emhass_publish:
    url: http://localhost:5000/action/publish-data
    method: POST
    headers:
      authorization: !secret emhass_bearer
      content-type: application/json
    payload: "{}"
```

```yaml
# secrets.yaml (Home Assistant): the word Bearer, a space, then the token
emhass_bearer: "Bearer PASTE_THE_TOKEN_HERE"
```

A Home Assistant `shell_command` runs inside Home Assistant, so it cannot read `$EMHASS_API_TOKEN`: write the token
in the command (or use `rest_command` with `!secret`, which is cleaner).

## What needs credentials

Every request except `GET /healthz`, `/login` and `/logout`: the web pages, `/action/<action>`, `/set-config`,
`/get-config`, `/get-json`, `/api/v1/*` and the static files. A request without valid credentials gets `401` (pages in a
browser go to `/login`). The health check stays open so container health checks keep working.

## The token

**Where it comes from**, in this order, a later source replacing an earlier one (the same order as the other secrets):

1. the `EMHASS_API_TOKEN` environment variable;
2. `emhass_api_token` in `options.json` (add-on option);
3. `emhass_api_token` in `secrets_emhass.yaml` (quote it: YAML can change an unquoted number).

**If none is set, EMHASS generates one** the first time it starts and keeps it in the data folder, in the file
`emhass_api_token` (mode `0600` on POSIX; never a link). Later starts reuse it, so it survives restarts. The log names the file,
never the token. In the add-on the web UI shows it to you through ingress. Set `EMHASS_API_TOKEN_AUTOGENERATE=false` to
disable generation: EMHASS then answers `503` until you configure a token (Home Assistant ingress is still served, since it needs no token).

**Rules.** At least 16 characters (use a long random value, for example `openssl rand -base64 32`; URL-safe characters
are safest). A source that exists but cannot be used (unreadable, broken YAML or JSON, a number instead of text, the key
twice) makes every request except `/healthz` answer `503`; it never falls back to running open. Changing the token needs
a restart, and signs out all browser sessions.

```{warning}
With Docker, keep the data folder: `-v /emhass/data:/data`. Without it, a container started with `--rm` forgets
the generated token and makes a new one on every start, and your automations keep failing. Setting
`EMHASS_API_TOKEN` avoids the question.
```

## Home Assistant add-on and ingress

The add-on is reached two ways. **Ingress** (the EMHASS panel in the sidebar) is already protected by Home Assistant's
login. **The API port** (5000, used by your automations) is reachable from your network, so it needs the token.

EMHASS therefore listens on two ports when it runs as an add-on:

| Port | Who may use it | Credentials |
|---|---|---|
| 5000 (API, published) | any client | bearer token or login cookie, always |
| 8099 (ingress, not published) | only the Supervisor, address `172.30.32.2` | none: Home Assistant has signed the user in. Any other peer is refused with `403`, even with the token |

This follows Home Assistant's own rule for add-ons: ingress runs on a port of its own, and only connections from
`172.30.32.2` are accepted. Ingress is recognised by **which port accepted the connection and who connected**, never by a
header: a client that reaches a port can send any header (`X-Ingress-Path`, `X-Forwarded-For`, `X-Remote-User-*`), so
none is used to recognise ingress. (The identity headers the Supervisor adds are read for one purpose only, the administrator-only token page below, and only after the port and peer rule has already held.)
The servers are configured not to rewrite the peer address from `X-Forwarded-For`.

**What EMHASS relies on from Home Assistant's ingress** (read from the Supervisor's `api/ingress.py` and the add-on developer documentation; not run against a live Supervisor):

- The Supervisor connects from `172.30.32.2` to the add-on's `ingress_port`, and refuses a browser that has no valid ingress session.
- With `panel_admin: true` (the default, and set explicitly in the add-on) it answers `403` unless the signed-in user is an administrator, and a request with no resolvable user is denied.
- It adds `X-Remote-User-Id` and `X-Remote-User-Groups` after removing any copy the client sent. EMHASS reads them **only** for the `/api-token` page and for the key link on the home page, and only on the ingress port: the page needs a user id and the group `system-admin`, so it stays administrator-only even for a configuration that set `panel_admin: false`. A request with no user identity gets `403`.
- It forwards redirects (`Location`), `Set-Cookie` and other response headers unchanged and does not follow redirects, and it serves the app under `/api/hassio_ingress/<token>/`, which the app does not see. Every URL, redirect and link EMHASS emits is therefore relative (a test scans the UI assets and the redirects), and the login cookie is set without a `Path`.

A request that arrives through ingress and changes something must be JSON (the web UI always sends JSON), so a forged
cross-site form post is refused.

Settings (normally automatic): `EMHASS_INGRESS_PORT` (default `8099` when `SUPERVISOR_TOKEN` is set, which the Supervisor
does for add-ons; unset otherwise) and `EMHASS_INGRESS_PEERS` (default `172.30.32.2`; a comma-separated list of
addresses or networks that **replaces** the default).

**Supervisor requirement.** The administrator-only token page trusts `X-Remote-User-Groups`, which only a Supervisor that adds the header *and* removes any client-supplied copy can vouch for (the version that introduced this has not been identified here, so it is not claimed). On a Supervisor without that behaviour, do not rely on the page: set `emhass_api_token` in the add-on options and leave the generated token unused. The same Supervisor behaviour is also what makes `panel_admin` effective. The add-on must declare `ingress_port: 8099` and the option `emhass_api_token` (see the add-on repository). The stable
add-on pins the EMHASS image to its own version, so both change in one update. Do not mix them: an older add-on
configuration (ingress on port 5000) with this image shows the login page and gives you no way to learn the token, and
this add-on configuration with an older image has nothing listening on 8099. The `emhass-test` add-on follows the
floating `test` image, so update both together there as well.

## Browser login

Open `/login`, enter the token. The session cookie is `HttpOnly` and `SameSite=Strict`, lasts 30 days, and stops working
when EMHASS restarts or the token changes. It is set without a `Path`, so the browser sends it only below the folder `/login`
was served from. A request that carries only the cookie must be JSON when it changes something. Logging out clears the
browser's copy; a copy held elsewhere stays valid until it expires or EMHASS restarts.

## Migrating from an unauthenticated install

**The contract.** After upgrading, API calls without a token fail with `401` and a message that names the fix. Nothing is
silently left open, and nothing silently keeps working without credentials.

1. **Upgrade.** EMHASS starts, finds no token, generates one and logs where it is.
2. **Find the token** (table above).
3. **Update your callers**: add the header to every `rest_command`, `shell_command`, `curl`, Node-RED HTTP request and
   script. The examples in this documentation already show it.
4. **Check**: `curl -i http://HOST:5000/api/v1/last-run` returns `401`; with the header it returns `200`.

Compatibility notes:

- *Dashboards and sensors that read `/api/v1/plan` or `/api/v1/last-run`* need the header too (reads are protected).
- *Dashboards that embed `http://host:5000/`* (Home Assistant iframe or webpage cards) and live on a different site than EMHASS cannot keep the `SameSite=Strict` login cookie and stay on the login page. Embed the EMHASS panel through Home Assistant ingress instead, or call the API with the bearer header.
- *Reverse proxies* (nginx, Traefik, Cloudflare tunnels) only need to pass the `Authorization` header through. They must not
  add credentials themselves unless you want the proxy to be the access control.
- *Trusted networks*: if you accept the risk on a network you fully control, `EMHASS_API_AUTH_DISABLED=true` restores the old
  behaviour. It is loud (a warning at every start), explicit, and it re-opens the advisory.
- *Rolling back*: run the previous version; nothing on disk changes except the harmless `emhass_api_token` file.
- *CLI* (`emhass --action ...`) does not use the web server and is unaffected.

## Threat model

Assets: the configuration and secrets (Home Assistant token, InfluxDB and forecast keys), the ability to run actions,
the Home Assistant entities EMHASS publishes. Attackers considered: another device on the network, a web page in a
user's browser, another add-on or container on the same host, a local process on the host.

| Area | Control | Residual risk |
|---|---|---|
| Routes | one `before_request` hook, default closed; only `/healthz`, `/login`, `/logout` are open; tests enumerate every route in the URL map, unauthenticated on the API port and as a stranger on the ingress port, and pin the open list | a future route is protected automatically; on the ingress port the three open endpoints still answer anyone |
| Token | 16+ characters; constant-time comparison; never logged, placed in runtime parameters or sent in an error or a response, except `/api-token` (a Home Assistant administrator through ingress, opened as a page, never to a script); every log record masks it | a token given under a wrong key name, or cut by code that edits a value before logging, can escape the mask (the known cases are fixed) |
| No token | generated and persisted (written to a temporary file, flushed and hard-linked into place, so it appears complete or not at all; on a file system without hard links an exclusive create is used and a crash can leave an empty file, which is replaced when EMHASS next starts at least 10 seconds after the crash; a restart inside those 10 seconds answers `503` until it is restarted again), else `503`; never open | whoever can read the data folder can read the token. On POSIX the file is mode `0600`; **on Windows it is not restricted to its owner** (a warning is logged when it is generated): set `EMHASS_API_TOKEN` there |
| Bearer vs cookie | bearer works everywhere; cookie is for browsers | a copied cookie works until it expires or EMHASS restarts |
| CSRF | `SameSite=Strict`; cookie and ingress requests that change something must be JSON; the existing `Sec-Fetch-Site: cross-site` refusal | none known; `SameSite` ignores ports, which is why JSON is also required |
| Cookie scope | no `Path`: browser scopes it to the login folder | the cookie is sent to every port of the same host (a browser rule) |
| Proxy trust | the TCP peer only; `forwarded_allow_ips` is empty and `proxy_headers=False`, so no header can change the peer | a deployment that re-enables forwarded headers lets a trusted forwarder choose the peer |
| Ingress | separate port; peer must be the Supervisor; every other peer gets `403`; no header is read | another add-on that can forge the Supervisor's source address could reach the ingress port (needs raw-socket rights and layer-2 position on the hassio network). With `--network host` the ingress port listens on every interface of the host, so a peer that can connect as `172.30.32.2` (a Supervised host) is the only one let in |
| `/api-token` | only through trusted ingress, only for a generated token, only when opened as a page (`Sec-Fetch-Mode: navigate`, so `fetch()` from another add-on's panel gets `403`), and served with `Content-Security-Policy: sandbox`, so a page that frames or opens it cannot read its text (opaque origin) | a user who opens the page can read it. Script on the shared Home Assistant origin can already drive EMHASS through ingress; treat the token as equal to EMHASS admin access |
| Failure modes | unreadable/invalid source, short token, unwritable data folder: `503`, never open | a misconfiguration is an outage, not a breach; `/healthz` stays up |
| Brute force | 16+ characters; failed logins wait one second; failures are logged as a count once a minute | no lockout; use a long random token |
| Information | `/healthz` is open (status only) | readiness, uptime, the EMHASS and Python versions, the last run's time and status and the schema version are visible |

## Reference

| Variable | Meaning |
|---|---|
| `EMHASS_API_TOKEN` | the token (lowest precedence: `options.json` and the secrets file override it, as for other secrets) |
| `EMHASS_API_TOKEN_AUTOGENERATE` | `false` stops EMHASS generating a token |
| `EMHASS_API_AUTH_DISABLED` | `true`, `1`, `yes` or `on` switches access control off (logged) |
| `EMHASS_INGRESS_PORT` | the ingress listener's port |
| `EMHASS_INGRESS_PEERS` | peers trusted on the ingress port (default `172.30.32.2`) |
| `DATA_PATH` | where the generated token is kept (default `/data/`) |

| Response | Meaning |
|---|---|
| `401` + `WWW-Authenticate: Bearer` | no or wrong token |
| `403` | a cookie or ingress request that changes something was not JSON; or any request or WebSocket handshake to the ingress port from a peer other than the Supervisor (only `/healthz`, `/login` and `/logout` still answer there); or `/api-token` opened by a script, by a user who is not an administrator, or with no user identity |
| `503` | access control is not usable: no token and none can be generated, or a token source is broken. Nothing ran |

## What has been verified

Verified by the automated tests in `tests/test_api_auth.py`, `tests/test_api_ingress.py` and `tests/test_api_provisioning.py` (they also check that the shipped gunicorn settings reach uvicorn as an empty trust list), and by manual checks outside CI on the shipped server stack (gunicorn with the shipped configuration, two listeners) and on the production Docker image with a probe container at `172.30.32.2` and a stranger container on the same network:
the closed default, generation and persistence, the ingress listener and peer rule, forged headers, the proxy-header attack,
failure modes, log hygiene, and that refused requests write nothing and publish nothing.
**Not verified against a real Home Assistant Supervisor:** the Supervisor was emulated by a proxy that copies its documented
and source-confirmed behaviour (the address `172.30.32.2`, the headers it adds). Confirm on a real installation that the
panel opens through ingress and that `GET /api-token` works there before relying on it.
