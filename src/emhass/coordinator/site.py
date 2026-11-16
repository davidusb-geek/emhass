"""`site`: the house in one list, for the coordinated backend.

Validated here by shape before anything reads it (it may come with a
request), and compiled into EMHASS's own keys where they can hold it: the
main meter's limits and one hybrid inverter on it. The coordinator
(`emhass.coordinator.backend`) reads the rest.
"""

from __future__ import annotations

import logging
import math
import re

_SITE_ID = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_SITE_DEVICE = re.compile(r"^(pv|battery[0-9]?|deferrable[0-9]{1,3})$")
_SITE_ROOT = "grid"
_SITE_NODE_TYPES = ("hybrid_inverter", "inverter", "panel", "breaker", "meter")
_SITE_SOLVERS = ("emhass",)
# A house's meter, inverters, panels and breakers fit well within these; they
# bound what a site sent with a request can make the coordinator do.
_SITE_MAX_ELEMENTS = 64
_SITE_MAX_ROWS = 16
# EMHASS's own secrets: a remote's token_secret may not name one, or a saved
# site would send it (Home Assistant's token, say) to the remote's URL.
_SITE_RESERVED_SECRETS = frozenset(
    {
        "hass_url",
        "long_lived_token",
        "participant_token",
        "remote_tokens",
        "server_ip",
        "time_zone",
        "lat",
        "lon",
        "alt",
        "latitude",
        "longitude",
        "altitude",
        "solcast_api_key",
        "solcast_rooftop_id",
        "solar_forecast_kwp",
        "influxdb_username",
        "influxdb_password",
        "victoriametrics_username",
        "victoriametrics_password",
    }
)


def site_token_secret_usable(name) -> bool:
    """Whether `name` may be a remote solver's token_secret: a secret's name
    (lower case, digits, _) that is not one of EMHASS's own."""
    return (
        isinstance(name, str)
        and bool(_SITE_ID.match(name))
        and name.lower() not in _SITE_RESERVED_SECRETS
    )


_SITE_KEYS = {
    "grid": {"id", "max_import", "max_export"},
    "node": {
        "id",
        "parent",
        "type",
        "max_import",
        "max_export",
        "efficiency_import",
        "efficiency_export",
    },
    "limit": {"id", "type", "max_import", "max_export"},
    "device": {"id", "parent", "solver", "group", "limits"},
    "remote": {"id", "parent", "solver", "limits"},
}


def _finite_number(value) -> bool:
    """A real number (not a boolean), finite."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def site_kind(element: dict) -> str:
    """What a `site` element is: "grid" (the main meter), "limit" (a cap on
    the devices tagged with it), "remote" (a solver on the network), "device"
    (an EMHASS device name or "pv") or "node" (anything else: an inverter, a
    panel, a breaker, a meter)."""
    if element.get("id") == _SITE_ROOT:
        return "grid"
    if element.get("type") == "limit":
        return "limit"
    if isinstance(element.get("solver"), dict):
        return "remote"
    if isinstance(element.get("id"), str) and _SITE_DEVICE.match(element["id"]):
        return "device"
    return "node"


def _site_element_problem(element, where: str, from_request: bool) -> str | None:
    """Why one `site` element cannot be used as given, or None (its shape and
    types; site_problem checks how the elements refer to each other)."""
    if not isinstance(element, dict) or not isinstance(element.get("id"), str):
        return f"{where} must be an object with an id"
    kind = site_kind(element)
    extra = set(element) - _SITE_KEYS[kind]
    if extra:
        allowed = ", ".join(sorted(_SITE_KEYS[kind] - {"id"}))
        return f"{where} ({kind} {element['id']!r}) may have only {allowed}, not {', '.join(sorted(extra))}"
    if kind in ("node", "limit", "remote") and (
        not _SITE_ID.match(element["id"]) or _SITE_DEVICE.match(element["id"])
    ):
        return f"{where}.id must be a short lower-case name (letters, digits, _), not a device's"
    if element["id"] == "solar":
        # the saving split's player for the PV: a device or group of that
        # name would take its place in the ledger
        return f"{where}.id 'solar' is reserved"
    if kind == "node" and element["id"] == "ac":
        # the coordinator's saving split keeps the main bus's own total as "ac"
        return f"{where}.id 'ac' is reserved"
    if (
        kind == "node"
        and element.get("type") is not None
        and element["type"] not in _SITE_NODE_TYPES
    ):
        return f"{where}.type must be one of {', '.join(_SITE_NODE_TYPES + ('limit',))}"
    for key in ("max_import", "max_export"):
        v = element.get(key)
        if v is not None and not (_finite_number(v) and v >= 0):
            return f"{where}.{key} must be a finite number of W, >= 0"
    for key in ("efficiency_import", "efficiency_export"):
        v = element.get(key)
        if v is not None and not (_finite_number(v) and 0 < v <= 1):
            return f"{where}.{key} must be in (0, 1]"
    if kind == "limit" and element.get("max_import") is None and element.get("max_export") is None:
        return f"{where} needs max_import, max_export or both (W)"
    limits = element.get("limits")
    if limits is not None and (
        not isinstance(limits, list)
        or len(limits) > 16
        or not all(isinstance(x, str) and _SITE_ID.match(x) for x in limits)
    ):
        return f"{where}.limits must be a list of limit ids"
    if kind == "remote":
        solver = element["solver"]
        url = solver.get("url")
        if (
            set(solver) - {"url", "token_secret", "trusted_network"}
            or not isinstance(solver.get("trusted_network", False), bool)
            or not isinstance(url, str)
            or not re.match(r"^https?://[^\s/?#]+(/[^\s?#]*)?$", url)
            or len(url) > 256
        ):
            return f"{where}.solver must be {{url: http(s)://host[:port]/path, token_secret, trusted_network: true/false}}"
        secret = solver.get("token_secret")
        if secret is not None and not site_token_secret_usable(secret):
            return (
                f"{where}.solver.token_secret must name a secret of its own "
                "(lower case, digits, _), not one of EMHASS's secrets"
            )
        if from_request:
            return f"{where}: a remote solver is accepted from the configuration only, not with a request"
    if kind == "device":
        if element["id"] == "pv" and set(element) & {"solver", "group", "limits"}:
            return f"{where}: the PV is a forecast: it has a parent and nothing else (no solver, group or limits)"
        if element.get("solver", "emhass") not in _SITE_SOLVERS:
            return f"{where}.solver must be {', '.join(_SITE_SOLVERS)} (EMHASS's own models)"
        group = element.get("group")
        if group is not None and not (isinstance(group, str) and _SITE_ID.match(group)):
            return f"{where}.group must be a short lower-case name (letters, digits, _)"
        if group == "solar":
            return f"{where}.group 'solar' is reserved"
    return None


def site_problem(site, from_request: bool = False) -> str | None:
    """Why `site` cannot be used as given, or None.

    `site` is the house in one list, each element an object with an `id`:

    - "grid": the main meter, with `max_import` / `max_export` (W).
    - a node (any other lower-case id): `parent` (`grid` or another node),
      `type` (hybrid_inverter, inverter, panel, breaker,
      meter), `max_import` / `max_export` (W) on its connection to its parent,
      `efficiency_import` / `efficiency_export` (in (0, 1]).
    - a limit (`type: "limit"`): `max_import` and/or `max_export` (W) on what
      the devices tagged with it draw together.
    - a device (pv, battery0 or battery, deferrableN): `parent`, `group`
      and `limits` (limit ids); `solver` may only be emhass, the default.
    - a remote solver (`solver: {url, token_secret, trusted_network}`):
      `parent`, `limits`. Its token goes over plain http:// to another
      machine only with `trusted_network: true`.
      From the configuration only: a site sent with a request may not name
      one, so no request makes EMHASS call a host of its choosing.

    Ids are unique. Every element but the main meter and the limits has a
    `parent` - `grid` or a node - and the parents form a tree.

    Args:
        site: The `site` value.
        from_request: Whether it arrived with a request (runtime parameters).

    Returns:
        str | None: The problem, for the log; None when it can be used.
    """
    if not isinstance(site, list) or len(site) > _SITE_MAX_ELEMENTS:
        return f"site must be a list of at most {_SITE_MAX_ELEMENTS} elements"
    ids: dict[str, str] = {}
    for i, element in enumerate(site):
        problem = _site_element_problem(element, f"site[{i}]", from_request)
        if problem:
            return problem
        if element["id"] in ids:
            return f"site[{i}]: {element['id']!r} appears twice"
        ids[element["id"]] = site_kind(element)
    nodes = {e for e, kind in ids.items() if kind == "node"}
    limits = {e for e, kind in ids.items() if kind == "limit"}
    if len(nodes) + len(limits) > _SITE_MAX_ROWS:
        # each is a row of the coordinator's LP in every slot, and its solve
        # grows about with the cube of the rows: a bound on what a request costs
        return f"site may have at most {_SITE_MAX_ROWS} nodes and limits together"
    parent: dict[str, str] = {}
    for i, element in enumerate(site):
        if ids[element["id"]] in ("grid", "limit"):
            continue
        if "parent" not in element:
            # one tariff, one meter: every row but the meter and the limits
            # says where it is wired
            return f"site[{i}]: {element['id']!r} needs a parent ('grid', the main meter, or a node's id)"
        up = element["parent"]
        if not isinstance(up, str) or (up != _SITE_ROOT and up not in nodes):
            return f"site[{i}].parent must be {_SITE_ROOT!r} or a node's id"
        parent[element["id"]] = up
        unknown = [x for x in element.get("limits") or [] if x not in limits]
        if unknown:
            return f"site[{i}].limits names {unknown[0]!r}, which is no limit"
    for nid in nodes:
        seen, at = set(), nid
        while at != _SITE_ROOT:
            if at in seen:
                return f"site: the parents of {nid!r} form a loop"
            seen.add(at)
            at = parent[at]
    return None


def site_grid(site: list) -> dict:
    """The main meter's limits in `site`, as EMHASS's own keys (W)."""
    keys = {}
    for element in site:
        if site_kind(element) == "grid":
            if element.get("max_import") is not None:
                keys["maximum_power_from_grid"] = element["max_import"]
            if element.get("max_export") is not None:
                keys["maximum_power_to_grid"] = element["max_export"]
    return keys


def site_inverter(site: list, battery: bool = False, plant_conf: dict | None = None) -> dict | None:
    """The hybrid inverter in `site` that EMHASS's own model can hold, as the
    plant_conf keys that describe it, or None.

    EMHASS's model has one hybrid inverter, on the main meter, with the PV and
    the battery on its DC side: a node of type hybrid_inverter whose parent is
    the main meter, with nothing under it but the PV and the battery, the PV
    included. Inverters otherwise (a second one, one behind a panel) are the
    coordinator's only. `battery`: whether EMHASS has a battery
    (set_use_battery). Its model puts that battery on the inverter, so the
    inverter qualifies only if site does too - not for a battery site leaves
    out (planned on the main meter) or wires elsewhere.

    When `plant_conf` is given, an omitted max_export / max_import on the
    qualifying node is filled from that plant's cap so both backends share
    it (the coordinator treats omitted as unlimited).
    """
    under: dict[str, set[str]] = {}
    for element in site:
        if site_kind(element) in ("node", "device", "remote"):
            under.setdefault(element.get("parent", _SITE_ROOT), set()).add(element["id"])
    fits = [
        e
        for e in site
        if site_kind(e) == "node"
        and e.get("type") == "hybrid_inverter"
        and e.get("parent", _SITE_ROOT) == _SITE_ROOT
        and "pv" in under.get(e["id"], set())
        and under[e["id"]] <= {"pv", "battery", "battery0"}
        and bool(under[e["id"]] & {"battery", "battery0"}) == battery
    ]
    if len(fits) != 1:
        return None
    n = fits[0]
    if plant_conf is not None:
        for site_key, plant_key in (
            ("max_export", "inverter_ac_output_max"),
            ("max_import", "inverter_ac_input_max"),
        ):
            if n.get(site_key) is not None:
                continue
            cap = plant_conf.get(plant_key)
            if isinstance(cap, list | tuple):
                cap = cap[0] if len(cap) == 1 else None
            if isinstance(cap, bool) or not isinstance(cap, int | float):
                continue
            n[site_key] = cap
        # an omitted efficiency is the plant's, as an omitted rating is
        for site_key, plant_key in (
            ("efficiency_export", "inverter_efficiency_dc_ac"),
            ("efficiency_import", "inverter_efficiency_ac_dc"),
        ):
            eff = plant_conf.get(plant_key)
            if isinstance(eff, list | tuple):
                eff = eff[0] if len(eff) == 1 else None
            if isinstance(eff, str):  # a number as text, as the plant's own keys allow
                try:
                    eff = float(eff)
                except ValueError:
                    eff = None
            if n.get(site_key) is None and _finite_number(eff) and 0 < eff <= 1:
                n[site_key] = eff
    keys = {"inverter_is_hybrid": True}
    if n.get("max_export") is not None:
        keys["inverter_ac_output_max"] = n["max_export"]
    if n.get("max_import") is not None:
        keys["inverter_ac_input_max"] = n["max_import"]
    # null is omitted: 1.0, never None into plant_conf
    for key, plant_key in (
        ("efficiency_export", "inverter_efficiency_dc_ac"),
        ("efficiency_import", "inverter_efficiency_ac_dc"),
    ):
        keys[plant_key] = 1.0 if n.get(key) is None else n[key]
    return keys


def compile_site(
    site: list, plant_conf: dict, logger: logging.Logger, battery: bool = False
) -> bool:
    """Fill EMHASS's own keys in `plant_conf` from `site`: the main meter's
    limits (`site_grid`) and the hybrid inverter EMHASS's model can hold
    (`site_inverter`), so the default solver - on a fallback, or as the
    backend - plans the same meter and inverter. `site` wins where the keys
    disagree, with a warning. An omitted inverter rating is copied onto the
    node from `plant_conf` first. Returns whether an inverter was filled.
    """
    keys = site_grid(site)
    inverter = site_inverter(site, battery, plant_conf)
    from emhass.utils import InverterConfigError, validate_num_inverters

    try:
        inverters = validate_num_inverters(plant_conf)
    except InverterConfigError:
        inverters = 0  # reported where the configuration is checked
    if inverter is not None and inverters != 1:
        # Per-inverter keys (#1150) are lists; one inverter's values would replace them.
        logger.warning(
            "site: its hybrid inverter is not copied to EMHASS's keys, which describe "
            "number_of_inverters inverters of their own; only the main meter is"
        )
        inverter = None
    elif inverter is None and any(
        site_kind(e) == "node" and e.get("type") == "hybrid_inverter" for e in site
    ):
        logger.warning(
            "site: its hybrid inverter(s) are the coordinator's only; EMHASS's own model "
            "holds one inverter, on the main meter, with just the PV and the battery"
        )
    keys.update(inverter or {})
    changed = [
        k
        for k, v in keys.items()
        if k in plant_conf and plant_conf[k] not in (None, v) and k != "inverter_is_hybrid"
    ]
    if changed:
        logger.warning("site replaces %s", ", ".join(f"{k}={plant_conf[k]}" for k in changed))
    plant_conf.update(keys)
    return inverter is not None
