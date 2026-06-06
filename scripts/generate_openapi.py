#!/usr/bin/env python3
"""Generate src/emhass/static/openapi.json from the EMHASS route map + schema sources.

Stdlib only. Run: python scripts/generate_openapi.py
"""

import argparse
import json
import re
from pathlib import Path

# HTML/UI + framework routes excluded from the JSON-API contract.
# `/static/{filename}` is Quart's auto-registered asset endpoint (not an API surface);
# the others are the browser config/landing/template pages.
SKIP = {"/", "/index", "/template", "/configuration", "/static/{filename}"}

# path -> set of documented methods (used by both the guard and the path assembly).
CURATED = {
    "/get-config": {"GET"},
    "/get-config/defaults": {"GET"},
    "/set-config": {"POST"},
    "/get-json": {"POST"},
    "/action/{action_name}": {"POST"},
    "/api/v1/last-run": {"GET"},
    "/healthz": {"GET"},
}

_ATOMIC = {
    "string": {"type": "string"},
    "int": {"type": "integer"},
    "float": {"type": "number"},
    "boolean": {"type": "boolean"},
    "time": {"type": "string"},
    "object": {"type": "object"},
}


def _base_schema(base: str, param: dict) -> dict:
    if base == "select":
        return {"type": "string", "enum": list(param.get("select_options", []))}
    if base in _ATOMIC:
        return dict(_ATOMIC[base])
    raise SystemExit(
        f"generate_openapi: unknown input base type {base!r} (param input={param.get('input')!r})"
    )


def _input_to_schema(input_str: str, param: dict) -> dict:
    """Map a param_definitions `input` string + param dict to an OpenAPI schema.

    For array.* inputs, the param `default_value` is the per-element scalar template,
    so it is placed on items.default; scalars get `default` directly. title/description/
    x-unit annotate the outermost property.
    """
    parts = input_str.split(".")
    leaf = _base_schema(parts[-1], param)
    dv = param.get("default_value", None)
    if dv is not None:
        leaf["default"] = dv
    schema = leaf
    for _ in parts[:-1]:  # one wrap per leading "array"
        schema = {"type": "array", "items": schema}
    if param.get("friendly_name"):
        schema["title"] = param["friendly_name"]
    desc = (param.get("Description") or "").strip()
    unit = param.get("unit")
    if unit and unit not in ("none", ""):
        schema["x-unit"] = unit
        desc = (desc + f" (unit: {unit})").strip()
    if desc:
        schema["description"] = desc
    return schema


def build_config_component(param_defs: dict) -> dict:
    """Flatten param_definitions sections into a single OpenAPI object schema.

    Raises if a param key appears in more than one section (don't silently overwrite).
    """
    props: dict = {}
    for params in param_defs.values():
        for key, param in params.items():
            if key in props:
                raise SystemExit(f"generate_openapi: duplicate param key {key!r} across sections")
            props[key] = _input_to_schema(param["input"], param)
    return {"type": "object", "properties": props}


def _normalise_path(rule: str) -> str:
    """Convert a Werkzeug rule to OpenAPI path style: ``<x>`` / ``<conv:x>`` -> ``{x}``."""
    return re.sub(r"<(?:[^:<>]+:)?([^<>]+)>", r"{\1}", rule)


def discovered_routes() -> set[tuple[str, str]]:
    """Return {(path, METHOD)} from the live Quart url_map. Normalises <x> -> {x}."""
    from emhass.web_server import app

    out: set[tuple[str, str]] = set()
    for rule in app.url_map.iter_rules():
        path = _normalise_path(rule.rule)
        for method in rule.methods or set():
            if method in ("HEAD", "OPTIONS"):
                continue
            out.add((path, method))
    return out


def assert_no_undocumented(routes: set, curated: dict, skip: set) -> None:
    for path, method in routes:
        if path in skip:
            continue
        if path not in curated or method not in curated[path]:
            raise SystemExit(
                f"generate_openapi: undocumented route {method} {path} — add to CURATED or SKIP"
            )


_REPO = Path(__file__).resolve().parents[1]
_PARAM_DEFS = _REPO / "src" / "emhass" / "static" / "data" / "param_definitions.json"
_LAST_RUN_SCHEMA = _REPO / "docs" / "api" / "v1" / "last-run.schema.json"
_HEALTHZ_SCHEMA = _REPO / "docs" / "api" / "healthz.schema.json"
_OUT = _REPO / "src" / "emhass" / "static" / "openapi.json"

_PLAN_OUTPUT_DOC = "https://github.com/davidusb-geek/emhass/blob/master/docs/plan_output_schema.md"


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("$schema", None)  # not meaningful inside an openapi component
    data.pop("$id", None)
    return data


def _schema_version() -> str:
    from emhass.command_line import EMHASS_SCHEMA_VERSION

    return EMHASS_SCHEMA_VERSION


def build_spec(routes: set | None = None) -> dict:
    # `routes` lets the caller pass an already-discovered set (avoids importing
    # emhass.web_server twice per generation); falls back to discovery when called standalone.
    if routes is None:
        routes = discovered_routes()
    param_defs = json.loads(_PARAM_DEFS.read_text(encoding="utf-8"))
    components = {
        "Config": build_config_component(param_defs),
        "LastRun": _load_json(_LAST_RUN_SCHEMA),
        "Healthz": _load_json(_HEALTHZ_SCHEMA),
    }
    config_ref = {"$ref": "#/components/schemas/Config"}

    def json_ct(schema: dict) -> dict:
        return {"content": {"application/json": {"schema": schema}}}

    paths = {
        "/get-config": {
            "get": {
                "summary": "Current config",
                "responses": {"200": {"description": "Config JSON", **json_ct(config_ref)}},
            }
        },
        "/get-config/defaults": {
            "get": {
                "summary": "Default config",
                "responses": {"200": {"description": "Default config JSON", **json_ct(config_ref)}},
            }
        },
        "/set-config": {
            "post": {
                "summary": "Save config",
                "requestBody": json_ct(config_ref),
                "responses": {
                    "200": {"description": "Saved"},
                    "400": {"description": "Empty/invalid config"},
                    "500": {"description": "Save failure"},
                },
            }
        },
        "/get-json": {
            "post": {
                "summary": "Convert legacy YAML config to JSON",
                "requestBody": {"content": {"text/plain": {"schema": {"type": "string"}}}},
                "responses": {
                    "200": {"description": "Config JSON", **json_ct(config_ref)},
                    "400": {"description": "YAML parse failure"},
                    "500": {"description": "Conversion failure"},
                },
            }
        },
        "/action/{action_name}": {
            "post": {
                "summary": "Run an EMHASS action",
                # externalDocs belongs on the Operation Object in OpenAPI 3.1
                # (not on a Response Object — strict validators reject the latter).
                "externalDocs": {
                    "description": "Plan output field reference",
                    "url": _PLAN_OUTPUT_DOC,
                },
                "parameters": [
                    {
                        "name": "action_name",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "requestBody": json_ct({"type": "object", "additionalProperties": True}),
                "responses": {
                    "200": {
                        "description": "Optimization plan",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    },
                    "400": {"description": "Action failure"},
                },
            }
        },
        "/api/v1/last-run": {
            "get": {
                "summary": "Most recent run metadata",
                "responses": {
                    "200": {
                        "description": "Last-run envelope",
                        **json_ct({"$ref": "#/components/schemas/LastRun"}),
                    }
                },
            }
        },
        "/healthz": {
            "get": {
                "summary": "Liveness/readiness probe",
                "responses": {
                    "200": {
                        "description": "Ready",
                        **json_ct({"$ref": "#/components/schemas/Healthz"}),
                    },
                    "503": {
                        "description": "Not ready",
                        **json_ct({"$ref": "#/components/schemas/Healthz"}),
                    },
                },
            }
        },
    }

    # filter CURATED paths down to what actually exists in url_map (e.g. /healthz pre-AC-4)
    live = {p for p, _ in routes}
    paths = {p: v for p, v in paths.items() if p in live}

    return {
        "openapi": "3.1.0",
        "info": {
            "title": "EMHASS API",
            "version": _schema_version(),
            "description": (
                "Machine-readable contract for the EMHASS JSON API. `default` values are "
                "sourced from param_definitions.json (the maintainer-declared source of "
                "truth); runtime currently loads config_defaults.json, which is kept aligned "
                "but not enforced in code."
            ),
        },
        "paths": paths,
        "components": {"schemas": components},
    }


def generate() -> dict:
    routes = discovered_routes()
    assert_no_undocumented(routes, CURATED, SKIP)
    return build_spec(routes)


def main(argv=None) -> int:
    argparse.ArgumentParser(description="Generate EMHASS openapi.json").parse_args(argv)
    spec = generate()
    _OUT.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {_OUT.relative_to(_REPO)} ({len(spec['paths'])} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
