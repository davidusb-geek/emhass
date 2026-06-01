#!/usr/bin/env python3
"""Generate src/emhass/static/openapi.json from the EMHASS route map + schema sources.

Stdlib only. Run: python scripts/generate_openapi.py
"""

import argparse
import json
from pathlib import Path

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
    for _ in parts[:-1]:          # one wrap per leading "array"
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
