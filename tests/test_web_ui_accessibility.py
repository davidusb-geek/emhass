"""Static accessibility checks for the web UI templates and stylesheet (#1278)."""

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

UI_ROOT = Path(__file__).resolve().parents[1] / "src" / "emhass"
TEMPLATES = [UI_ROOT / "templates" / "index.html", UI_ROOT / "templates" / "configuration.html"]
FRAGMENTS = [UI_ROOT / "static" / "configuration_list.html", UI_ROOT / "static" / "advanced.html"]
CSS = (UI_ROOT / "static" / "style.css").read_text(encoding="utf-8")
NAMED_COLORS = {"darkblue": "#00008b", "white": "#ffffff", "black": "#000000"}


class _Collector(HTMLParser):
    """Collect every element with its attributes, its ancestors and the labels' ``for`` targets."""

    def __init__(self):
        super().__init__()
        self.stack = []
        self.elements = []  # (tag, attrs, ancestor tags)
        self.label_targets = set()
        self.svg_owners = []  # (svg attrs, attrs of the enclosing a/button or None)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.elements.append((tag, attrs, [t for t, _ in self.stack]))
        if tag == "svg":
            owner = next((a for t, a in reversed(self.stack) if t in ("a", "button")), None)
            self.svg_owners.append((attrs, owner))
        if tag == "label" and "for" in attrs:
            self.label_targets.add(attrs["for"])
        if tag not in (
            "input",
            "br",
            "img",
            "use",
            "link",
            "meta",
            "hr",
            "area",
            "col",
            "source",
            "wbr",
        ):
            self.stack.append((tag, attrs))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break


def _parse(path):
    parser = _Collector()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def _luminance(hex_colour):
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(foreground, background):
    lighter, darker = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_decorative_svgs_are_hidden_and_icon_controls_are_named(path):
    """An icon-only control is named on the control itself, not on its decorative svg."""
    svgs = _parse(path).svg_owners
    assert svgs
    for svg, owner in svgs:
        assert svg.get("aria-hidden") == "true", svg
        assert owner is not None, svg
        assert (owner.get("title") or owner.get("aria-label") or "").strip(), svg


@pytest.mark.parametrize(
    ("path", "control_id"),
    [(TEMPLATES[0], "basicOrAdvanced"), (TEMPLATES[1], "json-toggle")],
    ids=["basicOrAdvanced", "json-toggle"],
)
def test_click_only_toggles_are_buttons(path, control_id):
    """A control that only has a click handler must be keyboard focusable: a button."""
    matches = [(t, a) for t, a, _ in _parse(path).elements if a.get("id") == control_id]
    assert [(tag, attrs.get("type")) for tag, attrs in matches] == [("button", "button")]
    assert (matches[0][1].get("title") or "").strip()


@pytest.mark.parametrize("path", FRAGMENTS, ids=lambda p: p.name)
def test_form_controls_have_a_label(path):
    """Every control has a non-empty accessible name; an empty <label> wrapper is not one."""
    page = _parse(path)
    controls = [
        attrs
        for tag, attrs, _ in page.elements
        if tag in ("select", "textarea")
        or (tag == "input" and attrs.get("type", "text") in ("number", "text", "checkbox"))
    ]
    assert controls
    for attrs in controls:
        assert (attrs.get("aria-label") or "").strip(), attrs


def test_mark_text_is_readable_on_its_background():
    rule = re.search(r"(?m)^mark\s*\{([^}]*)\}", CSS).group(1)

    def colour(prop):
        value = re.search(rf"(?<![\w-]){prop}\s*:\s*([^;]+);", rule).group(1).strip().lower()
        return NAMED_COLORS.get(value, value)

    assert _contrast(colour("color"), colour("background-color")) >= 4.5


def test_loading_spinner_stops_for_reduced_motion():
    base = re.search(r"(?m)^\.loading\s*\{", CSS)
    media = re.search(
        r"@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{(.*?)\n\}", CSS, flags=re.S
    )
    assert re.search(r"\.loading\s*\{[^}]*animation:\s*none", media.group(1))
    assert media.start() > base.start(), (
        "the media rule must follow the base rule to win the cascade"
    )
