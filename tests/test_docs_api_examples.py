"""Documentation examples: EMHASS API calls carry the API token, and nothing else receives it.

The web API requires ``Authorization: Bearer <token>`` (docs/api_authentication.md). Checked here, in
every ``curl`` (fenced or inline) and every Home Assistant ``rest_command`` ``url:`` block of the docs:

* a call to the EMHASS API carries a Bearer header (Basic auth does not count: EMHASS does not accept it);
* the EMHASS Bearer header is never sent to another service (VictoriaMetrics shares the ``/api/v1/``
  paths, so the service is told apart by its port, not by the path) and never beside Basic auth.

Node-RED and similar settings written as prose are not machine-checked.
"""

import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
API_PATH = re.compile(r"/(action/|set-config|get-config|get-json|api/v1/|configuration\b)")
# Other services the docs talk to (VictoriaMetrics, InfluxDB, Home Assistant, MQTT)
OTHER_SERVICE = re.compile(r":(8428|8086|8123|1883)\b|victoria|influx", re.I)
FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
BEARER = re.compile(r"(?i)authorization[^\n]{0,60}bearer|authorization:\s*!secret\s+emhass_bearer")
BASIC_AUTH = re.compile(r"(?<![\w-])(?:-u\S*|--user(?:=|\s)\S*)")
# The one deliberate example of a call WITHOUT the token (shows the 401)
EXEMPT = {("api_authentication.md", "last-run")}


def is_emhass_call(command: str) -> bool:
    return (
        bool(API_PATH.search(command))
        and not OTHER_SERVICE.search(command)
        and "healthz" not in command
    )


def exempt(name: str, command: str) -> bool:
    return any(name == f and key in command for f, key in EXEMPT)


def curl_commands():
    """(file, line, command) for every curl in a fenced code block (a command runs until the next
    curl) and every inline ``curl ...`` span in the prose."""
    for path in sorted(DOCS.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        name = path.relative_to(DOCS).as_posix()
        outside = FENCE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
        for span in re.finditer(r"`(curl [^`\n]*)`", outside):
            yield name, outside[: span.start()].count("\n") + 1, span.group(1)
        for block in FENCE.finditer(text):
            lines = block.group(1).split("\n")
            starts = [i for i, line in enumerate(lines) if re.search(r"\bcurl\b", line)]
            for n, i in enumerate(starts):
                end = starts[n + 1] if n + 1 < len(starts) else len(lines)
                yield name, text[: block.start()].count("\n") + 2 + i, "\n".join(lines[i:end])


def rest_command_entries():
    """(file, line, block) for each ``url:`` in a fenced block, with the lines up to the next ``url:``."""
    for path in sorted(DOCS.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        name = path.relative_to(DOCS).as_posix()
        for block in FENCE.finditer(text):
            lines = block.group(1).split("\n")
            starts = [i for i, line in enumerate(lines) if re.match(r"\s*url:\s*\S", line)]
            for n, i in enumerate(starts):
                end = starts[n + 1] if n + 1 < len(starts) else len(lines)
                yield name, text[: block.start()].count("\n") + 2 + i, "\n".join(lines[i:end])


class TestDocsApiExamples(unittest.TestCase):
    def test_the_enumeration_finds_the_examples(self):
        self.assertGreater(len(list(curl_commands())), 50)
        self.assertGreater(sum(1 for _, _, b in rest_command_entries() if is_emhass_call(b)), 10)

    def test_every_curl_to_the_api_carries_the_bearer_header(self):
        missing = [
            f"{name}:{line}"
            for name, line, cmd in curl_commands()
            if is_emhass_call(cmd) and not BEARER.search(cmd) and not exempt(name, cmd)
        ]
        self.assertEqual(missing, [], "EMHASS API curl examples without the Bearer header")

    def test_every_rest_command_to_the_api_carries_the_bearer_header(self):
        missing = [
            f"{name}:{line}"
            for name, line, block in rest_command_entries()
            if is_emhass_call(block) and not BEARER.search(block)
        ]
        self.assertEqual(missing, [], "EMHASS API rest_command examples without the Bearer header")

    def test_the_emhass_token_is_never_sent_to_another_service(self):
        stray = [
            f"{name}:{line}"
            for name, line, cmd in [*curl_commands(), *rest_command_entries()]
            if BEARER.search(cmd) and OTHER_SERVICE.search(cmd)
        ]
        self.assertEqual(stray, [], "Bearer header on a command for another service")

    def test_basic_auth_is_never_combined_with_the_bearer_header(self):
        both = [
            f"{name}:{line}"
            for name, line, cmd in curl_commands()
            if BASIC_AUTH.search(cmd) and BEARER.search(cmd)
        ]
        self.assertEqual(both, [])

    def test_the_lint_recognises_the_shapes_it_exists_to_catch(self):
        shapes_that_must_be_flagged = [
            'curl -H "Authorization: Bearer $EMHASS_API_TOKEN" --user vm:pw http://h:8428/api/v1/status/tsdb',
            'curl -H "Authorization: Bearer $EMHASS_API_TOKEN" -uvm:pw http://h:8428/api/v1/status/tsdb',
            'curl -H "Authorization: Bearer $EMHASS_API_TOKEN" http://h:8428/api/v1/query?query=up',
        ]
        for cmd in shapes_that_must_be_flagged:
            with self.subTest(cmd=cmd):
                self.assertTrue(
                    BEARER.search(cmd) and (OTHER_SERVICE.search(cmd) or BASIC_AUTH.search(cmd))
                )
        # Basic auth is not an EMHASS credential
        basic = "curl -u me:pw http://localhost:5000/action/dayahead-optim"
        self.assertTrue(is_emhass_call(basic) and not BEARER.search(basic))
        # a bare environment variable name is not a credential either
        bare = "EMHASS_API_TOKEN=x curl http://localhost:5000/action/dayahead-optim"
        self.assertTrue(is_emhass_call(bare) and not BEARER.search(bare))


class TestOpenapiGeneratorLeavesNoTokenFile(unittest.TestCase):
    def test_importing_the_web_server_for_the_spec_writes_no_token(self):
        with tempfile.TemporaryDirectory() as data:
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith("EMHASS_API") and k != "SUPERVISOR_TOKEN"
            }
            # a stray "true" must not switch generation back on, and no real secrets/options file may be read
            env["EMHASS_API_TOKEN_AUTOGENERATE"] = "true"
            env.update(
                DATA_PATH=data + os.sep,
                PYTHONPATH=str(ROOT / "src"),
                SECRETS_PATH=str(pathlib.Path(data) / "no_secrets.yaml"),
                OPTIONS_PATH=str(pathlib.Path(data) / "no_options.json"),
            )
            code = (
                "import sys; sys.path.insert(0, 'scripts'); import generate_openapi as g; "
                "g.discovered_routes()"
            )
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr[-500:])
            self.assertEqual(list(pathlib.Path(data).iterdir()), [], "a token file was written")


if __name__ == "__main__":
    unittest.main()
