"""Documentation examples that call the EMHASS API carry the API token, and only they do.

The web API requires ``Authorization: Bearer <token>`` (docs/api_authentication.md). An example that
omits it teaches a call that fails with 401. An example that sends the EMHASS token to another service
(for example a VictoriaMetrics check that uses ``curl -u``) leaks the token and breaks that check.
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
API_PATH = re.compile(r"/(action/|set-config|get-config|get-json|api/v1/)")
FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
# -u / --user: Basic auth for another service that also uses /api/v1/ paths (VictoriaMetrics)
CREDENTIALS = re.compile(
    r"(?i)authorization|bearer|EMHASS_API_TOKEN|YOUR_API_TOKEN|\s-u\s|--user\b"
)
BEARER = re.compile(r"(?i)authorization[^\n]{0,40}bearer")


def curl_commands():
    """(file, line, command) for every curl in a fenced code block (a command runs until the next
    curl) and every inline ``curl ...`` span in the prose."""
    for path in sorted(DOCS.rglob("*.md")):
        if path.name == "api_authentication.md":
            continue
        text = path.read_text(encoding="utf-8")
        outside = FENCE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
        for span in re.finditer(r"`(curl [^`\n]*)`", outside):
            yield (
                path.relative_to(DOCS).as_posix(),
                outside[: span.start()].count("\n") + 1,
                span.group(1),
            )
        for block in FENCE.finditer(text):
            lines = block.group(1).split("\n")
            starts = [i for i, line in enumerate(lines) if re.search(r"\bcurl\b", line)]
            for n, i in enumerate(starts):
                end = starts[n + 1] if n + 1 < len(starts) else len(lines)
                yield (
                    path.relative_to(DOCS).as_posix(),
                    text[: block.start()].count("\n") + 2 + i,
                    "\n".join(lines[i:end]),
                )


class TestDocsApiExamples(unittest.TestCase):
    def test_the_enumeration_finds_the_examples(self):
        self.assertGreater(len(list(curl_commands())), 50)

    def test_every_api_example_carries_credentials(self):
        missing = [
            f"{name}:{line}"
            for name, line, cmd in curl_commands()
            if API_PATH.search(cmd) and "healthz" not in cmd and not CREDENTIALS.search(cmd)
        ]
        self.assertEqual(missing, [], "API examples without the Authorization header")

    def test_the_emhass_token_is_never_sent_to_another_service(self):
        stray = [
            f"{name}:{line}"
            for name, line, cmd in curl_commands()
            if BEARER.search(cmd) and not API_PATH.search(cmd)
        ]
        self.assertEqual(stray, [], "Bearer header on a command that does not call the EMHASS API")

    def test_basic_auth_commands_do_not_also_send_a_bearer_header(self):
        both = [
            f"{name}:{line}"
            for name, line, cmd in curl_commands()
            if re.search(r"\s-u\s", cmd) and BEARER.search(cmd)
        ]
        self.assertEqual(both, [])


class TestOpenapiGeneratorLeavesNoTokenFile(unittest.TestCase):
    def test_importing_the_web_server_for_the_spec_writes_no_token(self):
        with tempfile.TemporaryDirectory() as data:
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith("EMHASS_API") and k != "SUPERVISOR_TOKEN"
            }
            env.update(DATA_PATH=data + os.sep, PYTHONPATH=str(ROOT / "src"))
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
