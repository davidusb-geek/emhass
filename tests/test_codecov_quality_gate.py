"""The CodeCov workflow must wait for the SonarQube Cloud quality gate and let its verdict fail the job.

These tests check the workflow configuration only. The pass, fail and timeout behaviour belongs to
SonarScanner and SonarQube Cloud, needs repository credentials and is not exercised here.
"""

import pathlib

import pytest
import yaml

WORKFLOW = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / "codecov.yaml"


@pytest.fixture(name="job")
def fixture_job():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["run"]


def _step_index(job, action):
    found = [
        i for i, step in enumerate(job["steps"]) if step.get("uses", "").startswith(action + "@")
    ]
    assert len(found) == 1, f"expected exactly one {action} step"
    return found[0]


def test_scan_waits_for_the_quality_gate_with_a_bounded_timeout(job):
    scan = job["steps"][_step_index(job, "SonarSource/sonarqube-scan-action")]
    gate_args = [
        arg for arg in scan["with"]["args"].split() if arg.startswith("-Dsonar.qualitygate.")
    ]
    assert gate_args == ["-Dsonar.qualitygate.wait=true", "-Dsonar.qualitygate.timeout=300"]


def test_a_failed_scan_fails_the_job(job):
    scan = job["steps"][_step_index(job, "SonarSource/sonarqube-scan-action")]
    assert "continue-on-error" not in scan
    assert "continue-on-error" not in job


def test_coverage_is_uploaded_before_the_gate_can_fail_the_job(job):
    upload = _step_index(job, "codecov/codecov-action")
    assert upload < _step_index(job, "SonarSource/sonarqube-scan-action")
