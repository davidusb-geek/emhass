import os
import re
import subprocess
from datetime import datetime

import requests


def _get_last_release_date_from_git() -> str:
    """
    Derive the last release date dynamically from the latest git tag.

    Returns the commit date (in ISO 8601 format) of the most recent tag.
    Falls back to an empty string if git commands fail, so callers can
    decide how to handle the absence of a release date.
    """
    try:
        last_tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"], text=True
        ).strip()
        last_tag_date = subprocess.check_output(
            ["git", "show", "-s", "--format=%cI", last_tag], text=True
        ).strip()
        return last_tag_date
    except Exception:
        return ""


def get_version() -> str:
    """Safely parse the version from __init__.py without importing."""
    # Goes up one level from 'scripts/', then into 'src/emhass/'
    init_path = os.path.join(os.path.dirname(__file__), "..", "src", "emhass", "__init__.py")
    with open(init_path, encoding="utf-8") as f:
        match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', f.read(), re.MULTILINE)
        if match:
            return match.group(1)
        raise RuntimeError(f"Unable to find version string in {init_path}")


# --- CONFIGURATION ---
REPO = "davidusb-geek/emhass"
LAST_RELEASE_DATE = os.getenv("LAST_RELEASE_DATE") or _get_last_release_date_from_git()
NEW_VERSION = get_version()
TOKEN = os.getenv("GITHUB_TOKEN")  # Add it using: $env:GITHUB_TOKEN="your_actual_token_here" on PS
# ---------------------

headers = {"Accept": "application/vnd.github.v3+json"}
if TOKEN:
    headers["Authorization"] = f"token {TOKEN}"


def get_merged_prs():
    url = f"https://api.github.com/repos/{REPO}/pulls?state=closed&per_page=100"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching PRs: {response.json().get('message')}")
        return []
    return response.json()


def generate_markdown():
    prs = get_merged_prs()
    last_release_dt = datetime.fromisoformat(LAST_RELEASE_DATE.replace("Z", "+00:00"))

    improvements = []
    fixes = []
    docs = []  # Added docs list

    for pr in prs:
        if not pr.get("merged_at"):
            continue

        merged_dt = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00"))

        if merged_dt > last_release_dt:
            title = pr["title"]
            user = pr["user"]["login"]
            labels = [lab["name"].lower() for lab in pr["labels"]]

            entry = f"- {title} (@{user})"

            # 1. Check for Documentation first
            if any(lab in labels for lab in ["documentation", "docs"]) or "docs" in title.lower():
                docs.append(entry)
            # 2. Check for Fixes
            elif any(lab in labels for lab in ["bug", "fix"]) or title.lower().startswith("fix"):
                fixes.append(entry)
            # 3. Default to Improvements
            else:
                improvements.append(entry)

    # Format the Output
    today = datetime.now().strftime("%Y-%m-%d")
    markdown = f"## {NEW_VERSION} - {today}\n"

    if improvements:
        markdown += "\n### Improvement\n" + "\n".join(improvements) + "\n"
    if docs:
        markdown += "\n### Documentation\n" + "\n".join(docs) + "\n"
    if fixes:
        markdown += "\n### Fix\n" + "\n".join(fixes) + "\n"

    return markdown


if __name__ == "__main__":
    changelog_chunk = generate_markdown()
    print(changelog_chunk)

    # Optional: Uncomment below to prepended automatically to your file
    # with open("CHANGELOG.md", "r+") as f:
    #     content = f.read()
    #     f.seek(0, 0)
    #     f.write(changelog_chunk + "\n" + content)
