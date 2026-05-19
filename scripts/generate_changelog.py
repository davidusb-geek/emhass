import os
from datetime import datetime

import requests

# --- CONFIGURATION ---
REPO = "davidusb-geek/emhass"
LAST_RELEASE_DATE = "2026-04-19T00:00:00Z"  # ISO format of your last release date
NEW_VERSION = "0.17.3"
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
            labels = [l["name"].lower() for l in pr["labels"]]

            entry = f"- {title} (@{user})"

            # 1. Check for Documentation first
            if any(l in labels for l in ["documentation", "docs"]) or "docs" in title.lower():
                docs.append(entry)
            # 2. Check for Fixes
            elif any(l in labels for l in ["bug", "fix"]) or title.lower().startswith("fix"):
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
